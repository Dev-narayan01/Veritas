"""``/api/analyses`` endpoints: create, stream progress, read, export, delete."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from .. import db
from ..config import settings
from ..parsers.base import ALL_EXT, ParseError, safe_name
from ..pipeline import InputDocument, Job, start_worker
from ..ratelimit import SlidingWindowLimiter
from ..schemas import (
    AnalysisOptions,
    AnalysisResult,
    AnalysisSummary,
    AnalyzeResponse,
    ErrorBody,
    ErrorResponse,
    ProgressEvent,
)
from ..logging_setup import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/analyses", tags=["analyses"])

limiter = SlidingWindowLimiter(settings.rate_limit_per_min)

_MAGIC_CHECK = {".pdf": (b"%PDF",)}


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=ErrorResponse(error=ErrorBody(code=code, message=message)).model_dump())


async def _save_upload(upload: UploadFile, analysis_id: str) -> tuple[InputDocument, int]:
    """Persist one upload, enforcing size and type limits before anything parses it."""
    name = safe_name(upload.filename or "upload")
    suffix = Path(name).suffix.lower()
    if suffix not in ALL_EXT and suffix != ".zip":
        raise _error(415, "unsupported_type", f"Unsupported file type: {suffix or 'none'}")
    target_dir = settings.uploads_path / analysis_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid.uuid4().hex[:8]}_{name}"
    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    with target.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                handle.close()
                target.unlink(missing_ok=True)
                raise _error(
                    413, "file_too_large", f"{name} exceeds the {settings.max_upload_mb} MB limit"
                )
            handle.write(chunk)
    if size == 0:
        target.unlink(missing_ok=True)
        raise _error(422, "empty_file", f"{name} is empty")
    expected = _MAGIC_CHECK.get(suffix)
    if expected:
        with target.open("rb") as handle:
            head = handle.read(8)
        if not any(head.startswith(m) for m in expected):
            target.unlink(missing_ok=True)
            raise _error(415, "type_mismatch", f"{name} does not look like a {suffix} file")
    return InputDocument(filename=name, path=target, is_archive=suffix == ".zip"), size


@router.post("", response_model=AnalyzeResponse, status_code=202)
async def create_analysis(
    request: Request,
    files: list[UploadFile] = File(default_factory=list),
    text: str | None = Form(default=None),
    url: str | None = Form(default=None),
    options: str | None = Form(default=None),
) -> AnalyzeResponse:
    """Accept documents (multipart), pasted text or a URL and queue an analysis."""
    client = request.client.host if request.client else "unknown"
    allowed, retry_after = limiter.allow(client)
    if not allowed:
        raise _error(429, "rate_limited", f"Too many analyses; retry in {retry_after} seconds")

    try:
        parsed_options = AnalysisOptions.model_validate_json(options) if options else AnalysisOptions()
    except ValueError as exc:
        raise _error(422, "invalid_options", f"options could not be parsed: {exc}") from exc

    if len(files) > settings.max_files_per_request:
        raise _error(413, "too_many_files", f"At most {settings.max_files_per_request} files per request")
    if not files and not (text and text.strip()) and not url:
        raise _error(422, "no_input", "Provide at least one file, some text or a URL")

    analysis_id = str(uuid.uuid4())
    documents: list[InputDocument] = []
    try:
        for upload in files:
            document, _ = await _save_upload(upload, analysis_id)
            documents.append(document)
        if url:
            from ..parsers.url import fetch_text, validate_url

            validate_url(url)
            fetched = fetch_text(url)
            documents.append(InputDocument(filename=fetched.filename, text=fetched.text))
        if text and text.strip():
            documents.append(InputDocument(filename="pasted-text.txt", text=text))
    except ParseError as exc:
        raise _error(exc.status, exc.code, exc.message) from exc
    except Exception as exc:  # noqa: BLE001 - never leak a traceback to the client
        # Anything else here means a fetch/IO path failed in a way we did not anticipate.
        # The log keeps the traceback; the client gets a message naming the input instead of
        # "Something went wrong on the server", which tells them nothing and looks like a bug.
        log.exception("input acquisition failed for %s", url or [d.filename for d in documents])
        raise _error(
            422,
            "input_failed",
            f"Could not read that input ({type(exc).__name__}). Try a file or pasted text.",
        ) from exc

    options_dict = parsed_options.model_dump()
    with db.session() as session:
        row = db.AnalysisRow(
            id=analysis_id,
            status="queued",
            mode=parsed_options.mode,
            options=options_dict,
            filenames=[d.filename for d in documents],
            retain=parsed_options.retain,
        )
        session.add(row)
        session.commit()
    db.add_event(analysis_id, "queued", 0.0, "analysis queued")

    job = Job(
        analysis_id=analysis_id,
        documents=documents,
        options={**options_dict, "exclusions": options_dict.get("exclusions", {})},
    )
    start_worker(job)
    return AnalyzeResponse(id=analysis_id, status="queued")


@router.get("/{analysis_id}/events")
async def stream_events(analysis_id: str) -> StreamingResponse:
    """Server-sent events with a polling fallback on the client."""
    _require_row(analysis_id)

    async def generator():  # noqa: ANN202 - async generator for SSE
        sent = 0
        idle = 0
        while True:
            events = db.get_events(analysis_id, after=sent)
            for event in events:
                sent = event["seq"]
                payload = ProgressEvent(
                    stage=event["stage"], progress=event["progress"], message=event.get("message", "")
                )
                yield f"event: progress\ndata: {json.dumps(payload.model_dump())}\n\n"
                if event["stage"] in {"done", "error"}:
                    yield f"event: {event['stage']}\ndata: {json.dumps(payload.model_dump())}\n\n"
                    return
            if not events:
                idle += 1
                if idle > 600:  # ~5 minutes of silence: stop, the client will poll
                    yield "event: timeout\ndata: {}\n\n"
                    return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _require_row(analysis_id: str) -> db.AnalysisRow:
    with db.session() as session:
        row = session.get(db.AnalysisRow, analysis_id)
    if row is None:
        raise _error(404, "not_found", "No analysis with that id")
    return row


@router.get("", response_model=list[AnalysisSummary])
def list_analyses(limit: int = 50, offset: int = 0) -> list[AnalysisSummary]:
    """History, newest first."""
    from sqlalchemy import select

    with db.session() as session:
        rows = session.execute(
            select(db.AnalysisRow).order_by(db.AnalysisRow.created_at.desc()).offset(offset).limit(min(limit, 200))
        ).scalars().all()
    return [
        AnalysisSummary(
            id=row.id,
            created_at=row.created_at.isoformat(),
            status=row.status,
            filenames=row.filenames or [],
            mode=row.mode,
            verdicts=row.verdicts or [],
            similarity=row.similarity,
        )
        for row in rows
    ]


@router.get("/{analysis_id}")
def get_analysis(analysis_id: str) -> dict[str, Any]:
    """Full result, or a status object while the analysis is still running."""
    row = _require_row(analysis_id)
    if row.status == "done" and row.result:
        if not row.retain and row.result:
            for document in row.result.get("documents", []):
                document["text"] = ""  # retain=false: text was purged, spans remain
        return row.result
    return {"id": row.id, "status": row.status, "error": row.error}


def _as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` as a dict, tolerating None and already-decoded payloads."""
    if isinstance(value, dict):
        return value
    return {}


@router.get("/{analysis_id}/export.json")
def export_json(analysis_id: str) -> Response:
    row = _require_row(analysis_id)
    if row.result is None:
        raise _error(409, "not_ready", "The analysis has not finished yet")
    return Response(
        content=json.dumps(row.result, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="veritas-{analysis_id}.json"'},
    )


@router.get("/{analysis_id}/export.csv")
def export_csv(analysis_id: str) -> Response:
    """Three tables in one file, each with its own header row.

    A single wide header cannot describe documents, sentences and plagiarism matches at
    once (the earlier version emitted one document header and then sentence rows under
    it, so every sentence column landed under the wrong name). Sections are separated by
    a blank line and each carries a ``section`` column, so a spreadsheet import stays
    readable.
    """
    row = _require_row(analysis_id)
    if row.result is None:
        raise _error(409, "not_ready", "The analysis has not finished yet")
    result = _as_dict(row.result)
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(
        ["section", "file", "word_count", "verdict", "probability", "t_human", "t_ai", "similarity_percent", "corpus_size"]
    )
    for document in result.get("documents", []):
        ai = _as_dict(document.get("ai"))
        plag = _as_dict(document.get("plagiarism"))
        thresholds = _as_dict(ai.get("thresholds"))
        writer.writerow(
            [
                "document",
                document.get("filename"),
                document.get("words"),
                ai.get("verdict"),
                ai.get("probability"),
                thresholds.get("t_human"),
                thresholds.get("t_ai"),
                plag.get("similarity_percent"),
                plag.get("corpus_size"),
            ]
        )

    writer.writerow([])
    writer.writerow(["section", "file", "sentence_start", "sentence_end", "ai_prob", "text"])
    for document in result.get("documents", []):
        for sentence in _as_dict(document.get("ai")).get("sentences") or []:
            writer.writerow(
                [
                    "sentence",
                    document.get("filename"),
                    sentence.get("start"),
                    sentence.get("end"),
                    sentence.get("ai_prob"),
                    (sentence.get("text") or "")[:200],
                ]
            )

    writer.writerow([])
    writer.writerow(
        ["section", "file", "source_title", "match_type", "score", "words", "query_start", "query_end", "in_quote", "in_bibliography"]
    )
    for document in result.get("documents", []):
        for match in _as_dict(document.get("plagiarism")).get("matches") or []:
            span = match.get("query_span") or [None, None]
            writer.writerow(
                [
                    "plagiarism_match",
                    document.get("filename"),
                    match.get("source_title"),
                    match.get("match_type"),
                    match.get("score"),
                    match.get("words"),
                    span[0],
                    span[1],
                    match.get("in_quote"),
                    match.get("in_bibliography"),
                ]
            )
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="veritas-{analysis_id}.csv"'},
    )


@router.get("/{analysis_id}/report.pdf")
def export_pdf(analysis_id: str) -> Response:
    from ..reports.pdf_report import build_report

    row = _require_row(analysis_id)
    if row.result is None:
        raise _error(409, "not_ready", "The analysis has not finished yet")
    payload = build_report(AnalysisResult.model_validate(row.result))
    return Response(
        content=payload,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="veritas-{analysis_id}.pdf"'},
    )


@router.delete("/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str) -> Response:
    _require_row(analysis_id)
    db.delete_analysis(analysis_id)
    import shutil

    shutil.rmtree(settings.uploads_path / analysis_id, ignore_errors=True)
    return Response(status_code=204)
