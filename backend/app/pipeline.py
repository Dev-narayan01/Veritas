"""The analysis pipeline that runs in a worker thread for each request.

Order: parse -> integrity -> AI -> plagiarism -> batch -> report.  Each stage emits a
real progress event, each optional feature is skipped with a note when its data is
missing, and ``retain=False`` deletes the stored text as soon as the run ends while
keeping the scores and spans.
"""

from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import db
from .artifacts import registry
from .config import DISCLAIMER, settings
from .detector.ensemble import AIDetector
from .integrity import engine as integrity_engine
from .logging_setup import get_logger
from .parsers import parse_file
from .parsers.archive import extract
from .parsers.base import ParseError, ParsedDoc
from .plagiarism.batch import BatchDocument, compare, merge_code_edges
from .plagiarism.code_sim import CodeFile, compare_all
from .plagiarism.corpus import PlagiarismCorpus
from .plagiarism.engine import analyse as analyse_plagiarism
from .progress import Progress
from .schemas import AIResult, AnalysisResult, DocumentResult

log = get_logger(__name__)

_corpus: PlagiarismCorpus | None = None


def get_corpus() -> PlagiarismCorpus:
    """The process-wide plagiarism corpus (loaded once, extended at runtime)."""
    global _corpus
    if _corpus is None:
        _corpus = PlagiarismCorpus()
        _corpus.load()
    return _corpus


@dataclass
class InputDocument:
    """One uploaded or pasted document waiting to be analysed."""

    filename: str
    path: Path | None = None
    text: str | None = None
    is_archive: bool = False
    parsed: ParsedDoc | None = None


@dataclass
class Job:
    """Everything the worker needs for one analysis."""

    analysis_id: str
    documents: list[InputDocument]
    options: dict = field(default_factory=dict)


def run_job(job: Job) -> None:
    """Execute the pipeline, recording the result or the error in the database."""
    progress = Progress(job.analysis_id)
    try:
        result = _run(job, progress)
    except ParseError as exc:
        log.warning("analysis %s failed: %s", job.analysis_id, exc.message)
        _store_failure(job.analysis_id, exc.code, exc.message)
        progress.fail(exc.message)
        return
    except Exception as exc:  # noqa: BLE001 - a worker crash must be recorded, not lost
        log.exception("analysis %s crashed", job.analysis_id)
        _store_failure(job.analysis_id, "internal_error", str(exc))
        progress.fail(str(exc))
        return

    with db.session() as session:
        row = session.get(db.AnalysisRow, job.analysis_id)
        if row is not None:
            row.status = "done"
            row.result = result.model_dump()
            row.verdicts = [d.ai.verdict if d.ai else "unknown" for d in result.documents]
            row.similarity = max(
                (d.plagiarism.similarity_percent for d in result.documents if d.plagiarism),
                default=None,
            )
            if not row.retain:
                _purge_uploads(job.analysis_id)
                row.artifacts_purged = True
            session.commit()
    progress.finish()


def _store_failure(analysis_id: str, code: str, message: str) -> None:
    with db.session() as session:
        row = session.get(db.AnalysisRow, analysis_id)
        if row is not None:
            row.status = "error"
            row.error = f"{code}: {message}"
            session.commit()


def _purge_uploads(analysis_id: str) -> None:
    target = settings.uploads_path / analysis_id
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def _parse(job: Job, progress: Progress) -> list[ParsedDoc]:
    docs: list[ParsedDoc] = []
    total = max(1, len(job.documents))
    for index, item in enumerate(job.documents):
        progress.stage("parse", f"parsing {item.filename}", index / total)
        if item.text is not None:
            docs.append(
                ParsedDoc(filename=item.filename, kind="text", text=item.text, meta={"pasted": True})
            )
            continue
        assert item.path is not None
        if item.is_archive:
            members, warnings = extract(item.path.read_bytes())
            workdir = settings.uploads_path / job.analysis_id / "_members"
            for member in members:
                try:
                    from .parsers import parse_bytes

                    parsed = parse_bytes(member.data, member.name, workdir)
                    parsed.warnings.extend(warnings)
                    docs.append(parsed)
                except ParseError as exc:
                    docs.append(
                        ParsedDoc(
                            filename=member.name,
                            kind="text",
                            text="",
                            warnings=[f"{exc.code}: {exc.message}"],
                        )
                    )
        else:
            docs.append(parse_file(item.path, filename=item.filename))
    progress.done("parse")
    return docs


def _run(job: Job, progress: Progress) -> AnalysisResult:
    options = job.options
    mode = options.get("mode") or registry.default_mode() or settings.default_mode
    parsed_docs = _parse(job, progress)
    if not parsed_docs:
        raise ParseError("no_documents", "No readable documents were supplied.", 422)

    progress.stage("integrity", "checking for hidden text", 0.1)
    detector = AIDetector(registry)
    documents: list[DocumentResult] = []
    notes: list[str] = []
    code_files: list[CodeFile] = []

    for index, parsed in enumerate(parsed_docs):
        progress.stage("integrity", f"integrity check on {parsed.filename}", index / len(parsed_docs))
        hidden_chars = int(parsed.meta.get("hidden_chars", 0) or 0)
        hidden_reasons = parsed.meta.get("hidden_reasons") or {}
        integrity = integrity_engine.run(
            parsed.text, hidden_chars=hidden_chars, hidden_reasons=hidden_reasons  # type: ignore[arg-type]
        )
        clean_text = integrity.text
        document = DocumentResult(
            id=f"doc-{index}",
            filename=parsed.filename,
            kind=parsed.kind,
            words=len(clean_text.split()),
            pages=parsed.pages,
            text=clean_text,
            warnings=list(parsed.warnings),
            integrity=integrity.result,
        )
        if parsed.kind == "code":
            code_files.append(_code_file(parsed))
            document.ai = AIResult(
                mode_requested=str(mode),
                verdict="too_short",
                warnings=["source code is compared structurally, not scored for AI authorship"],
            )
        elif options.get("ai", True):
            output = detector.run(clean_text, str(mode), explain=bool(options.get("explain", True)))
            document.ai = output.result
        if options.get("plagiarism", True) and parsed.kind != "code":
            progress.stage("plagiarism", f"comparing {parsed.filename}", index / len(parsed_docs))
            plag = analyse_plagiarism(
                clean_text,
                get_corpus(),
                {
                    "online": bool(options.get("online", False)),
                    "exclusions": options.get("exclusions", {}),
                },
            )
            document.plagiarism = plag.result
        documents.append(document)
    progress.done("integrity")
    progress.done("ai")

    batch_result = None
    if options.get("batch") and len(documents) > 1:
        progress.stage("batch", "comparing documents with each other", 0.3)
        batch_docs = [
            BatchDocument(
                id=d.id,
                label=d.filename,
                text=d.text,
                words=d.words,
                verdict=d.ai.verdict if d.ai else "uncertain",
            )
            for d in documents
        ]
        batch_result = compare(batch_docs)
        progress.done("batch")

    code_matches = compare_all(code_files) if len(code_files) > 1 else []
    if batch_result is not None and code_matches:
        batch_result = merge_code_edges(batch_result, code_matches)
    if len(code_files) == 1:
        notes.append("only one code file was submitted; cross-file comparison needs at least two")
    if options.get("online"):
        notes.append("online source check was enabled; only short snippets were sent")

    progress.stage("report", "assembling the report", 0.5)
    result = AnalysisResult(
        id=job.analysis_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        status="done",
        options=options,  # type: ignore[arg-type]
        documents=documents,
        batch=batch_result,
        code_matches=code_matches,
        exit_notes=notes,
        disclaimer=DISCLAIMER,
    )
    progress.done("report")
    if not options.get("retain", True):
        notes.append("retain=false: uploaded files and extracted text were deleted after analysis")
    return result


def _code_file(parsed: ParsedDoc) -> CodeFile:
    """Build a CodeFile from an already-parsed document."""
    from .plagiarism.code_sim import _generic_tokens, _python_tokens

    language = str(parsed.meta.get("language", "text"))
    text = parsed.text
    if language == "py":
        tokens, lines, functions = _python_tokens(text)
    else:
        tokens, lines, functions = _generic_tokens(text, language)
    return CodeFile(
        name=parsed.filename,
        language=language,
        text=text,
        tokens=tokens,
        token_lines=lines,
        functions=functions,
    )


def start_worker(job: Job) -> threading.Thread:
    """Run ``job`` in a daemon thread and return it (used by the API)."""
    thread = threading.Thread(target=run_job, args=(job,), name=f"analysis-{job.analysis_id}", daemon=True)
    thread.start()
    return thread
