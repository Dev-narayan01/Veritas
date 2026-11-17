"""Corpus endpoints: add documents at runtime and read the honest corpus stats."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..pipeline import get_corpus
from ..schemas import CorpusStats, ErrorBody, ErrorResponse
from ..text.clean import normalize

router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.get("/stats", response_model=CorpusStats)
def stats() -> CorpusStats:
    """Counts of what is really indexed (no estimated coverage claims)."""
    return get_corpus().stats()


@router.post("/documents", status_code=201)
def add_document(payload: dict) -> dict[str, str]:
    """Append a document to the in-memory corpus and persist the index."""
    title = str(payload.get("title") or "Untitled").strip()[:200]
    text = normalize(str(payload.get("text") or "")).text
    if len(text.split()) < 30:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error=ErrorBody(code="text_too_short", message="At least 30 words are required")
            ).model_dump(),
        )
    corpus = get_corpus()
    document = corpus.add_document(title=title, text=text, url=payload.get("url"), source="runtime")
    corpus.save()
    return {"id": document.id, "title": document.title}
