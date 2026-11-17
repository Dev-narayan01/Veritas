"""PPTX parsing: slide text frames and speaker notes."""

from __future__ import annotations

from pathlib import Path

from .base import ParsedDoc


def parse(path: str | Path, filename: str | None = None) -> ParsedDoc:
    """Extract text from every slide, including notes."""
    from pptx import Presentation

    path = Path(path)
    presentation = Presentation(str(path))
    parts: list[str] = []
    for number, slide in enumerate(presentation.slides, start=1):
        slide_parts: list[str] = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = shape.text_frame.text.strip()
                if text:
                    slide_parts.append(text)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                slide_parts.append(notes)
        if slide_parts:
            parts.append("\n".join(slide_parts))
    text = "\n\n".join(parts).strip()
    warnings = [] if text else ["empty_document"]
    return ParsedDoc(
        filename=filename or path.name,
        kind="pptx",
        text=text,
        pages=len(presentation.slides),
        warnings=warnings,
        meta={"slides": len(presentation.slides)},
    )
