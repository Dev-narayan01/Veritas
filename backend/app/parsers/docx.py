"""DOCX parsing: paragraphs plus table cells, in document order."""

from __future__ import annotations

from pathlib import Path

from .base import ParsedDoc


def parse(path: str | Path, filename: str | None = None) -> ParsedDoc:
    """Extract the readable text of a Word document."""
    import docx

    path = Path(path)
    document = docx.Document(str(path))
    parts: list[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n".join(parts).strip()
    warnings = [] if text else ["empty_document"]
    return ParsedDoc(
        filename=filename or path.name,
        kind="docx",
        text=text,
        pages=None,
        warnings=warnings,
        meta={"paragraphs": len(document.paragraphs), "tables": len(document.tables)},
    )
