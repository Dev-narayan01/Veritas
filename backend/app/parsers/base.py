"""Shared parser data structures.

Every parser returns the same ``ParsedDoc`` so the rest of the pipeline never
needs to know which file format produced the text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Extensions we accept, grouped by the parser that handles them.
PDF_EXT = {".pdf"}
DOCX_EXT = {".docx"}
PPTX_EXT = {".pptx"}
TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".rtf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}
ZIP_EXT = {".zip"}
CODE_EXT = {".py", ".java", ".c", ".cpp", ".h", ".js", ".ts", ".go", ".rs", ".cs", ".php", ".rb"}

ALL_EXT = PDF_EXT | DOCX_EXT | PPTX_EXT | TEXT_EXT | IMAGE_EXT | ZIP_EXT | CODE_EXT


class ParseError(Exception):
    """Raised when a file cannot be parsed; carries an API error code."""

    def __init__(self, code: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class OCRUnavailable(ParseError):
    """Raised when an image needs OCR but the Tesseract binary is missing."""

    def __init__(self) -> None:
        super().__init__(
            "ocr_unavailable",
            "OCR is not available on this server; upload a text-based document instead.",
            422,
        )


@dataclass
class ParsedDoc:
    """Normalised output of any parser."""

    filename: str
    kind: str  # pdf | docx | pptx | text | image | code | archive-member
    text: str
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def words(self) -> int:
        return len(self.text.split())


def safe_name(name: str) -> str:
    """Strip directory components and unsafe characters from an upload name."""
    base = Path(name).name
    cleaned = "".join(c for c in base if c.isalnum() or c in "._- ()[]").strip()
    cleaned = cleaned.replace(" ", "_")
    return cleaned[:120] or "document"
