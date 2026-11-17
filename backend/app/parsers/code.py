"""Source-code files: read as text but tagged ``kind="code"``.

Code is compared by the structure-aware plagiarism module, never by the AI-text
detector, because token statistics of Python are not comparable to prose.
"""

from __future__ import annotations

from pathlib import Path

from .base import ParsedDoc
from .text import decode_bytes


def parse(path: str | Path, filename: str | None = None) -> ParsedDoc:
    path = Path(path)
    text, encoding = decode_bytes(path.read_bytes())
    language = path.suffix.lstrip(".").lower()
    return ParsedDoc(
        filename=filename or path.name,
        kind="code",
        text=text,
        pages=None,
        warnings=[] if text else ["empty_document"],
        meta={"encoding": encoding, "language": language},
    )
