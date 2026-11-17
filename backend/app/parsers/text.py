"""Plain text / Markdown / CSV parsing with an encoding fallback chain."""

from __future__ import annotations

from pathlib import Path

from ..text.clean import join_hyphenated
from .base import ParsedDoc

_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")


def decode_bytes(data: bytes) -> tuple[str, str]:
    """Decode with utf-8 first, then latin-1; return (text, encoding used)."""
    for encoding in _ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


def parse(path: str | Path, filename: str | None = None) -> ParsedDoc:
    """Read a text-like file and report the encoding that worked."""
    path = Path(path)
    text, encoding = decode_bytes(path.read_bytes())
    text = join_hyphenated(text).strip()
    warnings = [] if text else ["empty_document"]
    if encoding.startswith("utf-8-replace"):
        warnings.append("encoding_loss")
    return ParsedDoc(
        filename=filename or path.name,
        kind="text",
        text=text,
        pages=None,
        warnings=warnings,
        meta={"encoding": encoding},
    )
