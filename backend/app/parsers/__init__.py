"""File parsers: turn uploaded bytes into :class:`ParsedDoc` objects."""

from __future__ import annotations

from pathlib import Path

from ..config import settings
from ..logging_setup import get_logger
from .base import (  # noqa: F401 - re-exported for callers
    ALL_EXT,
    CODE_EXT,
    DOCX_EXT,
    IMAGE_EXT,
    PDF_EXT,
    PPTX_EXT,
    TEXT_EXT,
    ZIP_EXT,
    ParseError,
    ParsedDoc,
    safe_name,
)

log = get_logger(__name__)

# Magic numbers used to confirm the extension. ``None`` means "no magic".
_MAGIC: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF",),
    ".docx": (b"PK\x03\x04",),
    ".pptx": (b"PK\x03\x04",),
    ".zip": (b"PK\x03\x04",),
    ".png": (b"\x89PNG",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".webp": (b"RIFF",),
    ".tiff": (b"II*\x00", b"MM\x00*"),
    ".tif": (b"II*\x00", b"MM\x00*"),
    ".bmp": (b"BM",),
}


def check_magic(path: Path, suffix: str) -> None:
    """Reject a file whose leading bytes contradict its extension."""
    expected = _MAGIC.get(suffix)
    if not expected:
        return
    with path.open("rb") as handle:
        head = handle.read(8)
    if not any(head.startswith(m) for m in expected):
        raise ParseError(
            "type_mismatch",
            f"The file content does not match the {suffix} extension.",
            415,
        )


def parse_file(path: str | Path, filename: str | None = None) -> ParsedDoc:
    """Dispatch to the right parser based on the extension (after magic checks)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ALL_EXT and suffix != ".zip":
        raise ParseError("unsupported_type", f"Unsupported file type: {suffix or 'none'}", 415)
    check_magic(path, suffix)

    if suffix in PDF_EXT:
        from . import pdf as mod
    elif suffix in DOCX_EXT:
        from . import docx as mod
    elif suffix in PPTX_EXT:
        from . import pptx as mod
    elif suffix in IMAGE_EXT:
        from . import image_ocr as mod
    elif suffix in CODE_EXT:
        from . import code as mod
    else:
        from . import text as mod
    doc = mod.parse(path, filename)
    if len(doc.text) > settings.max_chars_per_doc:
        doc.text = doc.text[: settings.max_chars_per_doc]
        doc.warnings.append("truncated_to_char_limit")
    return doc


def parse_bytes(data: bytes, filename: str, tmp_dir: Path) -> ParsedDoc:
    """Persist ``data`` under ``tmp_dir`` and parse it (used for ZIP members)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_name(filename)
    target = tmp_dir / safe
    target.write_bytes(data)
    return parse_file(target, filename=safe)
