"""Image parsing via Tesseract OCR (optional dependency, explicit failure)."""

from __future__ import annotations

from pathlib import Path

from ..logging_setup import get_logger
from .base import OCRUnavailable, ParsedDoc

log = get_logger(__name__)

_AVAILABLE: bool | None = None


def ocr_available() -> bool:
    """True when both ``pytesseract`` and the Tesseract binary are usable."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            _AVAILABLE = True
        except Exception as exc:  # noqa: BLE001 - optional dependency probe
            log.warning("OCR unavailable: %s", exc)
            _AVAILABLE = False
    return _AVAILABLE


def ocr_image_bytes(data: bytes) -> str:
    """OCR raw image bytes; raises ``OCRUnavailable`` when Tesseract is missing."""
    if not ocr_available():
        raise OCRUnavailable()
    import pytesseract
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(data))
    return pytesseract.image_to_string(image, lang="eng")


def parse(path: str | Path, filename: str | None = None) -> ParsedDoc:
    """OCR an image file into text."""
    path = Path(path)
    if not ocr_available():
        raise OCRUnavailable()
    text = ocr_image_bytes(path.read_bytes())
    warnings = [] if text.strip() else ["ocr_empty"]
    return ParsedDoc(
        filename=filename or path.name,
        kind="image",
        text=text.strip(),
        pages=1,
        warnings=warnings,
        meta={"ocr": True},
    )
