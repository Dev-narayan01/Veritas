"""PDF parsing with hidden-text handling and OCR fallback for scanned pages."""

from __future__ import annotations

from pathlib import Path

from ..integrity import pdf_hidden
from ..logging_setup import get_logger
from .base import ParsedDoc
from .image_ocr import ocr_available, ocr_image_bytes

log = get_logger(__name__)

MIN_PAGE_CHARS = 20


def parse(path: str | Path, filename: str | None = None) -> ParsedDoc:
    """Extract visible text from a PDF, page by page.

    Pages with fewer than ``MIN_PAGE_CHARS`` characters are treated as scanned and
    OCR'd when Tesseract is installed, otherwise a ``scanned_pdf`` warning is added
    so the UI can tell the user the text may be incomplete.
    """
    import pdfplumber

    path = Path(path)
    warnings: list[str] = []
    pages_text: list[str] = []
    hidden = pdf_hidden.analyse(path)

    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        for number, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < MIN_PAGE_CHARS:
                if ocr_available():
                    try:
                        image = page.to_image(resolution=200).original
                        buf = _to_png_bytes(image)
                        text = ocr_image_bytes(buf).strip()
                        if text:
                            warnings.append(f"page {number} read with OCR")
                            pages_text.append(text)
                            continue
                    except Exception as exc:  # noqa: BLE001 - OCR is best effort
                        log.warning("OCR failed on page %s: %s", number, exc)
                warnings.append("scanned_pdf")
            pages_text.append(text)

    # Hidden characters are removed from the text we analyse but counted for flags.
    text = "\n".join(pages_text).strip()
    if hidden.flagged:
        warnings.append(f"hidden_text_removed:{hidden.hidden_chars}")
    return ParsedDoc(
        filename=filename or path.name,
        kind="pdf",
        text=text,
        pages=page_count,
        warnings=_dedupe(warnings),
        meta={
            "hidden_chars": hidden.hidden_chars,
            "hidden_reasons": hidden.reasons,
            "visible_text_chars": len(hidden.visible_text),
        },
    )


def _to_png_bytes(image) -> bytes:  # noqa: ANN001 - PIL image, typed loosely
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
