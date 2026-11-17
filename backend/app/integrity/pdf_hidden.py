"""Hidden-text detection inside PDFs (white text, tiny fonts, text outside the box).

The analysis of a PDF should use only the text a reader can see.  ``pdfplumber``
exposes character-level metadata (colour, size, position), so we can flag and drop
characters that are invisible or placed off the page, then report the count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..logging_setup import get_logger

log = get_logger(__name__)

MIN_VISIBLE_FONT_SIZE = 3.0
_LUMA_WHITE_THRESHOLD = 0.90


@dataclass
class HiddenTextReport:
    """Visible text plus a description of what was hidden and removed."""

    hidden_chars: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    visible_text: str = ""

    @property
    def flagged(self) -> bool:
        return self.hidden_chars > 0


def _is_white(color: object) -> bool:
    """Interpret pdfplumber's colour representation (int or component tuple)."""
    if color is None:
        return False
    if isinstance(color, (int, float)):
        value = int(color)
        r, g, b = (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF
        return min(r, g, b) / 255.0 >= _LUMA_WHITE_THRESHOLD
    if isinstance(color, (list, tuple)):
        comps = [float(c) for c in color if isinstance(c, (int, float))]
        if not comps:
            return False
        # pdfplumber reports 0..1 components for most PDFs and 0..255 for others.
        scale = 255.0 if max(comps) > 1.0 else 1.0
        return min(comps) / scale >= _LUMA_WHITE_THRESHOLD
    return False


def analyse(path: str | Path) -> HiddenTextReport:
    """Scan a PDF for hidden characters, returning only visible text."""
    import pdfplumber  # imported lazily so the API starts without a PDF stack

    report = HiddenTextReport()
    visible_parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            width, height = page.width, page.height
            chars = page.chars or []
            if not chars:
                continue
            page_parts: list[str] = []
            for ch in chars:
                text = ch.get("text", "")
                size = float(ch.get("size", 0) or 0)
                x0, top = float(ch.get("x0", 0)), float(ch.get("top", 0))
                x1, bottom = float(ch.get("x1", 0)), float(ch.get("bottom", 0))
                reason = None
                if size < MIN_VISIBLE_FONT_SIZE:
                    reason = "tiny_font"
                elif _is_white(ch.get("non_stroking_color")):
                    reason = "white_text"
                elif x1 < 0 or top > height or bottom < 0 or x0 > width:
                    reason = "outside_page"
                if reason:
                    report.hidden_chars += 1
                    report.reasons[reason] = report.reasons.get(reason, 0) + 1
                    continue
                page_parts.append(text)
            visible_parts.append("".join(page_parts))
    report.visible_text = "\n".join(visible_parts)
    if report.flagged:
        log.info("pdf hidden text: %s characters (%s)", report.hidden_chars, report.reasons)
    return report
