"""Section tagging and exclusion filters (quotes, bibliography, code).

Plagiarism results are only fair if a student's own quotations and reference list
are excluded on request.  Tagging is done once per document, and every match keeps
its raw flags so the UI can re-filter instantly without re-running the engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Headings that start a bibliography when they appear in the last part of the text.
_BIB_HEADING = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(references|bibliography|works\s+cited|sources|literature\s+cited)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_QUOTE_BLOCK = re.compile(r"^\s*>.*$", re.MULTILINE)
_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_LAST_FRACTION_FOR_BIB = 0.70


@dataclass
class Section:
    """A tagged region of the document."""

    kind: str  # body | quote | bibliography | code
    start: int
    end: int


def _overlaps(span: tuple[int, int], sections: list[Section]) -> bool:
    return any(not (span[1] <= s.start or span[0] >= s.end) for s in sections)


def tag_sections(text: str) -> list[Section]:
    """Return the non-body sections of ``text`` in document order."""
    sections: list[Section] = []

    # Code fences are unambiguous.
    for m in _CODE_FENCE.finditer(text):
        sections.append(Section("code", m.start(), m.end()))

    # Markdown-style block quotes.
    for m in _QUOTE_BLOCK.finditer(text):
        if not _overlaps((m.start(), m.end()), sections):
            sections.append(Section("quote", m.start(), m.end()))

    # Inline quotations of 15+ words are quotations, not plagiarism.
    for m in re.finditer(r"[\"“]([^\"”]{60,})[\"”]", text):
        inner = m.group(1)
        if len(inner.split()) >= 15 and not _overlaps((m.start(), m.end()), sections):
            sections.append(Section("quote", m.start(), m.end()))

    # Bibliography: a heading in the last 30% of the text runs to the end.
    threshold = int(len(text) * _LAST_FRACTION_FOR_BIB)
    for m in _BIB_HEADING.finditer(text):
        if m.start() >= threshold:
            sections.append(Section("bibliography", m.start(), len(text)))
            break

    sections.sort(key=lambda s: s.start)
    return sections


def classify_span(span: tuple[int, int], sections: list[Section]) -> str:
    """Return the most specific tag covering ``span`` (body when none does)."""
    best = "body"
    for s in sections:
        if s.start <= span[0] and span[1] <= s.end:
            return s.kind
    for s in sections:
        if not (span[1] <= s.start or span[0] >= s.end):
            # Partial overlap: bibliography/code win, quotes are reported too.
            if s.kind in {"bibliography", "code"}:
                return s.kind
            best = s.kind if best == "body" else best
    return best


def is_excluded(span: tuple[int, int], text: str, sections: list[Section],
                exclude_quotes: bool, exclude_bibliography: bool) -> bool:
    """Decide whether a match span should be hidden by the current filters."""
    if not (exclude_quotes or exclude_bibliography):
        return False
    kind = classify_span(span, sections)
    if exclude_quotes and kind == "quote":
        return True
    if exclude_bibliography and kind in {"bibliography", "code"}:
        return True
    return False


def strip_bibliography(text: str) -> str:
    """Return ``text`` without its trailing bibliography (used by the corpus builder)."""
    for s in tag_sections(text):
        if s.kind == "bibliography":
            return text[: s.start]
    return text
