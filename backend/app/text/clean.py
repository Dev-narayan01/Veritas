"""Text normalisation with an offset map back to the original document.

Detectors work best on clean text, but every highlight in the UI must point at the
*original* characters the user uploaded.  ``normalize`` therefore returns the
cleaned string plus ``orig_offsets``: a list where element *i* is the character
index in the source text that produced cleaned character *i*.  Highlights are
built by mapping cleaned spans through this list, so sentence boundaries,
plagiarism spans and token views all align with the file the user sees.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Unicode categories that carry no visible content for analysis.
_CONTROL_CATEGORIES = {"Cc", "Cf"}
_WHITESPACE = " \t\r\n\f\v"


@dataclass
class NormalizedText:
    """A cleaned document and the mapping needed to highlight the original."""

    text: str
    orig_offsets: list[int] = field(default_factory=list)
    original_length: int = 0

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a (start, end) span in cleaned text to the original document."""
        if not self.orig_offsets:
            return start, end
        n = len(self.orig_offsets)
        start = max(0, min(start, n))
        end = max(0, min(end, n))
        if start >= end:
            return self.orig_offsets[start] if start < n else self.original_length, (
                self.orig_offsets[start] if start < n else self.original_length
            )
        return self.orig_offsets[start], self.orig_offsets[end - 1] + 1


def normalize(raw: str) -> NormalizedText:
    """NFKC-normalise, drop control characters and collapse whitespace runs.

    The output keeps one space between words (never at the start), and every
    emitted character remembers where it came from in ``raw``.
    """
    out: list[str] = []
    offsets: list[int] = []
    pending_space = False

    for idx, ch in enumerate(raw):
        # Fast path: normalise one character at a time so the map stays exact.
        norm = unicodedata.normalize("NFKC", ch)
        for c in norm:
            if c in _WHITESPACE or unicodedata.category(c) == "Zs":
                pending_space = bool(out)
                continue
            if unicodedata.category(c) in _CONTROL_CATEGORIES:
                continue
            if pending_space:
                out.append(" ")
                offsets.append(idx)
                pending_space = False
            out.append(c)
            offsets.append(idx)

    text = "".join(out).strip()
    # ``strip`` only removes leading/trailing spaces (each maps to one offset).
    if text:
        lead = 0
        while lead < len(offsets) and "".join(out[lead : lead + 1]) == " ":
            lead += 1
        offsets = offsets[lead : lead + len(text)]
    else:
        offsets = []
    return NormalizedText(text=text, orig_offsets=offsets, original_length=len(raw))


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def join_hyphenated(raw: str) -> str:
    """Re-join words split by a line-end hyphen, a common PDF extraction artefact."""
    return _HYPHEN_BREAK.sub(r"\1\2", raw)


def word_count(text: str) -> int:
    """Count whitespace-delimited tokens, the unit all detector limits use."""
    return len(text.split())


def collapse_for_compare(text: str) -> str:
    """Lowercase alphanumeric skeleton used for de-duplication of datasets."""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()
