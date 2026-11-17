"""Detection of invisible / zero-width characters used to evade detectors.

Embedding zero-width spaces between letters or adding soft hyphens changes token
statistics without changing what a human reads.  This module finds those
characters, reports them (so the UI can show an integrity flag) and removes them
before analysis.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Code points that render as nothing (or as direction changes) in the text.
INVISIBLE_CHARS: dict[str, str] = {
    "\u200b": "zero width space",
    "\u200c": "zero width non-joiner",
    "\u200d": "zero width joiner",
    "\u2060": "word joiner",
    "\ufeff": "byte order mark",
    "\u00ad": "soft hyphen",
    "\u180e": "mongolian vowel separator",
    "\u2061": "function application",
    "\u2062": "invisible times",
    "\u2063": "invisible separator",
    "\u2064": "invisible plus",
    "\u202a": "left-to-right embedding",
    "\u202b": "right-to-left embedding",
    "\u202c": "pop directional formatting",
    "\u202d": "left-to-right override",
    "\u202e": "right-to-left override",
    "\u2066": "left-to-right isolate",
    "\u2067": "right-to-left isolate",
    "\u2068": "first strong isolate",
    "\u2069": "pop directional isolate",
}
_TAG_CHARS = re.compile(r"[\U000E0000-\U000E007F]")  # Unicode tag block, also invisible


@dataclass
class InvisibleReport:
    """What was found and what the cleaned text looks like."""

    count: int
    kinds: dict[str, int] = field(default_factory=dict)
    positions: list[int] = field(default_factory=list)
    cleaned: str = ""

    @property
    def flagged(self) -> bool:
        return self.count > 0


def scan(text: str) -> InvisibleReport:
    """Count invisible characters and return the text with them removed."""
    kinds: dict[str, int] = {}
    positions: list[int] = []
    kept: list[str] = []
    for idx, ch in enumerate(text):
        name = INVISIBLE_CHARS.get(ch)
        if name is None and _TAG_CHARS.match(ch):
            name = "unicode tag character"
        if name is None and unicodedata.category(ch) == "Cf" and ch not in "\n\t":
            name = f"format character U+{ord(ch):04X}"
        if name is not None:
            kinds[name] = kinds.get(name, 0) + 1
            positions.append(idx)
            continue
        kept.append(ch)
    return InvisibleReport(
        count=len(positions), kinds=kinds, positions=positions, cleaned="".join(kept)
    )


def has_invisible(text: str) -> bool:
    """Cheap predicate used by the integrity pass."""
    return bool(INVISIBLE_CHARS.keys() & set(text)) or bool(_TAG_CHARS.search(text))


def insert_invisible(text: str, every: int = 25, char: str = "\u200b") -> str:
    """Perturbation used by the robustness evaluation (inserts a zero-width space)."""
    if not text:
        return text
    out: list[str] = []
    for i, word in enumerate(text.split(" ")):
        out.append(word)
        if i % max(1, every // 5) == 0:
            out[-1] = out[-1] + char
    return " ".join(out)
