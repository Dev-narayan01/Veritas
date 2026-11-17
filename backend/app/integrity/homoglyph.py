"""Homoglyph / mixed-script detection.

An evasion trick is to replace Latin letters with visually identical Cyrillic or
Greek characters ("раypal").  The text looks unchanged to a human, but the tokens
are different.  We detect words whose scripts are mixed, and normalise them back
to Latin with a small confusables table so the analysis sees the intended word.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

# The most common Cyrillic/Greek lookalikes for Latin letters (lowercase).
CONFUSABLES: dict[str, str] = {
    # Cyrillic
    "а": "a", "в": "b", "е": "e", "ё": "e", "к": "k", "м": "m", "н": "h", "о": "o",
    "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "ѕ": "s", "і": "i", "ј": "j",
    "ԁ": "d", "ԛ": "q", "ԝ": "w", "һ": "h", "ӏ": "l", "ɡ": "g", "ү": "y", "п": "n",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "У": "Y", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    # Greek
    "α": "a", "β": "b", "ε": "e", "η": "n", "ι": "i", "κ": "k", "ν": "v", "ο": "o",
    "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "ω": "w", "γ": "y", "ζ": "z", "μ": "u",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Common mathematical / fullwidth lookalikes are normalised by NFKC already.
}

_SCRIPT_GROUPS: dict[str, str] = {
    "LATIN": "latin",
    "CYRILLIC": "cyrillic",
    "GREEK": "greek",
    "ARMENIAN": "armenian",
    "HEBREW": "hebrew",
    "ARABIC": "arabic",
    "HIRAGANA": "japanese",
    "KATAKANA": "japanese",
    "CJK": "han",
    "HANGUL": "hangul",
    "THAI": "thai",
    "DEVANAGARI": "devanagari",
}


def script_of(ch: str) -> str | None:
    """Return the coarse script name of an alphabetic character."""
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch).split(" ")[0]
    except ValueError:
        return None
    return _SCRIPT_GROUPS.get(name)


@dataclass
class HomoglyphReport:
    """Mixed-script words found in a document."""

    words: list[str] = field(default_factory=list)
    cleaned: str = ""

    @property
    def flagged(self) -> bool:
        return bool(self.words)


def find_mixed_script_words(text: str) -> list[str]:
    """Words that mix Latin with Cyrillic/Greek lookalikes."""
    found: list[str] = []
    for token in text.split():
        scripts = {s for s in (script_of(c) for c in token) if s is not None}
        if len(scripts) > 1 and "latin" in scripts and scripts & {"cyrillic", "greek",
                                                                  "armenian", "hebrew"}:
            if token not in found:
                found.append(token)
    return found


def normalize_homoglyphs(text: str) -> tuple[str, list[str]]:
    """Replace known confusables inside mixed-script words; report what was fixed."""
    fixed: list[str] = []
    out_tokens: list[str] = []
    for token in text.split(" "):
        scripts = {s for s in (script_of(c) for c in token) if s is not None}
        if len(scripts) > 1 and "latin" in scripts:
            mapped = "".join(CONFUSABLES.get(c, c) for c in token)
            if mapped != token:
                fixed.append(token)
                out_tokens.append(mapped)
                continue
        out_tokens.append(token)
    return " ".join(out_tokens), fixed


def perturb_with_homoglyphs(text: str, rate: float = 0.15) -> str:
    """Robustness perturbation: swap some Latin letters for Cyrillic lookalikes."""
    reverse = {v: k for k, v in CONFUSABLES.items() if len(v) == 1 and v.islower()}
    out: list[str] = []
    for word in text.split():
        if len(word) > 2 and hash(word) % 100 < int(rate * 100):
            chars = list(word)
            for i, ch in enumerate(chars):
                if ch in reverse:
                    chars[i] = reverse[ch]
                    break
            word = "".join(chars)
        out.append(word)
    return " ".join(out)
