"""The integrity pass that runs before any detector.

It answers three questions for the user: is part of this document hidden, are there
invisible characters, and are words disguised with homoglyphs?  Everything that is
detected is reported *and* cleaned, so the detector is not fooled while the user
still learns what was removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import IntegrityFlag, IntegrityResult
from . import homoglyph, invisible


@dataclass
class IntegrityOutcome:
    """Cleaned text plus the API-facing flags."""

    text: str
    result: IntegrityResult
    removed_chars: int = 0
    notes: list[str] = field(default_factory=list)


def run(text: str, hidden_chars: int = 0, hidden_reasons: dict[str, int] | None = None) -> IntegrityOutcome:
    """Detect and strip invisible characters and homoglyphs; report hidden PDF text."""
    flags: list[IntegrityFlag] = []
    removed = 0

    inv = invisible.scan(text)
    if inv.flagged:
        kinds = ", ".join(f"{k} ×{v}" for k, v in sorted(inv.kinds.items()))
        flags.append(
            IntegrityFlag(
                kind="invisible_characters",
                count=inv.count,
                detail=kinds,
                examples=[f"U+{ord(c):04X}" for c in list(invisible.INVISIBLE_CHARS)[:5]],
            )
        )
        text = inv.cleaned
        removed += inv.count

    cleaned_homoglyphs, fixed = homoglyph.normalize_homoglyphs(text)
    if fixed:
        flags.append(
            IntegrityFlag(
                kind="homoglyph_substitution",
                count=len(fixed),
                detail="Mixed Latin/Cyrillic or Latin/Greek words normalised to Latin.",
                examples=fixed[:8],
            )
        )
        text = cleaned_homoglyphs

    if hidden_chars:
        detail = ", ".join(f"{k} ×{v}" for k, v in sorted((hidden_reasons or {}).items()))
        flags.append(
            IntegrityFlag(
                kind="hidden_pdf_text",
                count=hidden_chars,
                detail=detail or "invisible characters excluded from analysis",
            )
        )

    return IntegrityOutcome(
        text=text,
        result=IntegrityResult(
            flags=flags,
            cleaned_chars=removed,
            hidden_text_removed=hidden_chars,
        ),
        removed_chars=removed,
    )
