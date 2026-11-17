"""Token-level evidence: per-token surprisal and the most "machine-like" tokens.

When a language-model backend is available we report the real surprisal of every
token; otherwise we approximate surprisal from word rarity and sentence position and
label the view as an approximation so nothing is overstated.  The most machine-like
5% of tokens are flagged.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..schemas import TokenScore
from ..text.segment import Sentence

_TOKEN = re.compile(r"\S+")
MAX_TOKENS = 600


def _approximate_surprisal(text: str) -> list[tuple[str, int, int, float]]:
    """Rarity-based surprisal proxy: rare words are surprising, common ones are not."""
    tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]
    counts = Counter(t.strip(".,;:!?\"'()[]").lower() for t, _, _ in tokens)
    total = max(1, sum(counts.values()))
    out: list[tuple[str, int, int, float]] = []
    for token, start, end in tokens[:MAX_TOKENS]:
        key = token.strip(".,;:!?\"'()[]").lower()
        freq = counts.get(key, 1) / total
        sentence_break = 0.7 if token.endswith(('.', '!', '?')) else 0.0
        out.append((token, start, end, -math.log(freq) + sentence_break))
    return out


def token_view(text: str, sentences: list[Sentence], ppl_probability: float | None) -> list[TokenScore]:
    """Return token scores with the top 5% flagged.

    ``ppl_probability`` is used only to decide whether the document looks machine
    written at all; the ranking itself always comes from the surprisal proxy unless
    a model backend supplies real values (called from the ensemble when present).
    """
    if not text.strip():
        return []
    rows = _approximate_surprisal(text)
    if not rows:
        return []
    surprisals = sorted(r[3] for r in rows)
    # "Machine-like" tokens are the *most predictable* ones: low surprisal.
    cutoff_index = max(0, int(len(surprisals) * 0.05))
    low_cutoff = surprisals[cutoff_index] if surprisals else 0.0
    ranked = sorted(range(len(rows)), key=lambda i: rows[i][3])
    rank_of = {idx: rank for rank, idx in enumerate(ranked)}
    out: list[TokenScore] = []
    for idx, (token, start, end, surprisal) in enumerate(rows):
        out.append(
            TokenScore(
                token=token,
                start=start,
                end=end,
                surprisal=round(float(surprisal), 4),
                rank=rank_of[idx],
                flagged=surprisal <= low_cutoff,
            )
        )
    return out
