"""Stylometric features shared by the statistical components.

These are all cheap, interpretable measurements of writing style.  Machine text
tends to have lower variance in sentence length, fewer contractions, flatter word
choice and more repetition; the features below make those tendencies measurable
and every one of them is shown to the user in the explanation card.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

from ..text.segment import split_sentences

STOPWORDS = frozenset(
    """a about above after again against all am an and any are aren't as at be because
    been before being below between both but by can't cannot could couldn't did didn't
    do does doesn't doing don't down during each few for from further had hadn't has
    hasn't have haven't having he he'd he'll he's her here here's hers herself him
    himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
    let's me more most mustn't my myself no nor not of off on once only or other ought
    our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
    so some such than that that's the their theirs them themselves then there there's
    these they they'd they'll they're they've this those through to too under until up
    very was wasn't we we'd we'll we're we've were weren't what what's when when's where
    where's which while who who's whom why why's with won't would wouldn't you you'd
    you'll you're you've your yours yourself yourselves""".split()
)

_CONTRACTION = re.compile(r"\b\w+(?:'|’)t\b|\b\w+'(?:s|re|ve|ll|d|m)\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z']+")
_FUNCTION_PUNCT = set(".,;:!?'\"()[]-")


@dataclass
class StyleFeatures:
    """Interpretable style measurements for one text."""

    sentence_len_mean: float = 0.0
    sentence_len_std: float = 0.0
    word_len_mean: float = 0.0
    type_token_ratio: float = 0.0
    punctuation_rate: float = 0.0
    stopword_ratio: float = 0.0
    repetition_rate: float = 0.0
    contraction_rate: float = 0.0
    long_word_ratio: float = 0.0
    sentence_count: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_style(text: str) -> StyleFeatures:
    """Measure style features on cleaned text."""
    sentences = split_sentences(text) or []
    words = _WORD.findall(text.lower())
    total_words = max(1, len(words))
    lengths = [len(s.text.split()) for s in sentences] or [total_words]
    mean_len = _safe_div(sum(lengths), len(lengths))
    std_len = math.sqrt(_safe_div(sum((x - mean_len) ** 2 for x in lengths), len(lengths)))
    counts = Counter(words)
    unique = len(counts)
    # Repetition: the share of tokens belonging to words used more than twice.
    repeated = sum(c for w, c in counts.items() if c > 2 and w not in STOPWORDS)
    return StyleFeatures(
        sentence_len_mean=mean_len,
        sentence_len_std=std_len,
        word_len_mean=_safe_div(sum(len(w) for w in words), total_words),
        type_token_ratio=_safe_div(unique, total_words),
        punctuation_rate=_safe_div(
            sum(1 for c in text if c in _FUNCTION_PUNCT), max(1, len(text))
        ),
        stopword_ratio=_safe_div(sum(1 for w in words if w in STOPWORDS), total_words),
        repetition_rate=_safe_div(repeated, total_words),
        contraction_rate=_safe_div(len(_CONTRACTION.findall(text)), total_words),
        long_word_ratio=_safe_div(sum(1 for w in words if len(w) >= 8), total_words),
        sentence_count=float(len(sentences)),
    )


def length_bucket(words: int) -> str:
    """The length buckets used for per-length thresholds and evaluation breakdowns."""
    if words < 100:
        return "<100"
    if words < 250:
        return "100-250"
    if words < 500:
        return "250-500"
    return "500+"


def top_feature_contributions(style: StyleFeatures, reference: dict[str, dict[str, float]]) -> list[dict[str, float | str]]:
    """Rank style features by |z-score| against training statistics.

    ``reference`` maps feature name to {"mean", "std"}; it comes from the trained
    artifact so the explanation reflects the model's own view of "unusual".
    """
    out: list[dict[str, float | str]] = []
    for name, value in style.as_dict().items():
        stats = reference.get(name)
        if not stats or not stats.get("std"):
            continue
        z = (value - stats["mean"]) / stats["std"]
        out.append({"feature": name, "value": round(value, 4), "z": round(z, 3)})
    out.sort(key=lambda item: abs(float(item["z"])), reverse=True)
    return out
