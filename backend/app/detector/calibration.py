"""Platt scaling, threshold selection and the calibration guard.

The previous build collapsed because isotonic regression was fitted on the same
near-separable data the stacker had seen.  Here the protocol is strict:

* the stacker is fitted on ``val_stack`` and the scaler on ``val_cal`` (disjoint),
* calibration is Platt scaling (one-feature logistic regression) on the stacker's
  decision value, which is a strictly monotone map and therefore never produces a
  step function,
* thresholds come from ``val_cal`` only,
* and :func:`assert_score_diversity` aborts training when the calibrated validation
  scores have fewer than ``MIN_UNIQUE_SCORES`` distinct values, so a collapse can
  never be published as a result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

MIN_UNIQUE_SCORES = 100
LENGTH_BUCKETS = ("<100", "100-250", "250-500", "500+")
MIN_ROWS_PER_BUCKET = 200


class CalibrationCollapsed(RuntimeError):
    """Raised when calibration produces a degenerate score distribution."""


@dataclass
class PlattScaler:
    """One-feature logistic regression mapping a decision value to a probability."""

    a: float
    b: float

    @classmethod
    def fit(cls, decision: np.ndarray, labels: np.ndarray) -> PlattScaler:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(np.asarray(decision, dtype=float).reshape(-1, 1), labels)
        return cls(a=float(model.coef_[0][0]), b=float(model.intercept_[0]))

    def transform(self, decision: np.ndarray) -> np.ndarray:
        z = self.a * np.asarray(decision, dtype=float) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    def to_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> PlattScaler:
        return cls(a=float(data["a"]), b=float(data["b"]))


def assert_score_diversity(scores: np.ndarray, context: str = "val_cal") -> int:
    """Abort when the calibrated scores are (nearly) constant.

    A collapsed distribution means the stacker learned nothing generalisable, and
    publishing thresholds derived from it would be dishonest.  Raising here fails
    the training run with a clear message instead.
    """
    unique = int(np.unique(np.round(np.asarray(scores, dtype=float), 6)).size)
    if unique < MIN_UNIQUE_SCORES:
        raise CalibrationCollapsed(
            f"calibration collapsed on {context}: only {unique} unique scores "
            f"(need >= {MIN_UNIQUE_SCORES}). Check the split protocol and inputs."
        )
    return unique


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Maximum TPR whose FPR is <= target; returns (tpr, threshold)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    humans = scores[labels == 0]
    ai = scores[labels == 1]
    if humans.size == 0 or ai.size == 0:
        return 0.0, 2.0
    best_tpr, best_thr = 0.0, float(scores.max()) + 1e-9
    for thr in np.unique(np.concatenate([humans, ai])):
        if float(np.mean(humans >= thr)) <= target_fpr:
            tpr = float(np.mean(ai >= thr))
            if tpr >= best_tpr:
                best_tpr, best_thr = tpr, float(thr)
    return best_tpr, best_thr


def best_threshold_at_fpr(humans: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose human false-positive rate is <= ``target_fpr``.

    ``humans`` are calibrated human scores.  Scanning candidate values in ascending
    order and returning the first that satisfies the constraint gives the smallest
    such threshold, which maximises recall for a fixed false-positive budget.
    """
    humans = np.sort(np.asarray(humans, dtype=float))
    if humans.size == 0:
        return 1.0
    n = humans.size
    allowed = int(np.floor(target_fpr * n))
    if allowed == 0:
        return float(min(1.0, humans.max() + 1e-6))
    # Keep at most `allowed` humans at or above the threshold.
    candidates = humans[: n - allowed]
    for thr in candidates[::-1] if candidates.size else []:
        if float(np.mean(humans >= thr)) <= target_fpr:
            return float(thr)
    return float(min(1.0, humans.max()))


@dataclass
class ModeThresholds:
    """Thresholds for one mode, optionally per text-length bucket."""

    t_ai: float
    t_human: float
    target_fpr: float
    band_collapsed: bool = False
    per_bucket: dict[str, dict[str, float]] | None = None
    achieved: dict[str, float] | None = None

    def for_length(self, words: int) -> tuple[float, float]:
        """Return (t_human, t_ai) for a document of ``words`` words."""
        if self.per_bucket:
            entry = self.per_bucket.get(bucket_for(words))
            if entry:
                return entry["t_human"], entry["t_ai"]
        return self.t_human, self.t_ai

    def to_dict(self) -> dict[str, object]:
        return {
            "t_ai": self.t_ai,
            "t_human": self.t_human,
            "target_fpr": self.target_fpr,
            "band_collapsed": self.band_collapsed,
            "per_bucket": self.per_bucket or {},
            "achieved": self.achieved or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ModeThresholds:
        return cls(
            t_ai=float(data["t_ai"]),  # type: ignore[arg-type]
            t_human=float(data["t_human"]),  # type: ignore[arg-type]
            target_fpr=float(data.get("target_fpr", 0.01)),  # type: ignore[arg-type]
            band_collapsed=bool(data.get("band_collapsed", False)),
            per_bucket=data.get("per_bucket") or None,  # type: ignore[arg-type]
            achieved=data.get("achieved") or None,  # type: ignore[arg-type]
        )


def bucket_for(words: int) -> str:
    if words < 100:
        return "<100"
    if words < 250:
        return "100-250"
    if words < 500:
        return "250-500"
    return "500+"


def compute_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    words: np.ndarray | None = None,
    target_fpr: float = 0.01,
    ai_miss: float = 0.05,
) -> ModeThresholds:
    """Select ``t_ai`` (low human FPR) and ``t_human`` (bounded AI miss rate).

    ``t_ai`` is the smallest threshold whose human false-positive rate is at most
    ``target_fpr``.  ``t_human`` is the ``ai_miss`` quantile of AI scores.  If the
    two cross (the classes overlap too much) the band collapses and both take the
    conservative ``t_ai`` value; ``band_collapsed`` records that.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    humans = scores[labels == 0]
    ai = scores[labels == 1]
    if humans.size == 0 or ai.size == 0:
        raise ValueError("threshold selection needs both human and AI validation rows")

    t_ai = best_threshold_at_fpr(humans, target_fpr)
    t_human = float(np.quantile(ai, ai_miss))
    collapsed = t_human >= t_ai
    if collapsed:
        t_human = t_ai

    achieved = {
        "human_fpr_at_t_ai": float(np.mean(humans >= t_ai)),
        "ai_miss_at_t_human": float(np.mean(ai < t_human)),
        "abstention_rate": float(np.mean((scores > t_human) & (scores < t_ai))),
    }

    per_bucket: dict[str, dict[str, float]] = {}
    if words is not None:
        words = np.asarray(words)
        for bucket in LENGTH_BUCKETS:
            mask = np.array([bucket_for(int(w)) == bucket for w in words])
            h = scores[mask & (labels == 0)]
            a = scores[mask & (labels == 1)]
            # Only per-length thresholds when the bucket has enough of both classes.
            if h.size < MIN_ROWS_PER_BUCKET or a.size < MIN_ROWS_PER_BUCKET:
                continue
            bucket_t_ai = best_threshold_at_fpr(h, target_fpr)
            bucket_t_human = float(np.quantile(a, ai_miss))
            if bucket_t_human >= bucket_t_ai:
                bucket_t_human = bucket_t_ai
            per_bucket[bucket] = {"t_ai": bucket_t_ai, "t_human": bucket_t_human}

    return ModeThresholds(
        t_ai=t_ai,
        t_human=t_human,
        target_fpr=target_fpr,
        band_collapsed=collapsed,
        per_bucket=per_bucket or None,
        achieved=achieved,
    )


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Standard 10-bin ECE, used by the evaluation script."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == bins - 1
        mask = (scores >= lo) & ((scores <= hi) if last else (scores < hi))
        if mask.sum() == 0:
            continue
        total += mask.sum() * abs(scores[mask].mean() - labels[mask].mean())
    return float(total / max(1, scores.size))


def verdict_from(probability: float, t_human: float, t_ai: float) -> str:
    """Three-state verdict with an explicit abstain zone."""
    if probability >= t_ai:
        return "likely_ai"
    if probability <= t_human:
        return "likely_human"
    return "uncertain"
