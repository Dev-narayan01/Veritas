"""Mode stackers: combine component scores into one calibrated probability.

For each mode (``lite``, ``fast``, ``standard``, ``deep``) a logistic regression is
fitted on ``val_stack`` over the logits of each base component plus the standardised
zero-shot scores and ``log(word_count)``.  The scaler statistics are stored in the
artifact so inference standardises exactly like training did.

Only the components a mode declares may contribute features; a mode whose
component list cannot be fully satisfied is reported unavailable by the runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..config import MODE_COMPONENTS
from ..logging_setup import get_logger
from .calibration import ModeThresholds, PlattScaler

log = get_logger(__name__)

_EPS = 1e-6


def logit(probability: float | None) -> float:
    """Convert a probability to a logit, clamping to avoid infinities."""
    if probability is None:
        return 0.0
    p = min(1.0 - _EPS, max(_EPS, float(probability)))
    return math.log(p / (1.0 - p))


@dataclass
class StackerArtifact:
    """A fitted stacker plus the calibration and thresholds that go with it."""

    mode: str
    feature_names: list[str]
    classifier: Any
    scaler_mean: dict[str, float] = field(default_factory=dict)
    scaler_std: dict[str, float] = field(default_factory=dict)
    platt: PlattScaler | None = None
    thresholds: ModeThresholds | None = None
    components: list[str] = field(default_factory=list)
    trained_rows: int = 0

    def vector(self, features: dict[str, float]) -> list[float]:
        """Order features, standardise the scaled ones and append log(word_count)."""
        out: list[float] = []
        for name in self.feature_names:
            value = float(features.get(name, 0.0))
            if name in self.scaler_mean:
                std = self.scaler_std.get(name) or 1.0
                value = (value - self.scaler_mean[name]) / std
            out.append(value)
        return out

    def decision(self, features: dict[str, float]) -> float:
        vec = np.asarray([self.vector(features)], dtype=float)
        return float(self.classifier.decision_function(vec)[0])

    def probability(self, features: dict[str, float]) -> float:
        decision = self.decision(features)
        if self.platt is None:
            return float(self.classifier.predict_proba(np.asarray([self.vector(features)]))[0, 1])
        return float(self.platt.transform(np.asarray([decision]))[0])

    def to_meta(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "platt": self.platt.to_dict() if self.platt else None,
            "thresholds": self.thresholds.to_dict() if self.thresholds else None,
            "components": self.components,
            "trained_rows": self.trained_rows,
        }


def feature_row(component_probs: dict[str, float | None], zero_shot: dict[str, float],
                words: int, mode: str) -> dict[str, float]:
    """Build the raw feature dict for one document under one mode."""
    row: dict[str, float] = {"log_word_count": math.log(max(1, words))}
    for component in MODE_COMPONENTS[mode]:
        if component in {"classical", "ppl_gpt2", "transformer"}:
            row[f"logit_{component}"] = logit(component_probs.get(component))
        else:
            row[component] = float(zero_shot.get(component, 0.0))
    return row


def fit_stacker(
    mode: str,
    rows: list[dict[str, float]],
    labels: list[int],
    standardise: tuple[str, ...] = ("binoculars", "fastdetect"),
) -> StackerArtifact:
    """Fit the stacker for one mode on the ``val_stack`` rows."""
    from sklearn.linear_model import LogisticRegression

    feature_names = sorted({k for row in rows for k in row})
    mean = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in feature_names
            if k in standardise}
    std = {
        k: float(np.std([r.get(k, 0.0) for r in rows]) or 1.0)
        for k in feature_names if k in standardise
    }
    art = StackerArtifact(
        mode=mode, feature_names=feature_names, classifier=None, scaler_mean=mean,
        scaler_std=std, components=list(MODE_COMPONENTS[mode]), trained_rows=len(rows),
    )
    matrix = np.asarray([art.vector(r) for r in rows], dtype=float)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(matrix, labels)
    art.classifier = clf
    log.info("stacker[%s] fitted on %s rows with %s features", mode, len(rows), len(feature_names))
    return art


def save(artifact: StackerArtifact, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load(path: str | Path) -> StackerArtifact:
    return joblib.load(path)
