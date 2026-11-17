"""Classical detector: TF-IDF plus logistic regression.

Two vectorisers are fitted during training (character 3-5 grams and word 1-2 grams)
and the one with the better validation AUC is kept, so the deployed artifact only
holds the winner.  The model is cheap enough to run on the CPU and is therefore the
component that makes the ``lite`` mode and the Vercel Lite fallback possible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

log = logging.getLogger(__name__)

VARIANTS = ("char", "word")


def build_vectorizer(variant: str):  # noqa: ANN201 - sklearn type
    """Create an *unfitted* TF-IDF vectoriser for the requested variant."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if variant == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=120_000,
            sublinear_tf=True,
            lowercase=True,
        )
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=120_000,
        sublinear_tf=True,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )


@dataclass
class ClassicalModel:
    """A fitted vectoriser + logistic regression pair."""

    vectorizer: Any
    classifier: Any
    variant: str = "char"

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probability of the AI class for each text."""
        matrix = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(matrix)[:, 1]

    def decision(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        return self.classifier.decision_function(matrix)


def train(texts: list[str], labels: list[int], variant: str = "char") -> ClassicalModel:
    """Fit one variant on the training split."""
    from sklearn.linear_model import LogisticRegression

    vectorizer = build_vectorizer(variant)
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", n_jobs=None)
    classifier.fit(matrix, labels)
    log.info("classical[%s] trained on %s rows, %s features", variant, len(texts), matrix.shape[1])
    return ClassicalModel(vectorizer=vectorizer, classifier=classifier, variant=variant)


def pick_better(models: dict[str, ClassicalModel], texts: list[str], labels: list[int]) -> tuple[str, dict[str, float]]:
    """Choose the variant with the higher validation ROC-AUC."""
    from sklearn.metrics import roc_auc_score

    scores: dict[str, float] = {}
    for name, model in models.items():
        probs = model.predict_proba(texts)
        try:
            scores[name] = float(roc_auc_score(labels, probs))
        except ValueError:  # single-class validation slice
            scores[name] = 0.0
    best = max(scores, key=scores.get)
    return best, scores


def save(model: ClassicalModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: str | Path) -> ClassicalModel:
    return joblib.load(path)


class ClassicalDetector:
    """Runtime wrapper that lazily loads the artifact and exposes probabilities."""

    name = "classical"

    def __init__(self, artifact_path: Path) -> None:
        self.artifact_path = artifact_path
        self._model: ClassicalModel | None = None
        self._error: str | None = None
        self._tried = False

    @property
    def available(self) -> bool:
        self._ensure()
        return self._model is not None

    @property
    def reason(self) -> str | None:
        self._ensure()
        return self._error

    def _ensure(self) -> None:
        if self._tried:
            return
        self._tried = True
        if not self.artifact_path.exists():
            self._error = f"artifact missing: {self.artifact_path.name}"
            return
        try:
            self._model = load(self.artifact_path)
        except Exception as exc:  # noqa: BLE001 - corrupt artifact is a runtime condition
            self._error = f"artifact failed to load: {exc}"
            log.warning("classical artifact failed: %s", exc)

    def score(self, text: str) -> dict[str, float]:
        """Return the AI probability and the decision value for one text."""
        self._ensure()
        if self._model is None:
            raise RuntimeError(self._error or "classical model unavailable")
        proba = float(self._model.predict_proba([text])[0])
        decision = float(self._model.decision([text])[0])
        return {"probability": proba, "decision": decision}
