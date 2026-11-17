"""Stage A (dense) and Stage C (paraphrase verification).

Dense retrieval finds passages that mean the same thing without sharing wording.
Because a fixed cosine threshold alone produces false positives, every dense
candidate is verified by a cross-encoder reranker before it is reported as a
paraphrase.  If the reranker cannot load, the cosine bar is raised by 0.05 and the
result carries a warning that says so — the engine never pretends the check ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import settings
from ..logging_setup import get_logger

log = get_logger(__name__)

_RERANK_MARGIN_WITHOUT_RERANKER = 0.05


@dataclass
class DenseHit:
    """A paraphrase candidate passing both thresholds."""

    query_index: int
    corpus_index: int
    cosine: float
    rerank_score: float | None


class Embedder:
    """Sentence-transformers wrapper with an explicit availability flag."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.semantic_model
        self._model: Any = None
        self._error: str | None = None
        self._tried = False

    @property
    def available(self) -> bool:
        return self._ensure() is not None

    @property
    def reason(self) -> str | None:
        self._ensure()
        return self._error

    def _ensure(self) -> Any:
        if self._tried:
            return self._model
        self._tried = True
        try:  # pragma: no cover - requires sentence-transformers + weights
            import warnings

            from sentence_transformers import SentenceTransformer

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = SentenceTransformer(self.model_name)
            log.info("semantic model loaded: %s", self.model_name)
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self._error = f"semantic model unavailable: {exc}"
            log.warning(self._error)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray | None:
        model = self._ensure()
        if model is None:
            return None
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


class Reranker:
    """Cross-encoder that decides whether two passages are genuine paraphrases."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.rerank_model
        self._model: Any = None
        self._error: str | None = None
        self._tried = False

    @property
    def available(self) -> bool:
        return self._ensure() is not None

    @property
    def reason(self) -> str | None:
        self._ensure()
        return self._error

    def _ensure(self) -> Any:
        if self._tried:
            return self._model
        self._tried = True
        try:  # pragma: no cover - requires sentence-transformers + weights
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            log.info("reranker loaded: %s", self.model_name)
        except Exception as exc:  # noqa: BLE001
            self._error = f"reranker unavailable: {exc}"
            log.warning(self._error)
        return self._model

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._ensure()
        if model is None or not pairs:
            return []
        raw = model.predict(pairs)
        values = np.asarray(raw, dtype=float).reshape(-1)
        # Cross-encoders trained on STS/Quora emit logits; squash to 0..1.
        if values.min() < 0.0 or values.max() > 1.0:
            values = 1.0 / (1.0 + np.exp(-values))
        return [float(v) for v in values]


@dataclass
class ParaphraseOutcome:
    hits: list[DenseHit]
    warnings: list[str]
    reranker_used: bool


def verify_paraphrases(
    query_passages: list[str],
    query_vectors: np.ndarray | None,
    corpus_passages: list[str],
    corpus_vectors: np.ndarray | None,
    corpus_indices: list[int],
    cosines: list[float],
    reranker: Reranker,
) -> ParaphraseOutcome:
    """Apply the cosine bar and the reranker to dense candidates."""
    warnings: list[str] = []
    threshold = settings.semantic_min_cosine
    reranker_used = False
    if not reranker.available:
        threshold += _RERANK_MARGIN_WITHOUT_RERANKER
        warnings.append(
            "reranker unavailable: paraphrase matches required a higher cosine "
            f"({threshold:.2f} instead of {settings.semantic_min_cosine:.2f})"
        )
    else:
        reranker_used = True

    hits: list[DenseHit] = []
    pairs: list[tuple[str, str]] = []
    pending: list[tuple[int, int, float]] = []
    for query_index, corpus_index, cosine in zip(
        [c[0] for c in corpus_indices], [c[1] for c in corpus_indices], cosines
    ):
        if cosine < threshold:
            continue
        pairs.append((query_passages[query_index], corpus_passages[corpus_index]))
        pending.append((query_index, corpus_index, float(cosine)))
    if not pending:
        return ParaphraseOutcome([], warnings, reranker_used)

    scores = reranker.score(pairs) if reranker_used else [None] * len(pending)
    for (query_index, corpus_index, cosine), score in zip(pending, scores):
        if score is not None and score < settings.rerank_min_score:
            continue
        hits.append(DenseHit(query_index, corpus_index, cosine, score))
    return ParaphraseOutcome(hits, warnings, reranker_used)
