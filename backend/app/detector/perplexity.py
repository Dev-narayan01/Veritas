"""Token-statistics component ("ppl"): language-model surprisal plus stylometry.

The spec's primary backend is GPT-2 small.  GPT-2 weights are an optional download;
when ``transformers`` and the weights are unavailable, the component falls back to
an n-gram language model trained on the project's own training split and reports
``backend="ngram_lm"`` in the health output and in every result.  Both backends
produce the same feature vector, so the downstream stacker is unchanged.

Features (per text):
    lm_mean_logprob, lm_std_logprob  - average / dispersion of token log-probability
    burstiness                      - std of the difference between consecutive logprobs
    mean_rank_pct, top10_hit_rate   - how "predictable" each token was
    entropy_mean                    - mean predictive entropy of the context
    + the stylometry features from :mod:`features`
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

from ..config import settings
from ..logging_setup import get_logger
from .features import StyleFeatures, compute_style

log = get_logger(__name__)

Backend = Literal["gpt2", "ngram_lm"]
_VOCAB_CAP = 60_000


# --------------------------------------------------------------------------- #
# n-gram language model (no pretrained weights needed)
# --------------------------------------------------------------------------- #
@dataclass
class NgramLM:
    """A stupid-backoff trigram model over whitespace tokens."""

    order: int = 3
    counts: dict[tuple[str, ...], Counter] = field(default_factory=dict)
    vocab_size: int = 0
    total_tokens: int = 0

    @classmethod
    def train(cls, texts: list[str], order: int = 3) -> NgramLM:
        """Count n-grams over the mixture of human and AI training text."""
        counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        token_counts: Counter = Counter()
        for text in texts:
            tokens = text.split()
            token_counts.update(tokens)
            for i, token in enumerate(tokens):
                for n in range(1, order + 1):
                    if i + 1 < n:
                        continue
                    context = tuple(tokens[i - n + 1 : i])
                    counts[context][token] += 1
        vocab = [t for t, _ in token_counts.most_common(_VOCAB_CAP)]
        keep = set(vocab)
        pruned = {ctx: Counter({t: c for t, c in cnt.items() if t in keep})
                  for ctx, cnt in counts.items()}
        return cls(
            order=order,
            counts={ctx: cnt for ctx, cnt in pruned.items() if cnt},
            vocab_size=len(keep),
            total_tokens=max(1, sum(token_counts.values())),
        )

    def _context_counter(self, context: tuple[str, ...]) -> Counter:
        for n in range(len(context), 0, -1):
            counter = self.counts.get(tuple(context[-n:]))
            if counter:
                return counter
        return Counter()

    def distribution(self, context: tuple[str, ...]) -> dict[str, float]:
        """Smoothed next-token distribution for a context."""
        counter = self._context_counter(context)
        total = sum(counter.values())
        if not counter:
            return {"<unk>": 1.0}
        alpha = 0.1
        denom = total + alpha * self.vocab_size
        return {
            token: (count + alpha) / denom for token, count in counter.most_common(200)
        }

    def token_logprob(self, token: str, context: tuple[str, ...]) -> float:
        """Log-probability with stupid backoff (0.4 per level) and an OOV floor."""
        counter = self._context_counter(context)
        if counter:
            total = sum(counter.values())
            if counter.get(token):
                return math.log(counter[token] / total)
        # Back off: unigram probability with a smoothing floor.
        unigram = self.counts.get((), Counter())
        floor = 1.0 / (max(1, self.total_tokens) * 100)
        prob = (unigram.get(token, 0) / max(1, self.total_tokens)) if unigram else floor
        return math.log(max(prob, floor) * 0.4)

    def rank_percentile(self, token: str, context: tuple[str, ...]) -> float:
        """Percentile of the observed token in the predicted distribution (0=top)."""
        dist = self.distribution(context)
        ordered = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (candidate, _) in enumerate(ordered):
            if candidate == token:
                return rank / max(1, len(ordered))
        return 1.0


# --------------------------------------------------------------------------- #
# GPT-2 backend (optional)
# --------------------------------------------------------------------------- #
class Gpt2Backend:
    """Lazy GPT-2 wrapper computing per-token statistics with ``torch.no_grad``."""

    name = "gpt2"

    # Process-wide cache of loaded backends, keyed by model name. Loading GPT-2 costs
    # ~500 MB of disk reads; PerplexityArtifact created one per document before this.
    _shared: dict[str, "Gpt2Backend"] = {}

    def __init__(self, model_name: str | None = None, max_tokens: int = 512) -> None:
        self.model_name = model_name or settings.ppl_model
        self.max_tokens = max_tokens
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._error: str | None = None
        self._tried = False

    def load(self) -> bool:
        """Load tokenizer + model once per process per model name."""

        shared = Gpt2Backend._shared.get(self.model_name)
        if shared is not None and shared._model is not None:
            # Reuse: train_perplexity_model.py loads one to decide which backend to
            # use and then builds another to compute features, which read 548 MB of
            # weights from disk twice.
            self._model, self._tokenizer = shared._model, shared._tokenizer
            self._device, self._tried = shared._device, True
            return True
        if self._tried:
            return self._model is not None
        self._tried = True
        try:  # pragma: no cover - requires torch + weights
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # device is remembered rather than recomputed: the tensors have to follow the
            # model, and the model was silently placed on CUDA while the token ids stayed
            # on the CPU. That path had never run, so it raised on the first document
            # ("index is on cpu, different from other tensors on cuda:0").
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._model = (
                AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype)
                .eval()
                .to(self._device)
            )
            log.info("ppl backend gpt2 loaded (%s, %s, %s)",
                     self.model_name, dtype, self._device)
            Gpt2Backend._shared[self.model_name] = self
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self._error = f"GPT-2 unavailable: {exc}"
            log.warning(self._error)
        return self._model is not None

    @property
    def error(self) -> str | None:
        return self._error

    def features(self, text: str) -> dict[str, float]:  # pragma: no cover - needs torch
        import torch

        assert self._tokenizer is not None and self._model is not None
        ids = self._tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=self.max_tokens).input_ids.to(self._device)
        with torch.no_grad():
            logits = self._model(ids).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
        token_lp = logprobs[0, :-1].gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1).cpu().numpy()
        probs = logprobs[0, :-1].exp()
        entropy = (-(probs * logprobs[0, :-1]).sum(dim=-1)).cpu().numpy()
        ranks = (probs > probs.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1).unsqueeze(1)).sum(1)
        rank_pct = (ranks.float() / probs.shape[1]).cpu().numpy()
        return _summarise_lm(token_lp, entropy, rank_pct)


def _summarise_lm(
    token_lp: np.ndarray, entropy: np.ndarray, rank_pct: np.ndarray
) -> dict[str, float]:
    """Reduce per-token arrays to the summary features the classifier consumes."""
    if token_lp.size == 0:
        return {
            "lm_mean_logprob": 0.0,
            "lm_std_logprob": 0.0,
            "burstiness": 0.0,
            "mean_rank_pct": 0.5,
            "top10_hit_rate": 0.0,
            "entropy_mean": 0.0,
        }
    diffs = np.diff(token_lp)
    return {
        "lm_mean_logprob": float(np.mean(token_lp)),
        "lm_std_logprob": float(np.std(token_lp)),
        "burstiness": float(np.std(diffs)) if diffs.size else 0.0,
        "mean_rank_pct": float(np.mean(rank_pct)),
        "top10_hit_rate": float(np.mean(rank_pct <= 0.10)),
        "entropy_mean": float(np.mean(entropy)),
    }


def ngram_features(text: str, lm: NgramLM, max_tokens: int = 512) -> dict[str, float]:
    """Compute the same summary features with the n-gram model."""
    tokens = text.split()[:max_tokens]
    if len(tokens) < 2:
        return _summarise_lm(np.array([]), np.array([]), np.array([]))
    logprobs: list[float] = []
    ranks: list[float] = []
    entropies: list[float] = []
    history: list[str] = []
    for token in tokens:
        context = tuple(history[-(lm.order - 1):]) if lm.order > 1 else ()
        logprobs.append(lm.token_logprob(token, context))
        ranks.append(lm.rank_percentile(token, context))
        dist = lm.distribution(context)
        entropies.append(-sum(p * math.log(p) for p in dist.values() if p > 0))
        history.append(token)
    return _summarise_lm(np.array(logprobs), np.array(entropies), np.array(ranks))


# --------------------------------------------------------------------------- #
# The trained artifact
# --------------------------------------------------------------------------- #
@dataclass
class PerplexityArtifact:
    """Everything the ppl component needs at inference time."""

    backend: Backend
    classifier: Any
    feature_names: list[str]
    lm: NgramLM | None = None
    style_reference: dict[str, dict[str, float]] = field(default_factory=dict)
    model_name: str = ""
    # Cached Gpt2Backend. Loading GPT-2 costs 500 MB of download and a few seconds of
    # GPU transfer, and this used to be constructed fresh inside features() -- i.e. once
    # per document. It is excluded from pickling (see __getstate__) because the weights
    # are restored from the Hub/cache on first use, not shipped in the artifact.
    _gpt2: Any = field(default=None, repr=False, compare=False)

    def _gpt2_backend(self) -> "Gpt2Backend":
        """The one loaded GPT-2 for this artifact, created on first use."""
        if self._gpt2 is None:
            backend = Gpt2Backend(self.model_name)
            if not backend.load():
                # Raised, not defaulted: writing 0.0 for the LM features would look
                # like a score instead of a failure and would poison any classifier
                # fitted on it.
                raise RuntimeError(backend.error or "GPT-2 unavailable")
            self._gpt2 = backend
        return self._gpt2

    def features(self, text: str) -> dict[str, float]:
        style: StyleFeatures = compute_style(text)
        feats = style.as_dict()
        if self.backend == "ngram_lm":
            if self.lm is None:
                raise RuntimeError("ngram language model missing from artifact")
            feats.update(ngram_features(text, self.lm))
        else:  # pragma: no cover - needs torch
            feats.update(self._gpt2_backend().features(text))
        return feats

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_gpt2"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._gpt2 = None

    def matrix(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            feats = self.features(text)
            rows.append([float(feats.get(name, 0.0)) for name in self.feature_names])
        return np.asarray(rows, dtype=np.float64)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.classifier.predict_proba(self.matrix(texts))[:, 1]


def save(artifact: PerplexityArtifact, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load(path: str | Path) -> PerplexityArtifact:
    return joblib.load(path)


class PerplexityDetector:
    """Runtime wrapper with explicit availability reporting."""

    name = "ppl_gpt2"

    def __init__(self, artifact_path: Path, allow_download: bool = False) -> None:
        self.artifact_path = artifact_path
        self.allow_download = allow_download
        self._artifact: PerplexityArtifact | None = None
        self._error: str | None = None
        self._tried = False
        self.backend_note: str = ""

    @property
    def available(self) -> bool:
        self._ensure()
        return self._artifact is not None

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
            self._artifact = load(self.artifact_path)
        except Exception as exc:  # noqa: BLE001
            self._error = f"artifact failed to load: {exc}"
            log.warning("ppl artifact failed: %s", exc)
            return
        self.backend_note = self._artifact.backend
        if self._artifact.backend == "gpt2":  # pragma: no cover - needs torch
            if not Gpt2Backend(self._artifact.model_name).load():
                self._error = "artifact wants GPT-2 but the weights are unavailable"
                self._artifact = None

    def score(self, text: str) -> dict[str, float]:
        self._ensure()
        if self._artifact is None:
            raise RuntimeError(self._error or "ppl model unavailable")
        proba = float(self._artifact.predict_proba([text])[0])
        return {"probability": proba}

    def style_reference(self) -> dict[str, dict[str, float]]:
        self._ensure()
        return self._artifact.style_reference if self._artifact else {}
