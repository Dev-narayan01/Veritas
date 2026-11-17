"""Fast-DetectGPT: analytic sampling-discrepancy score (Bao et al., 2024).

With a single scoring model used as its own reference distribution, compute

    ll     = sum_t log p(x_t | x_<t)
    mu     = sum_t sum_v p(v) log p(v)
    sigma2 = sum_t [ sum_v p(v) (log p(v))^2 - (sum_v p(v) log p(v))^2 ]
    score  = (ll - mu) / sqrt(sigma2)

Higher scores mean the text is more machine-like.  Optional, like Binoculars.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..config import settings
from ..logging_setup import get_logger

log = get_logger(__name__)

MAX_TOKENS_PER_CHUNK = 512


@dataclass
class FastDetectScore:
    score: float
    log_likelihood: float
    mu: float
    sigma: float
    tokens_scored: int


class FastDetect:
    """Lazily loaded single-model scorer (Qwen2.5-0.5B, fallback GPT-2)."""

    name = "fastdetect"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.fastdetect_model
        self._model: Any = None
        self._tokenizer: Any = None
        self._error: str | None = None
        self._tried = False

    def load(self) -> bool:
        if self._tried:
            return self._model is not None
        self._tried = True
        for candidate in (self.model_name, "gpt2"):
            try:  # pragma: no cover - requires torch + weights
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(candidate)
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                model = AutoModelForCausalLM.from_pretrained(candidate, torch_dtype=dtype)
                model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
                self._model, self._tokenizer, self.model_name = model, tokenizer, candidate
                log.info("fastdetect loaded: %s", candidate)
                return True
            except Exception as exc:  # noqa: BLE001 - optional dependency
                self._error = f"FastDetect unavailable ({candidate}): {exc}"
                log.warning(self._error)
        return False

    @property
    def error(self) -> str | None:
        return self._error

    def score(self, text: str, max_tokens: int | None = None) -> FastDetectScore:
        """Compute the normalised sampling discrepancy."""
        if not self.load():
            raise RuntimeError(self._error or "fastdetect unavailable")
        return self._score(text, max_tokens or MAX_TOKENS_PER_CHUNK)

    def _score(self, text: str, max_tokens: int) -> FastDetectScore:  # pragma: no cover
        import torch

        assert self._tokenizer is not None
        ids = self._tokenizer(text, return_tensors="pt").input_ids
        ids = ids[:, :max_tokens].to(next(self._model.parameters()).device)
        if ids.shape[1] < 2:
            return FastDetectScore(0.0, 0.0, 0.0, 0.0, 0)
        with torch.no_grad():
            logits = self._model(ids).logits.float()[:, :-1]
            logp = torch.log_softmax(logits, dim=-1)
            probs = logp.exp()
            targets = ids[:, 1:]
            ll = logp.gather(2, targets.unsqueeze(-1)).squeeze(-1).sum()
            mu = (probs * logp).sum(-1).sum()
            second = (probs * logp.pow(2)).sum(-1).sum()
            first_sq = ((probs * logp).sum(-1) ** 2).sum()
            sigma2 = float(second - first_sq)
        sigma = math.sqrt(sigma2) if sigma2 > 0 else 1.0
        return FastDetectScore(
            score=float((ll - mu) / sigma),
            log_likelihood=float(ll),
            mu=float(mu),
            sigma=sigma,
            tokens_scored=int(ids.shape[1]),
        )
