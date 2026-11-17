"""Binoculars: zero-shot machine-text scoring from two paired models (Hans et al., 2024).

Score = mean performer log-perplexity / cross-perplexity between the observer and
performer distributions.  Machine text sits *below* human text on this ratio.

This module is strictly optional: without ``torch``/``transformers`` or the model
weights it reports itself unavailable and the ``deep`` mode says so instead of
pretending to have run it.
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
class BinocularsScore:
    score: float
    log_ppl: float
    x_ppl: float
    tokens_scored: int
    truncated: bool


class Binoculars:
    """Lazily loaded observer/performer pair sharing one tokenizer."""

    name = "binoculars"

    def __init__(
        self,
        observer_name: str | None = None,
        performer_name: str | None = None,
    ) -> None:
        self.observer_name = observer_name or settings.binoculars_observer
        self.performer_name = performer_name or settings.binoculars_performer
        self._observer: Any = None
        self._performer: Any = None
        self._tokenizer: Any = None
        self._error: str | None = None
        self._tried = False

    def load(self) -> bool:
        """Load both models fp16 on GPU, fp32 on CPU. Returns success."""
        if self._tried:
            return self._observer is not None
        self._tried = True
        try:  # pragma: no cover - requires torch + weights
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.observer_name)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._observer = AutoModelForCausalLM.from_pretrained(
                self.observer_name, torch_dtype=dtype
            ).eval().to(device)
            self._performer = AutoModelForCausalLM.from_pretrained(
                self.performer_name, torch_dtype=dtype
            ).eval().to(device)
            log.info("binoculars loaded: %s / %s", self.observer_name, self.performer_name)
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self._error = f"Binoculars unavailable: {exc}"
            log.warning(self._error)
        return self._observer is not None

    @property
    def error(self) -> str | None:
        return self._error

    def score(self, text: str, max_tokens: int | None = None) -> BinocularsScore:
        """Compute the Binoculars ratio over 512-token chunks."""
        if not self.load():
            raise RuntimeError(self._error or "binoculars unavailable")
        return self._score_chunks(text, max_tokens or MAX_TOKENS_PER_CHUNK)

    def _score_chunks(self, text: str, max_tokens: int) -> BinocularsScore:  # pragma: no cover
        import torch

        assert self._tokenizer is not None
        ids = self._tokenizer(text, return_tensors="pt").input_ids
        device = next(self._observer.parameters()).device
        ids = ids.to(device)
        log_ppls: list[float] = []
        x_ppls: list[float] = []
        with torch.no_grad():
            for start in range(0, max(1, ids.shape[1] - 1), max_tokens):
                chunk = ids[:, start : start + max_tokens]
                if chunk.shape[1] < 2:
                    continue
                obs_logits = self._observer(chunk).logits.float()
                perf_logits = self._performer(chunk).logits.float()
                obs_logp = torch.log_softmax(obs_logits[:, :-1], dim=-1)
                perf_logp = torch.log_softmax(perf_logits[:, :-1], dim=-1)
                targets = chunk[:, 1:]
                perf_nll = -perf_logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                cross = -(obs_logp.exp() * perf_logp).sum(-1)
                log_ppls.append(float(perf_nll.mean()))
                x_ppls.append(float(cross.mean()))
        if not log_ppls:
            return BinocularsScore(0.0, 0.0, 0.0, 0, False)
        log_ppl = sum(log_ppls) / len(log_ppls)
        x_ppl = sum(x_ppls) / len(x_ppls)
        ratio = log_ppl / x_ppl if x_ppl > 0 else math.inf
        return BinocularsScore(
            score=ratio,
            log_ppl=log_ppl,
            x_ppl=x_ppl,
            tokens_scored=int(ids.shape[1]),
            truncated=int(ids.shape[1]) > max_tokens,
        )
