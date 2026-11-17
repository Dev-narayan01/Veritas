"""Fine-tuned transformer component (roberta-base / distilroberta-base).

Weights live in the artifact bundle (or on the Hub repo) and are loaded lazily.
Inference slides a 256-token window with stride 128 over long documents and
averages the window logits, so a 5,000-word essay is scored in full instead of
being truncated to its first paragraph.

The label mapping is verified against the saved config before any score is used:
a silently flipped id2label would invert the whole component.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import settings
from ..logging_setup import get_logger

log = get_logger(__name__)

WINDOW = 256
STRIDE = 128


@dataclass
class TransformerModel:
    """Loaded tokenizer/model pair with a verified AI-class index."""

    model: Any
    tokenizer: Any
    ai_index: int
    device: str


class TransformerDetector:
    """Runtime wrapper reporting availability honestly."""

    name = "transformer"

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._state: TransformerModel | None = None
        self._error: str | None = None
        self._tried = False

    @property
    def available(self) -> bool:
        self._ensure()
        return self._state is not None

    @property
    def reason(self) -> str | None:
        self._ensure()
        return self._error

    def _ensure(self) -> None:
        if self._tried:
            return
        self._tried = True
        if not self.model_dir.exists():
            self._error = f"weights missing: {self.model_dir}"
            return
        try:  # pragma: no cover - requires torch + weights
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.eval().to(device)
            id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
            ai_index = None
            for idx, label in id2label.items():
                if label in {"ai", "label_1", "chatgpt", "machine", "fake", "generated"}:
                    ai_index = idx
                    break
            if ai_index is None:
                raise RuntimeError(f"cannot determine AI class from id2label={id2label}")
            self._state = TransformerModel(model, tokenizer, ai_index, device)
            log.info("transformer loaded from %s (ai_index=%s)", self.model_dir, ai_index)
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self._error = f"transformer unavailable: {exc}"
            log.warning(self._error)

    def score(self, text: str) -> dict[str, float]:  # pragma: no cover - needs torch
        """Average AI probability over sliding windows."""
        self._ensure()
        if self._state is None:
            raise RuntimeError(self._error or "transformer unavailable")
        import torch

        state = self._state
        max_len = settings.transformer_max_len
        enc = state.tokenizer(text, return_tensors="pt", truncation=False)
        ids = enc["input_ids"][0]
        if ids.numel() <= max_len:
            windows = [ids]
        else:
            windows = [
                ids[start : start + max_len]
                for start in range(0, max(1, ids.numel() - max_len + STRIDE), STRIDE)
            ]
        probs: list[float] = []
        with torch.no_grad():
            for window in windows:
                if window.numel() < 4:
                    continue
                batch = {
                    "input_ids": window.unsqueeze(0).to(state.device),
                    "attention_mask": torch.ones(1, window.numel(), dtype=torch.long).to(state.device),
                }
                logits = state.model(**batch).logits.float()
                probs.append(float(torch.softmax(logits, dim=-1)[0, state.ai_index]))
        probability = float(np.mean(probs)) if probs else 0.5
        return {"probability": probability, "windows": float(len(probs))}
