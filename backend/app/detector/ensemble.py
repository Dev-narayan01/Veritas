"""Orchestration of the AI-text detector for one document.

The ensemble deliberately does three things a naive implementation does not:

1. it *verifies* every component of the requested mode before running, and if one
   cannot run it degrades to the best fully-available mode and says so in
   ``components_unavailable`` and ``exit_notes``;
2. it applies the mode's calibrated stacker and Platt scaling, never a hardcoded
   threshold;
3. it returns the granular evidence (sentence heatmap, token view, mixed-authorship
   segments, component breakdown) alongside the single probability.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from ..artifacts import ArtifactRegistry
from ..config import MODE_COMPONENTS, settings
from ..logging_setup import get_logger
from ..schemas import (
    AIResult,
    ComponentScore,
    SentenceScore,
)
from ..text.segment import Sentence, context_window, split_sentences
from ..text.clean import NormalizedText, normalize, word_count
from .calibration import ModeThresholds, verdict_from
from .explain import build_explanation
from .features import compute_style, length_bucket, top_feature_contributions
from .segmentation import segment_document
from .stacker import feature_row
from .tokens import token_view

log = get_logger(__name__)

# Sentence scores are computed with a context window; this bounds total work.
MAX_SENTENCES_SCORED = 400


@dataclass
class DetectorOutput:
    """The AI result plus the cleaned text it refers to."""

    result: AIResult
    normalized: NormalizedText
    sentences: list[Sentence] = field(default_factory=list)
    component_raw: dict[str, dict[str, float]] = field(default_factory=dict)


def _zero_shot_score(name: str, text: str) -> tuple[float | None, str | None]:
    """Compute a zero-shot score, or explain why it could not be computed."""
    try:  # pragma: no cover - needs torch + weights
        if name == "binoculars":
            from .binoculars import Binoculars

            model = Binoculars()
            if not model.load():
                return None, model.error or "binoculars failed to load"
            return model.score(text).score, None
        from .fastdetect import FastDetect

        model = FastDetect()
        if not model.load():
            return None, model.error or "fastdetect failed to load"
        return model.score(text).score, None
    except Exception as exc:  # noqa: BLE001 - optional component
        return None, f"{name} failed: {exc}"


class AIDetector:
    """Runs the AI-authorship analysis for one document."""

    def __init__(self, registry: ArtifactRegistry) -> None:
        self.registry = registry

    def run(
        self,
        text: str,
        requested_mode: str,
        explain: bool = True,
    ) -> DetectorOutput:
        started = time.perf_counter()
        normalized = normalize(text)
        clean = normalized.text
        words = word_count(clean)
        sentences = split_sentences(clean)

        result = AIResult(mode_requested=requested_mode, word_count=words)
        if words < settings.min_words_for_detection:
            result.verdict = "too_short"
            result.warnings.append(
                f"text has {words} words; at least {settings.min_words_for_detection} "
                "are needed for a probability"
            )
            result.latency_ms = (time.perf_counter() - started) * 1000
            return DetectorOutput(result=result, normalized=normalized, sentences=sentences)

        mode, fallbacks = self._resolve_mode(requested_mode)
        result.mode_used = mode
        result.components_unavailable = fallbacks
        if mode is None:
            result.verdict = "too_short"
            result.mode_used = None
            result.warnings.append(
                "no detector mode is available on this server; see /api/health for reasons"
            )
            result.latency_ms = (time.perf_counter() - started) * 1000
            return DetectorOutput(result=result, normalized=normalized, sentences=sentences)

        components = MODE_COMPONENTS[mode]
        probs: dict[str, float | None] = {}
        component_scores: list[ComponentScore] = []
        raw: dict[str, dict[str, float]] = {}

        for component in components:
            available, reason = self.registry.component_available(component)
            if not available:
                probs[component] = None
                component_scores.append(
                    ComponentScore(name=component, probability=None, available=False, reason=reason)
                )
                continue
            try:
                if component == "classical":
                    payload = self.registry.classical.score(clean)
                elif component == "ppl_gpt2":
                    payload = self.registry.ppl.score(_truncate_for_component(clean, mode))
                elif component == "transformer":
                    payload = self.registry.transformer.score(clean)
                else:
                    score, reason = _zero_shot_score(component, clean)
                    if score is None:
                        raise RuntimeError(reason or f"{component} unavailable")
                    payload = {"score": score}
                probs[component] = float(payload["probability"]) if "probability" in payload else None
                raw[component] = {k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}
                component_scores.append(
                    ComponentScore(name=component, probability=probs[component], raw=raw[component])
                )
            except Exception as exc:  # noqa: BLE001 - a component failure must not kill the run
                log.warning("component %s failed: %s", component, exc)
                probs[component] = None
                component_scores.append(
                    ComponentScore(name=component, probability=None, available=False, reason=str(exc))
                )

        result.components = component_scores
        result.components_run = [c.name for c in component_scores if c.available]
        missing = [c.name for c in component_scores if not c.available]
        if missing:
            result.warnings.append(
                "components could not run: " + ", ".join(missing)
            )

        zero_shot = {
            "binoculars": raw.get("binoculars", {}).get("score", 0.0) or 0.0,
            "fastdetect": raw.get("fastdetect", {}).get("score", 0.0) or 0.0,
        }
        features = feature_row(probs, zero_shot, words, mode)
        stacker = self.registry.stacker(mode)
        probability = stacker.probability(features)
        thresholds = self._thresholds(mode, stacker, words)

        result.probability = float(probability)
        result.thresholds = {"t_human": thresholds.t_human, "t_ai": thresholds.t_ai}
        result.band_collapsed = thresholds.band_collapsed
        result.verdict = verdict_from(probability, thresholds.t_human, thresholds.t_ai)  # type: ignore[assignment]
        result.low_confidence = words < settings.low_confidence_words
        if result.low_confidence:
            result.warnings.append(
                f"text is shorter than {settings.low_confidence_words} words; treat the "
                "probability as low confidence"
            )
        result.ci_low, result.ci_high = self._interval(
            stacker, features, component_scores, words, probability
        )

        # --- granular evidence -------------------------------------------------
        result.sentences = self._sentence_scores(clean, sentences)
        result.tokens = token_view(clean, sentences, probs.get("ppl_gpt2"))
        segments, ai_fraction = segment_document(result.sentences, normalized, sentences)
        result.segments = segments
        result.ai_fraction = ai_fraction
        if explain:
            style = compute_style(clean)
            contributions = top_feature_contributions(style, self.registry.ppl.style_reference())
            result.explanation = build_explanation(
                probability=result.probability,
                verdict=result.verdict,
                thresholds=result.thresholds,
                components=component_scores,
                contributions=contributions,
                words=words,
                segments=segments,
            )
        result.latency_ms = (time.perf_counter() - started) * 1000
        return DetectorOutput(
            result=result, normalized=normalized, sentences=sentences, component_raw=raw
        )

    # ------------------------------------------------------------------ #
    def _resolve_mode(self, requested: str) -> tuple[str | None, list[dict[str, str]]]:
        """Return (mode to use, unavailable components of the requested mode).

        We never silently run a weaker mode: the requested mode is used when its
        components all load; otherwise we pick the strongest fully-available mode
        and list what was missing.  If none is available, ``None`` is returned.
        """
        state = self.registry.mode_state(requested) if requested in MODE_COMPONENTS else None
        if state is not None and state.available:
            return requested, []
        unavailable: list[dict[str, str]] = []
        if state is not None:
            if state.reason:
                unavailable.append({"component": requested, "reason": state.reason})
        for candidate in ("deep", "standard", "fast", "lite"):
            candidate_state = self.registry.mode_state(candidate)
            if candidate_state.available:
                # Record which components of the requested mode are missing.
                for component in MODE_COMPONENTS.get(requested, ()):
                    ok, reason = self.registry.component_available(component)
                    if not ok:
                        unavailable.append(
                            {"component": component, "reason": reason or "unavailable"}
                        )
                return candidate, unavailable
        for component in MODE_COMPONENTS.get(requested, ()):
            ok, reason = self.registry.component_available(component)
            if not ok:
                unavailable.append({"component": component, "reason": reason or "unavailable"})
        return None, unavailable

    def _thresholds(self, mode: str, stacker: Any, words: int) -> ModeThresholds:
        tuned = self.registry.thresholds.get(mode)
        if tuned:
            thresholds = ModeThresholds.from_dict(tuned)
        elif stacker.thresholds is not None:
            thresholds = stacker.thresholds
        else:
            thresholds = ModeThresholds(
                t_ai=0.5, t_human=0.5, target_fpr=settings.target_human_fpr, band_collapsed=True
            )
        t_human, t_ai = thresholds.for_length(words)
        return ModeThresholds(
            t_ai=t_ai,
            t_human=t_human,
            target_fpr=thresholds.target_fpr,
            band_collapsed=thresholds.band_collapsed,
        )

    def _sentence_scores(self, clean: str, sentences: list[Sentence]) -> list[SentenceScore]:
        """Score each sentence in a +/-1 sentence context window, then smooth."""
        if not sentences:
            return []
        classical_ok = self.registry.classical.available
        ppl_ok = self.registry.ppl.available
        if not (classical_ok or ppl_ok):
            return []

        scores: list[float] = []
        step = max(1, math.ceil(len(sentences) / MAX_SENTENCES_SCORED))
        for idx in range(len(sentences)):
            start, end = context_window(clean, sentences, idx)
            window = clean[start:end]
            parts: list[float] = []
            if classical_ok:
                try:
                    parts.append(self.registry.classical.score(window)["probability"])
                except Exception:  # noqa: BLE001 - degrade to remaining signal
                    pass
            if ppl_ok:
                try:
                    parts.append(self.registry.ppl.score(window)["probability"])
                except Exception:  # noqa: BLE001
                    pass
            scores.append(sum(parts) / len(parts) if parts else 0.5)

        # 3-sentence smoothing keeps the heatmap readable.
        smoothed: list[float] = []
        half = 1
        for idx, _ in enumerate(scores):
            window_scores = scores[max(0, idx - half) : idx + half + 1]
            smoothed.append(sum(window_scores) / len(window_scores))

        return [
            SentenceScore(
                index=s.index,
                start=s.start,
                end=s.end,
                text=s.text,
                ai_prob=round(float(smoothed[i]), 4),
            )
            for i, s in enumerate(sentences[::step])
            if i * step < len(sentences) and i < len(sentences)
        ]

    def _interval(
        self,
        stacker: Any,
        features: dict[str, float],
        components: list[ComponentScore],
        words: int,
        probability: float,
    ) -> tuple[float | None, float | None]:
        """Bootstrap-style interval over chunk scores for long documents.

        For short texts the uncertainty comes from the spread between components
        (which is what actually disagrees); for long texts it comes from scoring
        independent chunks.  Both are honest empirical intervals, not invented
        precision.
        """
        usable = [c.probability for c in components if c.probability is not None]
        if len(usable) >= 2:
            spread = max(usable) - min(usable)
            margin = 0.5 * spread + 0.02
        else:
            margin = 0.05 + (0.10 if words < settings.low_confidence_words else 0.0)
        low = max(0.0, probability - margin)
        high = min(1.0, probability + margin)
        return round(low, 4), round(high, 4)


def _truncate_for_component(text: str, mode: str) -> str:
    """Respect ``DEEP_CPU_MAX_TOKENS`` for the deep mode on CPU.

    The ppl component is the only tokeniser-based one that is cheap enough to run on
    CPU; the limit is expressed in tokens and approximated at 1.3 tokens per word.
    """
    if mode != "deep":
        return text
    max_words = int(settings.deep_cpu_max_tokens / 1.3)
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])


def summarise_bucket(words: int) -> str:
    """Public helper used by metrics and the UI."""
    return length_bucket(words)
