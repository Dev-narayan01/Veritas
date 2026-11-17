"""Detector internals: calibration, thresholds, segmentation, tokens, mode assembly."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.config import MODE_COMPONENTS, settings
from app.detector import calibration
from app.detector.segmentation import segment_document, viterbi
from app.detector.tokens import token_view
from app.schemas import SentenceScore

ARTIFACTS = settings.artifact_path


def test_platt_scaling_is_monotone_and_bounded() -> None:
    raw = np.linspace(-3, 3, 25)
    scaler = calibration.PlattScaler(a=1.2, b=-0.3)
    probabilities = scaler.transform(raw)
    assert probabilities.shape == raw.shape
    assert np.all(np.diff(probabilities) > 0), "Platt scaling must be monotone"
    assert probabilities.min() > 0 and probabilities.max() < 1


def test_platt_scaler_round_trips_through_json() -> None:
    scaler = calibration.PlattScaler(a=2.5, b=-1.5)
    restored = calibration.PlattScaler.from_dict(json.loads(json.dumps(scaler.to_dict())))
    assert (restored.a, restored.b) == (2.5, -1.5)


def test_platt_fit_separates_two_classes() -> None:
    humans = np.linspace(0.0, 0.4, 60)
    ai = np.linspace(0.6, 1.0, 60)
    scaler = calibration.PlattScaler.fit(np.concatenate([humans, ai]), np.array([0] * 60 + [1] * 60))
    assert scaler.transform(np.array([0.05]))[0] < 0.2
    assert scaler.transform(np.array([0.95]))[0] > 0.8


def test_collapsed_calibration_is_detected() -> None:
    with pytest.raises(calibration.CalibrationCollapsed):
        calibration.assert_score_diversity(np.full(500, 0.42))
    assert calibration.assert_score_diversity(np.linspace(0, 1, 500)) == 500


def test_threshold_at_fixed_false_positive_rate() -> None:
    rng = np.random.default_rng(42)
    humans = rng.beta(1, 6, size=2000)
    threshold = calibration.best_threshold_at_fpr(humans, 0.01)
    assert 0.0 <= threshold <= 1.0
    assert float(np.mean(humans >= threshold)) <= 0.01 + 1e-9


def test_tpr_at_fpr_reports_operating_point() -> None:
    rng = np.random.default_rng(42)
    humans = rng.beta(1, 8, size=2000)
    ai = rng.beta(8, 1, size=2000)
    scores = np.concatenate([humans, ai])
    labels = np.array([0] * len(humans) + [1] * len(ai))
    tpr, threshold = calibration.tpr_at_fpr(scores, labels, 0.01)
    assert 0.0 <= tpr <= 1.0
    assert float(np.mean(humans >= threshold)) <= 0.01 + 1e-9


def test_verdict_uses_the_uncertain_band() -> None:
    assert calibration.verdict_from(0.05, t_human=0.25, t_ai=0.7) == "likely_human"
    assert calibration.verdict_from(0.45, t_human=0.25, t_ai=0.7) == "uncertain"
    assert calibration.verdict_from(0.9, t_human=0.25, t_ai=0.7) == "likely_ai"


def test_expected_calibration_error_bounds() -> None:
    perfect = np.array([0.0, 1.0] * 50)
    labels = np.array([0, 1] * 50)
    assert calibration.expected_calibration_error(perfect, labels) == pytest.approx(0.0, abs=1e-6)
    wrong = np.array([1.0, 0.0] * 50)
    assert calibration.expected_calibration_error(wrong, labels) > 0.9


def test_viterbi_produces_one_label_per_sentence() -> None:
    scores = [0.9] * 6 + [0.1] * 6
    labels = viterbi(scores)
    assert len(labels) == len(scores)
    assert set(labels) <= {0, 1}
    assert labels[:6] == [1] * 6, "a run of high scores is one AI segment"
    assert labels[6:] == [0] * 6


def test_viterbi_keeps_segments_coherent() -> None:
    # A single low score inside a run of high ones must not split the segment: that is
    # what the stay probability is for, and it is what makes the stripe in the UI readable.
    labels = viterbi([0.9, 0.9, 0.15, 0.9, 0.9])
    assert len(set(labels)) == 1


def test_segment_document_covers_every_sentence() -> None:
    sentences = [
        SentenceScore(index=i, start=i * 10, end=(i + 1) * 10, text=f"sentence {i} here", ai_prob=prob)
        for i, prob in enumerate([0.95] * 4 + [0.05] * 4)
    ]
    segments, ai_fraction = segment_document(sentences)
    assert segments[0].start == 0
    assert segments[-1].end == 80
    assert {segment.label for segment in segments} == {"ai", "human"}
    assert 0.0 <= ai_fraction <= 1.0
    for first, second in zip(segments, segments[1:]):
        assert first.end <= second.start


def test_token_view_flags_the_most_predictable_tokens() -> None:
    text = "The the the cat sat on the mat and the the dog."
    tokens = token_view(text, [], None)
    assert tokens
    assert any(token.flagged for token in tokens)
    for token in tokens:
        assert 0 <= token.start < token.end <= len(text)
        assert text[token.start : token.end] == token.token


def test_every_mode_declares_its_components() -> None:
    assert set(MODE_COMPONENTS) == {"lite", "fast", "standard", "deep"}
    for mode, components in MODE_COMPONENTS.items():
        assert components, f"{mode} must declare at least one component"
        assert len(set(components)) == len(components), f"{mode} lists a component twice"
    # Upgrade path: each mode is a strict superset of the previous one.
    order = ["lite", "fast", "standard", "deep"]
    for earlier, later in zip(order, order[1:]):
        assert set(MODE_COMPONENTS[earlier]) <= set(MODE_COMPONENTS[later])


def test_thresholds_artifact_is_truthful() -> None:
    path = ARTIFACTS / "thresholds.json"
    if not path.exists():
        pytest.skip("thresholds.json has not been produced in this checkout")
    payload = json.loads(path.read_text())
    for mode, entry in payload.get("modes", {}).items():
        assert mode in MODE_COMPONENTS
        if entry.get("fit"):
            assert entry["t_human"] is not None and entry["t_ai"] is not None
            assert entry["t_human"] <= entry["t_ai"], mode
            assert 0.0 <= entry["t_human"] <= 1.0 and 0.0 <= entry["t_ai"] <= 1.0
        else:
            assert entry.get("reason"), f"{mode} is unfitted and must say why"
