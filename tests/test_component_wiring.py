"""Two failures where a component silently did nothing, and the site looked fine.

Both were found on the real machine, both after a long run had already succeeded at
everything else:

1. ``train_perplexity_model.py`` constructed ``Gpt2Backend()`` *inside* its per-document
   ``features()`` function, so ``load()`` was never called and every document hit
   ``assert self._tokenizer is not None``. The run downloaded 548 MB of GPT-2 weights,
   printed "ppl backend gpt2 loaded", and then died on the first document.
2. ``train_stackers.py`` decided which components a mode needs with
   ``if c in probs``, where ``probs`` only ever held the three scored components. For
   ``deep`` (which also lists binoculars and fastdetect) that selector came out empty,
   so the "are all components present" check passed trivially and a stacker was fitted
   without them.

Neither is reproducible without GPT-2 or a GPU, so these tests exercise the shape of the
bug: the backend must be loaded once and reused, and the component selector must fail
closed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "backend"))
sys.path.insert(0, str(_ROOT / "ml"))

from app.detector.perplexity import PerplexityArtifact  # noqa: E402
from train_stackers import _mode_components, _rows_for_mode  # noqa: E402


class _FakeBackend:
    """Records how many times it was loaded, so a per-document reload is visible."""

    def __init__(self) -> None:
        self.loads = 0

    def load(self) -> bool:
        self.loads += 1
        return True

    def features(self, text: str) -> dict[str, float]:
        return {"lm_mean_logprob": -1.0}


def test_the_gpt2_backend_is_loaded_once_not_once_per_document() -> None:
    """Loading GPT-2 per document is the bug that killed the ppl run."""
    artifact = PerplexityArtifact(
        backend="gpt2", classifier=None, feature_names=["lm_mean_logprob"],
        model_name="gpt2",
    )
    fake = _FakeBackend()
    artifact._gpt2 = fake  # stand in for an already-loaded backend
    for _ in range(5):
        artifact.features("some text that needs scoring")
    assert fake.loads == 0, "an already-loaded backend must not be reloaded"


def test_a_missing_backend_raises_instead_of_scoring_zero() -> None:
    """A failed load must not turn into a 0.0 feature: that looks like a score."""

    class _Broken:
        error = "GPT-2 unavailable: no network"

        def load(self) -> bool:
            return False

    artifact = PerplexityArtifact(
        backend="gpt2", classifier=None, feature_names=["lm_mean_logprob"],
        model_name="gpt2",
    )
    artifact._gpt2 = None
    # Inject the failing backend through the same attribute the artifact uses.
    import app.detector.perplexity as ppl

    original = ppl.Gpt2Backend
    ppl.Gpt2Backend = lambda *_a, **_k: _Broken()  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="GPT-2 unavailable"):
            artifact.features("some text")
    finally:
        ppl.Gpt2Backend = original  # type: ignore[assignment]


def test_the_backend_is_not_pickled_with_the_artifact() -> None:
    """Weights are restored from cache on load; shipping them in a pkl would be wrong."""
    artifact = PerplexityArtifact(
        backend="gpt2", classifier=None, feature_names=[], model_name="gpt2",
    )
    artifact._gpt2 = _FakeBackend()
    state = artifact.__getstate__()
    assert state["_gpt2"] is None
    restored = PerplexityArtifact.__new__(PerplexityArtifact)
    restored.__setstate__(state)
    assert restored._gpt2 is None


def test_zero_shot_components_are_not_treated_as_scored_columns() -> None:
    """``deep`` needs binoculars and fastdetect; they must not come out of ``probs``."""
    scored, zero_shot = _mode_components("deep")
    assert scored == ("classical", "ppl_gpt2", "transformer")
    assert sorted(zero_shot) == ["binoculars", "fastdetect"]


def test_deep_without_zero_shot_scores_yields_no_rows() -> None:
    """The old selector produced rows here, fitting a stacker that had not run."""
    frame = pd.DataFrame(
        {
            "id": [str(i) for i in range(60)],
            "label": [i % 2 for i in range(60)],
            "text": ["x"] * 60,
            "words": [100] * 60,
            "p_classical": np.linspace(0.1, 0.9, 60),
            "p_ppl_gpt2": np.linspace(0.1, 0.9, 60),
            "p_transformer": np.linspace(0.1, 0.9, 60),
        }
    )
    rows, labels = _rows_for_mode(frame, "deep")
    assert rows == [] and labels == []


def test_a_mode_with_every_scored_component_present_is_fitted() -> None:
    """``standard`` needs only the three scored components, so it must produce rows."""
    frame = pd.DataFrame(
        {
            "id": [str(i) for i in range(60)],
            "label": [i % 2 for i in range(60)],
            "text": ["x"] * 60,
            "words": [100] * 60,
            "p_classical": np.linspace(0.1, 0.9, 60),
            "p_ppl_gpt2": np.linspace(0.1, 0.9, 60),
            "p_transformer": np.linspace(0.1, 0.9, 60),
        }
    )
    rows, labels = _rows_for_mode(frame, "standard")
    assert len(rows) == 60 and len(labels) == 60
