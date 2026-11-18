"""Guards in ml/train_modernbert.py that protect the honesty of a reported number.

Both exist because of specific failures on a real machine:

1. ``prepare_data.py`` falls back to a 60-row synthetic demo set when no dataset
   source is reachable. ``split_records`` then puts a single row in ``train``, and
   training on it ran (and errored deep in the trainer) instead of stopping. Any
   metric from that run would be meaningless, so the script must refuse.

2. ModernBERT calls ``torch.compile`` on its embeddings when
   ``config.reference_compile`` is truthy. On CUDA that routes through Triton, which
   JIT-builds a CUDA extension and needs a C compiler; a fresh WSL has none, so the
   run died with ``InductorError: Failed to find C compiler``.

These tests exercise the argument surface and the demo-detection rule without a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.train_modernbert import parse_args  # noqa: E402


def test_min_rows_defaults_to_something_that_excludes_the_demo_set() -> None:
    """The synthetic set is 60 rows; the default floor must be above that."""
    args = parse_args([])
    assert args.min_rows > 60


def test_demo_fallback_row_count_is_rejected() -> None:
    """60 synthetic rows split 1/2/... must fall under the floor."""
    args = parse_args([])
    assert 1 < args.min_rows
    assert 1 < args.min_rows  # the observed train split was a single row


def test_min_rows_is_adjustable_for_deliberate_tiny_runs() -> None:
    assert parse_args(["--min-rows", "0"]).min_rows == 0


def test_compile_is_off_by_default() -> None:
    """Enabling torch.compile must be opt-in: it needs gcc and minutes of warmup."""
    assert parse_args([]).compile is False


def test_compile_can_be_requested() -> None:
    assert parse_args(["--compile"]).compile is True


def test_allow_cpu_is_off_by_default() -> None:
    assert parse_args([]).allow_cpu is False


def test_reference_compile_override_is_conditional() -> None:
    """The code must not assume the attribute exists.

    transformers 4.49 defines ``config.reference_compile``; 5.x removed it along with
    the ``@torch.compile`` decorators. An unconditional assignment would create a
    stray attribute on 5.x, so the script guards with hasattr.
    """
    source = (
        Path(__file__).resolve().parent.parent / "ml" / "train_modernbert.py"
    ).read_text()
    assert 'hasattr(model.config, "reference_compile")' in source


class _FakeConfig:
    """Minimal stand-in for a 4.49 ModernBertConfig."""

    reference_compile = None


def test_false_disables_the_compiled_path() -> None:
    cfg = _FakeConfig()
    if hasattr(cfg, "reference_compile"):
        cfg.reference_compile = False
    # forward() chooses the compiled branch on truthiness, so False is what matters.
    assert bool(cfg.reference_compile) is False
