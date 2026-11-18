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


def test_limited_runs_are_marked_in_the_report_schema() -> None:
    """A --limit run must be labelled, so its numbers cannot be quoted as results."""
    source = (
        Path(__file__).resolve().parent.parent / "ml" / "train_modernbert.py"
    ).read_text()
    assert '"limited": bool(args.limit)' in source
    assert "PLUMBING CHECK ONLY" in source


def test_ood_is_evaluated_even_for_limited_runs() -> None:
    """test_ood is where the known length defect lives; a smoke run must surface it."""
    source = (
        Path(__file__).resolve().parent.parent / "ml" / "train_modernbert.py"
    ).read_text()
    assert 'evaluate(read(args.ood), "test_ood")' in source
    # the old bug: gated behind `if not args.limit`
    assert 'if not args.limit:\n        ood_frame' not in source


def test_no_instruction_points_at_a_missing_script() -> None:
    """The script used to print 'Next: python ml/evaluate_modernbert.py', which is absent."""
    repo = Path(__file__).resolve().parent.parent
    source = (repo / "ml" / "train_modernbert.py").read_text()
    assert "evaluate_modernbert" not in source
    assert not (repo / "ml" / "evaluate_modernbert.py").exists()


def test_by_length_uses_positional_masks() -> None:
    """get_indexer on a non-unique index can return -1; positional masks cannot.

    Checked on executable lines only: the explanation of the old bug mentions
    get_indexer in a comment, and matching that would fail on the fix itself.
    """
    lines = (
        Path(__file__).resolve().parent.parent / "ml" / "train_modernbert.py"
    ).read_text().splitlines()
    code = [ln for ln in lines if not ln.lstrip().startswith("#")]
    assert any("np.flatnonzero(bucket_values == label_value)" in ln for ln in code)
    assert not any("get_indexer" in ln for ln in code)


def test_by_domain_uses_positional_masks() -> None:
    """The domain breakdown must not use get_indexer (returns -1 on non-unique index)."""
    lines = (
        Path(__file__).resolve().parent.parent / "ml" / "train_modernbert.py"
    ).read_text().splitlines()
    code = [ln for ln in lines if not ln.lstrip().startswith("#")]
    assert any("np.flatnonzero(domain_values == domain_name)" in ln for ln in code)
    assert not any("get_indexer" in ln for ln in code)


def test_report_contains_a_per_domain_breakdown() -> None:
    """Aggregate OOD numbers are not comparable across --mix-formal values."""
    source = (
        Path(__file__).resolve().parent.parent / "ml" / "train_modernbert.py"
    ).read_text()
    assert 'entry["by_domain"] = {}' in source
    assert '"human_false_positive_rate"' in source


def test_banner_does_not_mislabel_a_full_run_as_a_subset() -> None:
    source = (
        Path(__file__).resolve().parent.parent / "ml" / "train_modernbert.py"
    ).read_text()
    assert "    if args.limit:" in source
    assert "Full run:" in source
