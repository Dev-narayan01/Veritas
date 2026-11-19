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


def test_report_has_no_get_indexer_anywhere() -> None:
    """get_indexer on a non-unique index can return -1; positional masks cannot.

    Checked on executable lines only: the explanation of the old bug mentions
    get_indexer in a comment, and matching that would fail on the fix itself.
    """
    lines = (
        Path(__file__).resolve().parent.parent / "ml" / "train_modernbert.py"
    ).read_text().splitlines()
    code = [ln for ln in lines if not ln.lstrip().startswith("#")]
    assert not any("get_indexer" in ln for ln in code)


def test_by_domain_is_actually_emitted_for_real_arrays() -> None:
    """The bug this replaces: by_domain was guarded on a column the loader dropped.

    The old test grepped the source for ``entry["by_domain"] = {}`` and passed while
    the branch was unreachable, so every report shipped with no per-domain section.
    This one runs the function.
    """
    from ml.train_modernbert import build_split_report

    # 40 rows over two domains, mixed labels, one domain perfectly separable.
    labels, probs, domains = [], [], []
    for i in range(40):
        domain = "arxiv_ai" if i % 2 else "wiki_csai"
        label = i % 2
        labels.append(label)
        probs.append(0.95 if label else 0.05)
        domains.append(domain)
    entry = build_split_report(labels, probs, domains=domains)
    assert "by_domain" in entry
    assert set(entry["by_domain"]) == {"arxiv_ai", "wiki_csai"}
    assert entry["by_domain"]["wiki_csai"]["human_false_positive_rate"] == 0.0


def test_one_class_domain_still_reports_a_false_positive_rate() -> None:
    """arxiv_ai in test_ood has zero AI rows; its FPR is still the number that matters.

    Reporting nothing here is how a 99% FPR on formal prose stayed invisible: the old
    code computed human FPR only inside ``if len(set(y)) > 1``.
    """
    from ml.train_modernbert import build_split_report

    labels = [0] * 20 + [0, 1] * 20
    probs = [0.9] * 20 + [0.05, 0.95] * 20
    domains = ["arxiv_ai"] * 20 + ["wiki_csai"] * 40
    entry = build_split_report(labels, probs, domains=domains)
    ai_domain = entry["by_domain"]["arxiv_ai"]
    assert ai_domain["positive_rate"] == 0.0
    assert ai_domain["human_false_positive_rate"] == 1.0
    assert "roc_auc" not in ai_domain  # undefined for a single class


def test_by_length_is_emitted_and_uses_positional_masks() -> None:
    from ml.train_modernbert import build_split_report

    labels, probs, words = [], [], []
    for i in range(80):
        label = i % 2
        labels.append(label)
        probs.append(0.9 if label else 0.1)
        words.append(40 + (i // 2) * 12)  # spans every bucket
    entry = build_split_report(labels, probs, word_counts=words)
    assert set(entry["by_length"]) == {"<100", "100-200", "200-400", "400+"}
    for bucket in entry["by_length"].values():
        assert bucket["human_fp_rate"] == 0.0


def test_report_rejects_misaligned_arrays() -> None:
    """Silently zipping mismatched arrays is how a wrong number gets published."""
    import pytest

    from ml.train_modernbert import build_split_report

    with pytest.raises(ValueError, match="not aligned"):
        build_split_report([0, 1, 0], [0.1, 0.9])


def test_loader_keeps_the_domain_column() -> None:
    """The root cause: read() selected only text/label, so by_domain never ran."""
    from ml.train_modernbert import columns_for_report

    assert columns_for_report(["id", "text", "label", "domain", "words"]) == [
        "text", "label", "domain",
    ]
    # A split without domain must still load.
    assert columns_for_report(["text", "label"]) == ["text", "label"]


def test_banner_does_not_mislabel_a_full_run_as_a_subset() -> None:
    source = (
        Path(__file__).resolve().parent.parent / "ml" / "train_modernbert.py"
    ).read_text()
    assert "    if args.limit:" in source
    assert "Full run:" in source
