"""Tests for ml/adopt_modernbert.py -- the bridge from a checkpoint to the served model.

The script exists because three things failed silently in sequence: the API loads
``<artifacts>/transformer`` and a ModernBERT run writes ``<artifacts>/<run>/``, the
served window defaulted to 256 tokens against a model trained at 1024, and adopting a
checkpoint flips ``standard`` from unavailable to available, which changes the mode the
API picks on its own. Each guard below corresponds to one of those.

None of these need torch: the script only reads JSON, a config and a weights filename.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.adopt_modernbert import (  # noqa: E402
    install,
    staleness,
    update_manifest,
    verify,
)


def _run(tmp_path: Path, **overrides) -> Path:
    """A minimal but realistic train_modernbert.py output directory."""
    run = tmp_path / "modernbert_1ep"
    run.mkdir()
    (run / "model.safetensors").write_bytes(b"weights")
    (run / "tokenizer.json").write_text("{}")
    (run / "config.json").write_text(json.dumps({"id2label": {"0": "human", "1": "ai"}}))
    report = {
        "model": "answerdotai/ModernBERT-base",
        "device": "cuda",
        "max_len": 1024,
        "epochs": 1.0,
        "train_rows": 23342,
        "limited": False,
        "note": "full training run",
        "splits": {
            "test_ood": {
                "n": 5282,
                "roc_auc": 0.91,
                "human_false_positive_rate": 0.12,
                "by_domain": {
                    "wiki_csai": {"n": 1496, "human_false_positive_rate": 0.4471},
                },
            }
        },
    }
    report.update(overrides)
    (run / "eval_report.json").write_text(json.dumps(report))
    return run


def test_a_limited_checkpoint_is_refused(tmp_path: Path) -> None:
    """A --limit run says so in its own report; adopting it would ship plumbing."""
    run = _run(tmp_path, limited=True, train_rows=200,
               note="PLUMBING CHECK ONLY: trained on 200 rows via --limit. Not a result.")
    with pytest.raises(SystemExit) as exc:
        verify(run)
    message = str(exc.value)
    assert "PLUMBING CHECK ONLY" in message
    assert "without --limit" in message


def test_a_checkpoint_without_a_report_is_refused(tmp_path: Path) -> None:
    """No eval_report.json means no number was ever recorded to judge it by."""
    run = _run(tmp_path)
    (run / "eval_report.json").unlink()
    with pytest.raises(SystemExit, match="no eval_report.json"):
        verify(run)


def test_an_unresolvable_ai_label_is_refused(tmp_path: Path) -> None:
    """The runtime resolves the AI column by name; an unnameable one must not ship."""
    run = _run(tmp_path)
    (run / "config.json").write_text(
        json.dumps({"id2label": {"0": "clean", "1": "synthetic"}})
    )
    with pytest.raises(SystemExit, match="no recognisable AI class"):
        verify(run)


def test_install_copies_weights_and_records_the_source_run(tmp_path: Path) -> None:
    run = _run(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    install(run, artifacts / "transformer", 1024, dry_run=False)

    assert (artifacts / "transformer" / "model.safetensors").exists()
    assert (artifacts / "transformer" / "tokenizer.json").exists()
    meta = json.loads((artifacts / "transformer_meta.json").read_text())
    # The availability check keys off `trained`; without it `standard` stays off.
    assert meta["trained"] is True
    assert meta["rows"] == 23342
    assert meta["trained_max_len"] == 1024
    assert meta["source_run"].endswith("modernbert_1ep")


def test_install_keeps_the_previous_weights_instead_of_destroying_them(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    target = tmp_path / "artifacts" / "transformer"
    target.mkdir(parents=True)
    (target / "model.safetensors").write_bytes(b"old-roberta-weights")

    install(run, target, 1024, dry_run=False)
    assert (tmp_path / "artifacts" / "transformer.previous" / "model.safetensors").exists()


def test_dry_run_touches_nothing(tmp_path: Path) -> None:
    run = _run(tmp_path)
    artifacts = tmp_path / "artifacts"
    plan = install(run, artifacts / "transformer", 1024, dry_run=True)
    assert plan["train_rows"] == 23342
    assert not (artifacts / "transformer").exists()


def test_manifest_records_the_served_window(tmp_path: Path) -> None:
    """TRANSFORMER_MAX_LEN is the number that silently truncates; record what shipped."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "manifest.json").write_text(
        json.dumps({"product": "Veritas", "components_trained": ["classical"]})
    )
    assert update_manifest(artifacts, 1024) is True
    manifest = json.loads((artifacts / "manifest.json").read_text())
    assert manifest["transformer_max_len"] == 1024
    assert "transformer" in manifest["components_trained"]


def test_manifest_update_is_a_no_op_when_there_is_no_bundle(tmp_path: Path) -> None:
    assert update_manifest(tmp_path, 1024) is False


def test_a_missing_standard_stacker_is_reported_as_a_serving_change(
    tmp_path: Path,
) -> None:
    """Adopting a checkpoint flips standard on, and standard is the default mode."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "stacker_meta.json").write_text(
        json.dumps(
            {
                "modes": {
                    "lite": {"fitted": True, "features": ["log_word_count", "logit_classical"]},
                    "fast": {"fitted": True, "features": ["log_word_count"]},
                    "standard": {"fitted": False, "reason": "only 0 usable rows"},
                }
            }
        )
    )
    notes = "\n".join(staleness(artifacts))
    assert "standard/deep have no stacker" in notes
    assert "default_mode is standard" in notes
    assert "lite/fast stackers were fitted before this corpus existed" in notes


def test_a_complete_bundle_reports_nothing_stale(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "stacker_meta.json").write_text(
        json.dumps({"modes": {"standard": {"fitted": True, "features": []}}})
    )
    assert staleness(artifacts) == []
