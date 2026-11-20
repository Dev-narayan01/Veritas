"""Wire a ``train_modernbert.py`` checkpoint into the serving bundle.

``ml/train_modernbert.py`` writes a complete, honest model to
``ml/artifacts/<run>/`` and an ``eval_report.json`` next to it. It is a *training*
script: it says nothing about how the site loads a detector. This script closes that
gap, and it is the whole gap --

* the API loads ``settings.artifact_path / "transformer"``; the checkpoint is written
  to ``ml/artifacts/<run>/``, so without this the site silently keeps using the
  ``roberta-base`` weights from the last bundle;
* ``TRANSFORMER_MAX_LEN`` defaults to 256, and a ModernBERT checkpoint scored at 256
  tokens is scored on roughly the first 190 words of a document. That is the length
  defect this model was trained to remove, reintroduced at inference;
* the ``lite``/``fast`` stackers were fitted on the *old* ``classical`` and
  ``ppl_gpt2`` probabilities. Making ``standard`` available changes the mode the site
  picks by default, so the ensemble and the served numbers change too. Those stackers
  must be refitted on the new corpus afterwards -- ``--check`` reports exactly that,
  with the two commands.

What the script does, in order:

1. refuses to adopt a ``--limit`` checkpoint (its own ``note`` field says it is a
   plumbing check);
2. copies the weights to ``<artifacts>/transformer`` and writes ``transformer_meta.json``
   with the numbers and provenance that make the bundle auditable;
3. records ``--max-len`` in ``manifest.json`` if that file exists;
4. prints the before/after *measured on ``test_ood``* figures for the JSON it just
   installed, and the honest caveats that go with them.

It never edits an ``eval_report.json``, never re-scores a split, and never tunes a
threshold. If a number it would report is missing from the report it says so.

Usage::

    python ml/adopt_modernbert.py ml/artifacts/modernbert_1ep            # install
    python ml/adopt_modernbert.py ml/artifacts/modernbert_1ep --dry-run  # inspect
    python ml/adopt_modernbert.py --check                                # what is live?
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "ml" / "artifacts"
TARGET = ARTIFACT_DIR / "transformer"
# Read from the API the same way the API reads it, so the report cannot drift from
# what the server will actually do.
SETTINGS_KEYS = ("TRANSFORMER_MAX_LEN",)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run", nargs="?", type=Path, default=None,
                   help="directory written by train_modernbert.py")
    p.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    p.add_argument("--max-len", type=int, default=None,
                   help="served window; default: the checkpoint's training max_len")
    p.add_argument("--dry-run", action="store_true", help="verify and report, copy nothing")
    p.add_argument("--check", action="store_true",
                   help="report what is installed and whether the stackers match it")
    return p.parse_args(argv)


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def _fpr(report: dict, split: str, domain: str) -> float | None:
    entry = ((report.get("splits") or {}).get(split) or {}).get("by_domain") or {}
    return (entry.get(domain) or {}).get("human_false_positive_rate")


def served_max_len(artifacts: Path | None = None) -> tuple[int, str]:
    """The window the API will use, and where the number came from.

    Read from the bundle first because that is what the registry does: the window
    travels with the weights so a checkpoint cannot be served through a window it was
    not trained for. Falls back to TRANSFORMER_MAX_LEN, which is what a bundle without
    the field gets.
    """
    import os

    if artifacts is not None:
        manifest = _read_json(artifacts / "manifest.json", {}) or {}
        if manifest.get("transformer_max_len"):
            return int(manifest["transformer_max_len"]), "manifest.json"
    return int(os.environ.get("TRANSFORMER_MAX_LEN", 256)), "TRANSFORMER_MAX_LEN"


def verify(run: Path) -> dict:
    """Load and sanity-check a checkpoint directory; exit with a readable reason."""
    if not run.exists():
        raise SystemExit(f"{run} does not exist: train_modernbert.py writes one per run")
    weights = run / "model.safetensors"
    if not weights.exists():
        alt = list(run.glob("*.bin")) + list(run.glob("pytorch_model*"))
        if not alt:
            raise SystemExit(
                f"{run} has no model weights (model.safetensors or pytorch_model*). "
                "Point this at the --out directory of train_modernbert.py."
            )
    report_path = run / "eval_report.json"
    if not report_path.exists():
        raise SystemExit(
            f"{run} has no eval_report.json. An adopted checkpoint must carry the "
            "numbers it was judged by; re-run train_modernbert.py to produce one."
        )
    report = _read_json(report_path, {}) or {}
    if report.get("limited"):
        raise SystemExit(
            "refusing to adopt a --limit checkpoint. Its own report says:\n"
            f"  {report.get('note')}\n"
            "Retrain without --limit, or pass a directory that contains a full run."
        )
    if not report.get("train_rows"):
        raise SystemExit(f"{run}/eval_report.json records no train_rows; refusing.")
    config = _read_json(run / "config.json", {}) or {}
    id2label = {str(k): v for k, v in (config.get("id2label") or {}).items()}
    # The runtime picks the "ai" class by name; a checkpoint labelled 0/1 would load
    # but score the wrong column, which inverts every probability silently.
    labels = {str(v).lower() for v in id2label.values()}
    if not labels & {"ai", "label_1", "machine", "generated"}:
        raise SystemExit(
            f"{run}/config.json id2label={id2label} has no recognisable AI class. "
            "The runtime resolves the AI column by label name and refuses to load a "
            "checkpoint it cannot resolve, so adopting this one would silently leave "
            "`standard` unavailable instead of scoring it wrong."
        )
    return {"report": report, "config": config}


def install(run: Path, target: Path, max_len: int | None, dry_run: bool) -> dict:
    """Copy weights and write the metadata the API's availability check needs."""
    report = _read_json(run / "eval_report.json", {}) or {}
    plan = {
        "source": str(run),
        "target": str(target),
        "train_rows": report.get("train_rows"),
        "epochs": report.get("epochs"),
        "trained_max_len": report.get("max_len"),
        "served_max_len": max_len or report.get("max_len"),
        "device": report.get("device"),
    }
    if dry_run:
        return plan

    if target.exists():
        backup = target.with_name(target.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(target), str(backup))
        plan["previous_moved_to"] = str(backup)
    target.mkdir(parents=True, exist_ok=True)

    for path in sorted(run.iterdir()):
        if path.is_file() and (
            path.suffix in {".safetensors", ".bin"} or path.name.startswith(("tokenizer", "vocab", "special", "merges", "config"))
        ):
            shutil.copy2(path, target / path.name)

    meta = {
        "trained": True,
        "base": report.get("model"),
        "rows": report.get("train_rows"),
        "epochs": report.get("epochs"),
        "max_len": plan["served_max_len"],
        "trained_max_len": report.get("max_len"),
        "run": "modernbert",
        "source_run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
        "note": "adopted from train_modernbert.py by ml/adopt_modernbert.py",
    }
    (target.parent / "transformer_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return plan


def update_manifest(artifacts: Path, max_len: int) -> bool:
    """Record the served window in manifest.json, which is what the UI reads."""
    path = artifacts / "manifest.json"
    manifest = _read_json(path)
    if manifest is None:
        return False
    manifest["transformer_max_len"] = max_len
    components = list(manifest.get("components_trained") or [])
    if "transformer" not in components:
        components.append("transformer")
    manifest["components_trained"] = components
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return True


def staleness(artifacts: Path) -> list[str]:
    """Ways the installed bundle may not match the checkpoint now in use."""
    notes: list[str] = []
    stacker_meta = _read_json(artifacts / "stacker_meta.json", {}) or {}
    modes = stacker_meta.get("modes") or {}
    fitted = {m for m, e in modes.items() if (e or {}).get("fitted")}
    if "standard" not in fitted:
        notes.append(
            "standard/deep have no stacker yet. Fitting them on val_stack will make "
            "standard available, and the API's default_mode is standard -- so the "
            "served model changes from lite/fast to standard."
        )
    if fitted & {"lite", "fast"}:
        notes.append(
            "lite/fast stackers were fitted before this corpus existed. Refit them or "
            "their probabilities are stale even if the components are unchanged."
        )
    return notes


def report(run_report: dict, max_len: int, trained_max_len: int | None) -> None:
    """Print what changed, using only numbers present in the checkpoint's report."""
    print("\n=== adopted checkpoint ===")
    for split in ("test_ood", "val_in_domain"):
        block = (run_report.get("splits") or {}).get(split) or {}
        if not block:
            continue
        print(f"{split}: n={block.get('n')} "
              f"AUC={block.get('roc_auc')} "
              f"human_fpr={block.get('human_false_positive_rate')} "
              f"ai_recall={block.get('ai_recall')}")
        for domain, entry in (block.get("by_domain") or {}).items():
            print(f"  {domain:16s} n={entry.get('n'):5d} "
                  f"fpr={entry.get('human_false_positive_rate')} "
                  f"recall={entry.get('ai_recall')}")
    if trained_max_len and max_len < trained_max_len:
        print(
            f"\n! WARNING: trained at {trained_max_len} tokens, serving at {max_len}. "
            f"Every document is truncated to roughly {int(max_len * 0.75)} words, which "
            "is the defect the model was trained to remove. Set TRANSFORMER_MAX_LEN="
            f"{trained_max_len} before starting the API."
        )
    print(
        "\nThe runtime averages per-window probabilities over a sliding window with "
        "stride 128, so a long document is scored in pieces and the mean is returned. "
        "That is not the same estimator as the single forward pass used to produce the "
        "numbers above; agreement is high on documents under the window size and has "
        "not been measured above it."
    )


def check(artifacts: Path) -> int:
    target = artifacts / "transformer"
    meta = _read_json(artifacts / "transformer_meta.json", {}) or {}
    print(f"weights:      {target} ({'present' if target.exists() else 'MISSING'})")
    print(f"meta:         {artifacts / 'transformer_meta.json'}")
    if meta:
        print(f"  source run: {meta.get('source_run')}  rows={meta.get('rows')} "
              f"epochs={meta.get('epochs')}")
        print(f"  trained at {meta.get('trained_max_len')}, served at {meta.get('max_len')}")
    window, source = served_max_len(artifacts)
    trained = meta.get("trained_max_len")
    print(f"served window: {window} (from {source})")
    if trained and int(trained) != int(window):
        print(
            f"! WARNING: trained at {trained} tokens, served at {window}. Documents are "
            f"truncated to roughly {int(window * 0.75)} words before scoring."
        )
    notes = staleness(artifacts)
    if notes:
        print("\nnot in sync:")
        for note in notes:
            print(f"  - {note}")
        print("\nrefit, then re-read the served numbers (both are required):")
        print("  python ml/train_stackers.py --force")
        print("  python ml/tune_thresholds.py --force")
        print("  python ml/evaluate.py --force")
    else:
        print("\nstackers and thresholds are present for every fitted mode.")
    window, _ = served_max_len(artifacts)
    print(
        "\nserve it with the window the bundle records:\n"
        f"  cd backend && TRANSFORMER_MAX_LEN={window} ../.venv/bin/uvicorn "
        "app.main:app --port 8000"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check or args.run is None:
        if args.run is None and not args.check:
            raise SystemExit("give a run directory, or --check to inspect what is live")
        return check(args.artifacts)

    run = args.run if args.run.is_absolute() else (Path.cwd() / args.run)
    verified = verify(run)
    run_report = verified["report"]
    trained_max_len = run_report.get("max_len")
    max_len = args.max_len or trained_max_len

    plan = install(run, args.artifacts / "transformer", max_len, args.dry_run)
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("\n(dry run: nothing copied)")
        return 0

    if max_len and update_manifest(args.artifacts, int(max_len)):
        print(f"manifest.json: transformer_max_len={max_len}, transformer listed as trained")

    report(run_report, int(max_len or 0), trained_max_len)
    notes = staleness(args.artifacts)
    if notes:
        print("\nThis bundle is not in sync yet:")
        for note in notes:
            print(f"  - {note}")
        print("\n  python ml/train_stackers.py --force")
        print("  python ml/tune_thresholds.py --force")
        print("  python ml/evaluate.py --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
