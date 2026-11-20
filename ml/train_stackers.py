"""Fit one stacker per operating mode on ``val_stack``.

For every mode, the feature row contains the logits of the base components the mode
declares, the standardised zero-shot scores, and ``log(word_count)``.  A mode is only
fitted when all of its components produced scores; otherwise the script records why
and leaves that mode without a stacker, which makes the API report it unavailable.
This is what keeps ``fast`` genuinely different from ``standard`` and ``deep``.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from common import ARTIFACT_DIR, RunConfig, log, prepare_dirs, seed_everything, timer, write_json
from featurestore import merge

sys.path.insert(0, str(ARTIFACT_DIR.parents[1] / "backend"))
from app.config import MODE_COMPONENTS, MODES  # noqa: E402
from app.detector.stacker import feature_row, fit_stacker, save  # noqa: E402

# The base components this script expects to find in the feature store.
SCORED_COMPONENTS = ("classical", "ppl_gpt2", "transformer")


def _mode_components(mode: str) -> tuple[tuple[str, ...], list[str]]:
    """The scored components this mode needs, and the ones that are not scored here.

    ``binoculars`` and ``fastdetect`` are zero-shot scorers with their own feature store
    and their own availability check; they are not part of ``probs``. Treating them as
    if they were meant ``needed`` came out empty for ``deep``, which then silently
    looked like "all components present" and fitted a stacker without them.
    """
    scored = tuple(c for c in MODE_COMPONENTS[mode] if c in SCORED_COMPONENTS)
    zero_shot = [c for c in MODE_COMPONENTS[mode] if c not in SCORED_COMPONENTS]
    return scored, zero_shot


def _rows_for_mode(frame: pd.DataFrame, mode: str) -> tuple[list[dict], list[int]]:
    rows: list[dict] = []
    labels: list[int] = []
    needed_components, _ = _mode_components(mode)
    for record in frame.to_dict("records"):
        probs = {
            "classical": record.get("p_classical"),
            "ppl_gpt2": record.get("p_ppl_gpt2"),
            "transformer": record.get("p_transformer"),
        }
        zero_shot = {
            "binoculars": record.get("z_binoculars") or 0.0,
            "fastdetect": record.get("z_fastdetect") or 0.0,
        }
        if any(
            c in {"binoculars", "fastdetect"} and record.get(f"z_{c}") is None
            for c in MODE_COMPONENTS[mode]
        ):
            return [], []
        if any(probs[c] is None or pd.isna(probs[c]) for c in needed_components):
            return [], []
        rows.append(feature_row(probs, zero_shot, int(record.get("words", 100)), mode))
        labels.append(int(record["label"]))
    return rows, labels


def _missing_columns(frame: pd.DataFrame, mode: str) -> list[str]:
    """Which of the mode's components produced no column at all.

    A bare "0 usable rows" sends the reader to the data; the actual cause is usually
    that one component never scored this split, which is a different command to run.
    """
    needed, _ = _mode_components(mode)
    missing = []
    for component in needed:
        column = f"p_{component}"
        if column not in frame.columns or frame[column].isna().all():
            missing.append(component)
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    prepare_dirs()
    seed_everything()

    with timer("train_stackers"):
        stack_frame = merge("val_stack")
        log.info("val_stack rows available for stacking: %s", len(stack_frame))
        fitted: dict[str, dict] = {}
        for mode in MODES:
            rows, labels = _rows_for_mode(stack_frame, mode)
            if len(rows) < 40 or len(set(labels)) < 2:
                missing = _missing_columns(stack_frame, mode)
                fitted[mode] = {
                    "fitted": False,
                    "reason": f"only {len(rows)} usable rows for mode {mode}",
                    "missing": missing,
                }
                log.warning(
                    "mode %s not fitted: %s%s", mode, fitted[mode]["reason"],
                    f" (no scores for: {', '.join(missing)})" if missing else "",
                )
                continue
            artifact = fit_stacker(mode, rows, labels)
            save(artifact, ARTIFACT_DIR / f"stacker_{mode}.pkl")
            fitted[mode] = {"fitted": True, "rows": len(rows), "features": artifact.feature_names}
            log.info("stacker_%s.pkl written (%s rows)", mode, len(rows))
        write_json(
            ARTIFACT_DIR / "stacker_meta.json",
            {"modes": fitted, "run": cfg.label, "components": {m: list(MODE_COMPONENTS[m]) for m in MODES}},
        )
        if not any(entry.get("fitted") for entry in fitted.values()):
            raise RuntimeError("no mode could be fitted; check that base components scored val_stack")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
