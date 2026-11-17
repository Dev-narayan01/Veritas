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


def _rows_for_mode(frame: pd.DataFrame, mode: str) -> tuple[list[dict], list[int]]:
    rows: list[dict] = []
    labels: list[int] = []
    for record in frame.to_dict("records"):
        probs = {
            "classical": record.get("p_classical"),
            "ppl_gpt2": record.get("p_ppl_gpt2"),
            "transformer": record.get("p_transformer"),
        }
        needed = [c for c in MODE_COMPONENTS[mode] if c in probs]
        if any(probs[c] is None or pd.isna(probs[c]) for c in needed):
            return [], []
        zero_shot = {
            "binoculars": record.get("z_binoculars") or 0.0,
            "fastdetect": record.get("z_fastdetect") or 0.0,
        }
        if any(
            c in {"binoculars", "fastdetect"} and record.get(f"z_{c}") is None
            for c in MODE_COMPONENTS[mode]
        ):
            return [], []
        rows.append(feature_row(probs, zero_shot, int(record.get("words", 100)), mode))
        labels.append(int(record["label"]))
    return rows, labels


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
                fitted[mode] = {
                    "fitted": False,
                    "reason": f"only {len(rows)} usable rows for mode {mode}",
                }
                log.warning("mode %s not fitted: %s", mode, fitted[mode]["reason"])
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
