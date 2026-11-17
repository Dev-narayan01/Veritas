"""Calibrate each fitted stacker (Platt) on ``val_cal`` and select its thresholds.

The protocol is the one the spec requires, and it is the reason the previous build's
collapsed calibration cannot happen here:

* the stacker is already fitted (on ``val_stack``), so ``val_cal`` is untouched data;
* Platt scaling is fitted on the stacker's decision values from ``val_cal``;
* the calibration guard aborts the run when those scores have fewer than 100 unique
  values, instead of publishing meaningless thresholds;
* ``t_ai`` and ``t_human`` are selected from ``val_cal`` only, with per-length-bucket
  thresholds only where a bucket has at least 200 rows of each class.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from common import ARTIFACT_DIR, RunConfig, log, prepare_dirs, seed_everything, timer, write_json
from featurestore import merge

sys.path.insert(0, str(ARTIFACT_DIR.parents[1] / "backend"))
from app.config import MODES, settings  # noqa: E402
from app.detector.calibration import (  # noqa: E402
    CalibrationCollapsed,
    PlattScaler,
    assert_score_diversity,
    compute_thresholds,
)
from app.detector.stacker import load as load_stacker, save as save_stacker  # noqa: E402

from train_stackers import _rows_for_mode  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    _cfg = RunConfig(smoke=args.smoke, full=args.full)
    del _cfg
    prepare_dirs()
    seed_everything()

    target = ARTIFACT_DIR / "thresholds.json"
    if target.exists() and not args.force:
        log.info("thresholds.json already exists; use --force to retune")
        return 0

    with timer("tune_thresholds"):
        cal_frame = merge("val_cal")
        log.info("val_cal rows: %s", len(cal_frame))
        thresholds: dict[str, dict] = {}
        for mode in MODES:
            path = ARTIFACT_DIR / f"stacker_{mode}.pkl"
            if not path.exists():
                log.info("mode %s has no stacker; skipping", mode)
                continue
            rows, labels = _rows_for_mode(cal_frame, mode)
            if len(rows) < 20 or len(set(labels)) < 2:
                log.warning("mode %s: not enough calibration rows (%s)", mode, len(rows))
                continue
            stacker = load_stacker(path)
            decisions = np.array([stacker.decision(row) for row in rows], dtype=float)
            platt = PlattScaler.fit(decisions, np.asarray(labels))
            scores = platt.transform(decisions)
            unique = assert_score_diversity(scores, context=f"val_cal/{mode}")
            words = np.array([int(round(np.exp(row["log_word_count"]))) for row in rows])
            mode_thresholds = compute_thresholds(
                scores,
                np.asarray(labels),
                words=words,
                target_fpr=settings.target_human_fpr,
                ai_miss=settings.ai_miss_for_human_verdict,
            )
            stacker.platt = platt
            stacker.thresholds = mode_thresholds
            save_stacker(stacker, path)
            thresholds[mode] = mode_thresholds.to_dict() | {"unique_scores": unique}
            log.info(
                "mode %-8s t_human=%.3f t_ai=%.3f collapsed=%s unique=%s",
                mode, mode_thresholds.t_human, mode_thresholds.t_ai,
                mode_thresholds.band_collapsed, unique,
            )
        if not thresholds:
            raise CalibrationCollapsed("no mode could be calibrated: check the val_cal features")
        # A single thresholds.json per mode is what the API reads.
        write_json(target, thresholds)
        log.info("wrote %s", target)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CalibrationCollapsed as exc:
        log.error("CALIBRATION GUARD: %s", exc)
        raise SystemExit(2) from exc
