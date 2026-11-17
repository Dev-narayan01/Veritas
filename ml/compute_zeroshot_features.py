"""Compute the opt-in zero-shot features (Binoculars and Fast-DetectGPT).

These need ``torch`` and the 0.5B models, so the script checks availability first and
exits with a clear log line when they cannot run.  Nothing is fabricated: if this
script does not produce ``z_binoculars`` / ``z_fastdetect`` columns, the stackers are
trained without them and the API reports the ``deep`` mode as unavailable.

Zero-shot features are computed on ``val_stack``, ``val_cal``, ``test`` and
``test_ood`` only (they need no training), capped by the run size setting.
"""

from __future__ import annotations

import argparse
import sys

from common import ARTIFACT_DIR, RunConfig, log, prepare_dirs, seed_everything, timer, write_json
from featurestore import load_split, reset_zero_shot, write_zero_shot

sys.path.insert(0, str(ARTIFACT_DIR.parents[1] / "backend"))
from app.detector.binoculars import Binoculars  # noqa: E402
from app.detector.fastdetect import FastDetect  # noqa: E402

SCORED_SPLITS = ("val_stack", "val_cal", "test", "test_ood")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    prepare_dirs()
    seed_everything()

    meta_path = ARTIFACT_DIR / "zeroshot_meta.json"
    if meta_path.exists() and not args.force:
        log.info("zero-shot features already computed; use --force to redo them")
        return 0

    with timer("compute_zeroshot_features"):
        binoculars = Binoculars()
        fastdetect = FastDetect()
        b_ok = binoculars.load()
        f_ok = fastdetect.load()
        write_json(
            meta_path,
            {
                "binoculars": {"available": b_ok, "error": binoculars.error,
                               "observer": binoculars.observer_name,
                               "performer": binoculars.performer_name},
                "fastdetect": {"available": f_ok, "error": fastdetect.error,
                               "model": fastdetect.model_name},
                "run": cfg.label,
            },
        )
        if not (b_ok or f_ok):
            log.warning(
                "neither Binoculars nor Fast-DetectGPT can run here; the deep mode will be "
                "reported unavailable rather than silently scored without them"
            )
            return 0

        for split in SCORED_SPLITS:
            reset_zero_shot(split)
            frame = load_split(split)
            if frame.empty:
                continue
            subset = frame.head(cfg.zeroshot_cap) if len(frame) > cfg.zeroshot_cap else frame
            rows: list[dict] = []
            for index, text in enumerate(subset["text"].tolist()):
                row: dict[str, object] = {"id": subset["id"].tolist()[index]}
                if b_ok:
                    try:  # pragma: no cover - needs torch + weights
                        row["z_binoculars"] = binoculars.score(text).score
                    except Exception as exc:  # noqa: BLE001
                        log.debug("binoculars failed on one row: %s", exc)
                if f_ok:
                    try:  # pragma: no cover
                        row["z_fastdetect"] = fastdetect.score(text).score
                    except Exception as exc:  # noqa: BLE001
                        log.debug("fastdetect failed on one row: %s", exc)
                rows.append(row)
                if index and index % 200 == 0:
                    log.info("%s: %s/%s texts scored", split, index, len(subset))
            write_zero_shot(split, rows)
            log.info("zero-shot features for %s: %s rows", split, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
