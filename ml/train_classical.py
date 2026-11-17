"""Train the classical TF-IDF + logistic-regression detector and score every split.

Two vectoriser variants (character 3-5 grams, word 1-2 grams) are fitted on the
training split only; the one with the better *validation* ROC-AUC is kept and saved as
``classical.pkl``.  The held-out test and OOD splits are never used to choose it.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from common import ARTIFACT_DIR, RunConfig, log, prepare_dirs, seed_everything, timer, write_json
from featurestore import SPLITS, cap_rows, load_split, reset, write_rows

sys.path.insert(0, str(ARTIFACT_DIR.parents[1] / "backend"))
from app.detector.classical import pick_better, save, train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    prepare_dirs()
    seed_everything()

    target = ARTIFACT_DIR / "classical.pkl"
    if target.exists() and not args.force:
        log.info("classical.pkl already exists; use --force to retrain")
        return 0

    with timer("train_classical"):
        train_frame = cap_rows(load_split("train"), cfg.classical_cap)
        val_frame = load_split("val_stack")
        texts = train_frame["text"].tolist()
        labels = train_frame["label"].astype(int).tolist()
        val_texts = val_frame["text"].tolist()
        val_labels = val_frame["label"].astype(int).tolist()

        models = {}
        for variant in ("char", "word"):
            try:
                models[variant] = train(texts, labels, variant)
            except Exception as exc:  # noqa: BLE001 - a variant may fail on tiny data
                log.warning("variant %s failed: %s", variant, exc)
        if not models:
            raise RuntimeError("no classical variant could be trained")

        best, scores = pick_better(models, val_texts, val_labels)
        log.info("validation AUC per variant: %s -> keeping %s", scores, best)
        model = models[best]
        save(model, target)
        write_json(
            ARTIFACT_DIR / "classical_meta.json",
            {"variant": best, "validation_auc": scores, "rows": len(texts), "run": cfg.label},
        )

        for split in SPLITS:
            reset(split, "classical")
            frame = load_split(split)
            if frame.empty:
                continue
            probs = model.predict_proba(frame["text"].tolist())
            write_rows(
                split,
                "classical",
                [
                    {"id": row_id, "p_classical": float(p)}
                    for row_id, p in zip(frame["id"].tolist(), probs)
                ],
            )
            log.info("scored %s rows for split %s (mean p=%.3f)", len(frame), split, float(np.mean(probs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
