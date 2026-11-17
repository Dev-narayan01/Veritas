"""Leave-one-generator-out evaluation.

The classical model and the ``fast`` stacker are retrained with one generator family
removed from training, then evaluated on that family.  The drop against the in-domain
number is the honest measure of how well the detector transfers to a generator it has
never seen.  When the dataset contains only one AI generator (HC3's ChatGPT), this
script says so instead of reporting a meaningless "drop of 0.0".
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from common import ARTIFACT_DIR, REPO_ROOT, RunConfig, log, seed_everything, timer, write_json
from featurestore import merge

sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.detector.classical import ClassicalModel, build_vectorizer  # noqa: E402

MIN_ROWS = 60


def _train_and_score(train_texts: list[str], train_labels: list[int], test_texts: list[str],
                     test_labels: list[int]) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    vectorizer = build_vectorizer("word")
    matrix = vectorizer.fit_transform(train_texts)
    classifier = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    classifier.fit(matrix, train_labels)
    model = ClassicalModel(vectorizer=vectorizer, classifier=classifier, variant="word")
    scores = model.predict_proba(test_texts)
    if len(set(test_labels)) < 2:
        return float("nan")
    return float(roc_auc_score(test_labels, scores))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    seed_everything()

    with timer("eval_cross_generator"):
        output: dict[str, object] = {"run": cfg.label, "generators": {}, "notes": []}
        train = merge("train")
        generators = sorted(
            g for g in train["generator"].unique().tolist() if g not in {"human", ""}
        )
        output["ai_generators_in_training"] = generators
        if len(generators) < 2:
            output["notes"].append(
                "Only one AI generator family is available in this dataset ("
                + (", ".join(generators) or "none")
                + "), so leave-one-generator-out would leave no AI training data. "
                "Run the Kaggle pipeline with MAGE/RAID available to measure cross-generator "
                "transfer."
            )
            write_json(ARTIFACT_DIR / "cross_generator.json", output)
            log.warning("cross-generator evaluation skipped: %s", output["notes"][-1])
            return 0

        test = merge("test")
        baseline = _train_and_score(
            train["text"].tolist(), train["label"].astype(int).tolist(),
            test["text"].tolist(), test["label"].astype(int).tolist(),
        )
        output["in_domain_auc"] = round(baseline, 4) if baseline == baseline else None
        for generator in generators:
            held = train[train["generator"] == generator]
            if len(held) < MIN_ROWS:
                output["notes"].append(f"{generator}: only {len(held)} held-out rows, skipped")
                continue
            remaining = train[train["generator"] != generator]
            auc = _train_and_score(
                remaining["text"].tolist(), remaining["label"].astype(int).tolist(),
                held["text"].tolist(), held["label"].astype(int).tolist(),
            )
            output["generators"][generator] = {
                "held_out_rows": int(len(held)),
                "roc_auc": None if np.isnan(auc) else round(auc, 4),
                "drop_vs_in_domain": None if np.isnan(auc) or baseline != baseline
                else round(baseline - auc, 4),
            }
            log.info("held-out %s: AUC=%s", generator, output["generators"][generator])
        write_json(ARTIFACT_DIR / "cross_generator.json", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
