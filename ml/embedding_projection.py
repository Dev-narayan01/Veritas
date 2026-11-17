"""3D projection of test-set features for the Evaluation page.

Up to 3,000 test texts are represented by a compact feature vector (TF-IDF reduced by
SVD plus the component scores), projected to three dimensions with UMAP when it is
installed and with PCA otherwise, and saved with true labels and per-point correctness
so the UI can ring misclassified points.  The caption in the UI states plainly that
this is a projection of model features for visual intuition, not evidence.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from common import ARTIFACT_DIR, REPO_ROOT, RunConfig, log, seed_everything, timer, write_json
from featurestore import merge

sys.path.insert(0, str(REPO_ROOT / "backend"))

MAX_POINTS = 3000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    seed_everything()

    with timer("embedding_projection"):
        frame = merge("test")
        if frame.empty:
            log.warning("no test rows; projection skipped")
            return 0
        subset = frame.head(MAX_POINTS)
        try:
            from sklearn.decomposition import PCA, TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import StandardScaler

            vectorizer = TfidfVectorizer(max_features=4000, min_df=2, sublinear_tf=True)
            matrix = vectorizer.fit_transform(subset["text"].tolist())
            reduced = TruncatedSVD(n_components=min(40, matrix.shape[1] - 1), random_state=42).fit_transform(matrix)
            scores = np.column_stack(
                [
                    subset.get("p_classical", np.zeros(len(subset))).fillna(0.5).to_numpy(),
                    subset.get("p_ppl_gpt2", np.zeros(len(subset))).fillna(0.5).to_numpy(),
                    subset.get("p_transformer", np.zeros(len(subset))).fillna(0.5).to_numpy(),
                ]
            )
            features = StandardScaler().fit_transform(np.hstack([reduced, scores]))
            method = "pca"
            if cfg.full or not cfg.smoke:
                try:  # pragma: no cover - umap is optional
                    import umap

                    coords = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(features)
                    method = "umap"
                except Exception as exc:  # noqa: BLE001 - optional dependency
                    log.info("UMAP unavailable (%s); using PCA", exc)
                    coords = PCA(n_components=3, random_state=42).fit_transform(features)
            else:
                coords = PCA(n_components=3, random_state=42).fit_transform(features)
        except Exception as exc:  # noqa: BLE001 - projection is a nice-to-have
            log.warning("projection failed: %s", exc)
            return 0

        predicted = (
            subset.get("p_classical", np.zeros(len(subset))).fillna(0.5).to_numpy() >= 0.5
        ).astype(int)
        payload = {
            "method": method,
            "caption": "projection of model features; for visual intuition only",
            "labels": subset["label"].astype(int).tolist(),
            "misclassified": (predicted != subset["label"].astype(int).to_numpy()).astype(int).tolist(),
            "points": [[round(float(v), 4) for v in row] for row in coords],
            "scores": [round(float(v), 4) for v in
                       subset.get("p_classical", np.zeros(len(subset))).fillna(0.5).tolist()],
        }
        write_json(ARTIFACT_DIR / "embedding_projection.json", payload)
        log.info("projection written for %s points using %s", len(coords), method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
