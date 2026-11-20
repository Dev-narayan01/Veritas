"""Train the token-statistics component (``ppl_gpt2``).

On a GPU run with the Hugging Face Hub reachable, this trains the model on GPT-2
small's per-token statistics.  In a CPU-only or offline environment it trains the
same downstream classifier on summary statistics from a trigram language model fitted
to the training split, and records ``backend="ngram_lm"`` in the artifact so every
health response and result tells the truth about which backend produced the numbers.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from common import ARTIFACT_DIR, RunConfig, log, prepare_dirs, seed_everything, timer, write_json
from featurestore import SPLITS, cap_rows, load_split, reset, write_rows

sys.path.insert(0, str(ARTIFACT_DIR.parents[1] / "backend"))
from app.detector.features import compute_style  # noqa: E402
from app.detector.perplexity import (  # noqa: E402
    Gpt2Backend,
    NgramLM,
    PerplexityArtifact,
    ngram_features,
    save,
)

FEATURE_NAMES = [
    "sentence_len_mean", "sentence_len_std", "word_len_mean", "type_token_ratio",
    "punctuation_rate", "stopword_ratio", "repetition_rate", "contraction_rate",
    "long_word_ratio", "sentence_count",
    "lm_mean_logprob", "lm_std_logprob", "burstiness", "mean_rank_pct",
    "top10_hit_rate", "entropy_mean",
]


def _prefer_gpt2() -> bool:
    """Use GPT-2 when torch is installed *and* its weights are available."""
    try:  # pragma: no cover - GPU/Kaggle path
        import torch  # noqa: F401

        backend = Gpt2Backend()
        return bool(backend.load())
    except Exception as exc:  # noqa: BLE001 - optional dependency
        log.info("GPT-2 backend not usable here (%s); using the n-gram backend", exc)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    prepare_dirs()
    seed_everything()

    target = ARTIFACT_DIR / "ppl_gpt2_model.pkl"
    if target.exists() and not args.force:
        log.info("ppl_gpt2_model.pkl already exists; use --force to retrain")
        return 0

    with timer("train_perplexity_model"):
        backend = "gpt2" if _prefer_gpt2() else "ngram_lm"
        train_frame = cap_rows(load_split("train"), cfg.ppl_cap)
        val_frame = load_split("val_stack")

        lm = None
        if backend == "ngram_lm":
            lm = NgramLM.train(train_frame["text"].tolist(), order=3)
            log.info("trigram LM: %s contexts, vocab %s", len(lm.counts), lm.vocab_size)

        # One backend for the whole run. The previous version constructed Gpt2Backend()
        # inside features(), so load() was never called and every document hit
        # `assert self._tokenizer is not None` -- the run died on the first text after
        # 146 seconds of downloading weights.
        gpt2 = None
        if backend == "gpt2":  # pragma: no cover - GPU path
            gpt2 = Gpt2Backend()
            if not gpt2.load():
                raise SystemExit(f"GPT-2 backend reported available but failed to load: {gpt2.error}")

        def features(text: str) -> dict[str, float]:
            row = compute_style(text).as_dict()
            if backend == "ngram_lm":
                assert lm is not None
                row.update(ngram_features(text, lm))
            else:  # pragma: no cover - Kaggle path
                assert gpt2 is not None
                row.update(gpt2.features(text))
            return row

        train_rows = [features(t) for t in train_frame["text"].tolist()]
        matrix = np.asarray([[r.get(n, 0.0) for n in FEATURE_NAMES] for r in train_rows], dtype=np.float64)
        labels = train_frame["label"].astype(int).to_numpy()

        from sklearn.ensemble import GradientBoostingClassifier

        classifier = GradientBoostingClassifier(
            n_estimators=60 if cfg.smoke else 200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            random_state=42,
        )
        classifier.fit(matrix, labels)

        # Style statistics become the reference for the explanation card's z-scores.
        style_names = list(compute_style("a short reference sentence .").as_dict())
        style_reference = {
            name: {
                "mean": float(np.mean([row.get(name, 0.0) for row in train_rows])),
                "std": float(np.std([row.get(name, 0.0) for row in train_rows]) or 1.0),
            }
            for name in style_names
        }

        artifact = PerplexityArtifact(
            backend=backend,  # type: ignore[arg-type]
            classifier=classifier,
            feature_names=FEATURE_NAMES,
            lm=lm,
            style_reference=style_reference,
            model_name="gpt2" if backend == "gpt2" else "ngram_lm",
        )
        save(artifact, target)
        write_json(
            ARTIFACT_DIR / "ppl_meta.json",
            {"backend": backend, "rows": len(train_frame), "features": FEATURE_NAMES,
             "validation_rows": len(val_frame), "run": cfg.label},
        )
        log.info("ppl artifact saved (backend=%s)", backend)

        for split in SPLITS:
            reset(split, "ppl_gpt2")
            frame = load_split(split)
            if frame.empty:
                continue
            probs = artifact.predict_proba(frame["text"].tolist())
            write_rows(
                split,
                "ppl_gpt2",
                [{"id": i, "p_ppl_gpt2": float(p)} for i, p in zip(frame["id"].tolist(), probs)],
            )
            log.info("scored %s rows for split %s", len(frame), split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
