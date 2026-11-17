"""Third-party baseline: ``Hello-SimpleAI/chatgpt-detector-roberta``.

This model is **not ours** and it is labelled third-party everywhere it appears.  It
was trained on HC3, so:

* its label mapping is inspected from ``model.config.id2label`` and confirmed on a few
  known human and known ChatGPT samples before any number is reported;
* it is reported **only on ``test_ood``**, because in-domain HC3 numbers would be
  flattering by construction and not comparable;
* if the model cannot be downloaded the script records why and exits 0.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from common import ARTIFACT_DIR, REPO_ROOT, RunConfig, log, seed_everything, timer, write_json
from featurestore import load_split

sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.detector.calibration import tpr_at_fpr  # noqa: E402

MODEL_ID = "Hello-SimpleAI/chatgpt-detector-roberta"


def _load_pipeline():  # noqa: ANN202
    try:  # pragma: no cover - needs torch + network
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        return pipeline(
            "text-classification", model=model, tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1, top_k=None,
        ), model.config.id2label
    except Exception as exc:  # noqa: BLE001 - optional third-party model
        log.warning("baseline detector unavailable: %s", exc)
        return None, str(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    _cfg = RunConfig(smoke=args.smoke, full=args.full)
    del _cfg
    seed_everything()

    output: dict[str, object] = {
        "model": MODEL_ID,
        "third_party": True,
        "label": "third-party baseline (trained by Hello-SimpleAI on HC3), OOD only",
        "notes": [],
    }
    with timer("eval_baseline_detector"):
        classifier, id2label = _load_pipeline()
        if classifier is None:
            output["available"] = False
            output["reason"] = str(id2label)
            output["notes"].append(
                "The third-party baseline could not be downloaded in this environment; "
                "no number is reported rather than a guessed one."
            )
            write_json(ARTIFACT_DIR / "baseline.json", output)
            return 0

        output["available"] = True
        output["id2label"] = {str(k): str(v) for k, v in id2label.items()}  # type: ignore[union-attr]
        ai_label = next(
            (str(v) for v in id2label.values() if str(v).lower() in {"chatgpt", "ai", "label_1"}),  # type: ignore[union-attr]
            "ChatGPT",
        )
        output["ai_label"] = ai_label

        # --- label sanity check on known samples -----------------------------
        test = load_split("test")
        human_samples = test[test["label"] == 0]["text"].head(8).tolist()
        ai_samples = test[test["label"] == 1]["text"].head(8).tolist()
        human_probs = _probabilities(classifier, human_samples, ai_label)
        ai_probs = _probabilities(classifier, ai_samples, ai_label)
        output["sanity_check"] = {
            "mean_probability_known_human": round(float(np.mean(human_probs)), 4) if human_probs else None,
            "mean_probability_known_ai": round(float(np.mean(ai_probs)), 4) if ai_probs else None,
        }
        sanity_ok = bool(human_probs and ai_probs and np.mean(ai_probs) > np.mean(human_probs))
        output["label_mapping_confirmed"] = sanity_ok
        if not sanity_ok:
            output["notes"].append(
                "Label mapping could not be confirmed on known samples, so no metrics are reported."
            )
            write_json(ARTIFACT_DIR / "baseline.json", output)
            return 0

        # --- OOD-only metrics -------------------------------------------------
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        ood = load_split("test_ood")
        texts = ood["text"].tolist()
        labels = ood["label"].astype(int).to_numpy()
        probs = np.asarray(_probabilities(classifier, texts, ai_label))
        predicted = (probs >= 0.5).astype(int)
        tpr_1, _ = tpr_at_fpr(probs, labels, 0.01)
        output["test_ood"] = {
            "n": int(len(labels)),
            "accuracy": round(float(accuracy_score(labels, predicted)), 4),
            "f1": round(float(f1_score(labels, predicted, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(labels, probs)), 4),
            "tpr_at_1pct_fpr": round(float(tpr_1), 4),
            "human_fpr_at_0.5": round(float(np.mean(probs[labels == 0] >= 0.5)), 4),
        }
        output["notes"].append(
            "Reported on the held-out (out-of-domain) split only: this model was trained on "
            "HC3, so in-domain numbers would not be comparable."
        )
        write_json(ARTIFACT_DIR / "baseline.json", output)
        log.info("baseline OOD metrics: %s", output["test_ood"])
    return 0


def _probabilities(classifier, texts: list[str], ai_label: str) -> list[float]:  # noqa: ANN001
    out: list[float] = []
    for text in texts:
        try:
            scores = classifier(text[:2000])[0]
            out.append(
                float(next(item["score"] for item in scores if item["label"] == ai_label))
            )
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
            log.debug("baseline scoring failed on a row: %s", exc)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
