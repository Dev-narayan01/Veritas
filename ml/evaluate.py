"""Evaluate every component and every mode, on ``test`` and on ``test_ood``.

Outputs ``ml/artifacts/metrics.json`` with, per split and per scorer:

* accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
* TPR at 1% and 0.1% FPR (with bootstrap CIs for AUROC and TPR@1%),
* human FPR at ``t_ai`` and AI miss rate at ``t_human``, abstention rate,
* expected calibration error,
* breakdowns by domain, generator and length bucket.

The held-out (OOD) block is written first in the JSON and the UI renders it first, so
the optimistic in-domain numbers can never become the headline.  The ablation table
compares the four modes on ``test_ood``.  Plots are written to ``artifacts/plots/``.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from common import (
    ARTIFACT_DIR,
    PLOTS_DIR,
    REPO_ROOT,
    RunConfig,
    bootstrap_ci,
    log,
    read_json,
    seed_everything,
    timer,
    write_json,
)
from featurestore import merge

sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.config import MODE_COMPONENTS, MODES  # noqa: E402
from app.detector.calibration import (  # noqa: E402
    ModeThresholds,
    expected_calibration_error,
    tpr_at_fpr,
)
from app.detector.features import length_bucket  # noqa: E402
from app.detector.stacker import load as load_stacker  # noqa: E402

EVAL_SPLITS = ("test_ood", "test")


def _mode_scores(frame: pd.DataFrame, mode: str, stacker) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Calibrated probabilities and the row ids that actually had every component."""
    from app.detector.stacker import feature_row

    probabilities: list[float] = []
    labels: list[int] = []
    ids: list[str] = []
    for record in frame.to_dict("records"):
        probs = {
            "classical": record.get("p_classical"),
            "ppl_gpt2": record.get("p_ppl_gpt2"),
            "transformer": record.get("p_transformer"),
        }
        needed = [c for c in MODE_COMPONENTS[mode] if c in probs]
        if any(probs[c] is None or pd.isna(probs[c]) for c in needed):
            continue
        zero_shot = {
            "binoculars": record.get("z_binoculars"),
            "fastdetect": record.get("z_fastdetect"),
        }
        if any(
            c in {"binoculars", "fastdetect"} and (zero_shot[c] is None or pd.isna(zero_shot[c]))
            for c in MODE_COMPONENTS[mode]
        ):
            continue
        row = feature_row(
            probs, {k: float(v or 0.0) for k, v in zero_shot.items()},
            int(record.get("words", 100)), mode,
        )
        probabilities.append(float(stacker.probability(row)))
        labels.append(int(record["label"]))
        ids.append(str(record["id"]))
    return np.asarray(probabilities), np.asarray(labels), ids


def _classification_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if scores.size == 0 or len(set(labels.tolist())) < 2:
        return {}
    predicted = (scores >= 0.5).astype(int)
    tpr_1, _ = tpr_at_fpr(scores, labels, 0.01)
    tpr_01, _ = tpr_at_fpr(scores, labels, 0.001)
    auc = float(roc_auc_score(labels, scores))
    pairs = np.column_stack([labels, scores])
    auc_ci = bootstrap_ci(pairs, lambda rows: roc_auc_score(rows[:, 0], rows[:, 1]))
    tpr_ci = bootstrap_ci(pairs, lambda rows: tpr_at_fpr(rows[:, 1], rows[:, 0], 0.01)[0])
    return {
        "n": int(scores.size),
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 4),
        "roc_auc": round(auc, 4),
        "roc_auc_ci": [None if v is None else round(v, 4) for v in auc_ci],
        "pr_auc": round(float(average_precision_score(labels, scores)), 4),
        "tpr_at_1pct_fpr": round(tpr_1, 4),
        "tpr_at_1pct_fpr_ci": [None if v is None else round(v, 4) for v in tpr_ci],
        "tpr_at_0.1pct_fpr": round(tpr_01, 4),
    }


def _threshold_metrics(
    scores: np.ndarray, labels: np.ndarray, thresholds: ModeThresholds,
    words: np.ndarray,
) -> dict[str, float]:
    verdicts = []
    for score, count in zip(scores, words):
        t_human, t_ai = thresholds.for_length(int(count))
        verdicts.append("ai" if score >= t_ai else "human" if score <= t_human else "uncertain")
    verdicts_arr = np.asarray(verdicts)
    humans = labels == 0
    ai = labels == 1
    return {
        "human_fpr_at_t_ai": round(float(np.mean(verdicts_arr[humans] == "ai")), 4)
        if humans.any() else 0.0,
        "ai_miss_at_t_human": round(float(np.mean(verdicts_arr[ai] == "human")), 4)
        if ai.any() else 0.0,
        "abstention_rate": round(float(np.mean(verdicts_arr == "uncertain")), 4),
        "t_human": round(thresholds.t_human, 4),
        "t_ai": round(thresholds.t_ai, 4),
        "band_collapsed": bool(thresholds.band_collapsed),
    }


def _confusion(scores: np.ndarray, labels: np.ndarray, thresholds: ModeThresholds,
               words: np.ndarray) -> dict[str, int]:
    """2x2 confusion matrix of the *reported verdicts*, uncertain included.

    ``positive`` means "the system said likely_ai". Counting abstentions separately is the
    point: a detector that abstains on the hard cases is different from one that guesses.
    """
    verdicts = []
    for score, count in zip(scores, words):
        t_human, t_ai = thresholds.for_length(int(count))
        verdicts.append("ai" if score >= t_ai else "human" if score <= t_human else "uncertain")
    verdicts_arr = np.asarray(verdicts)
    humans, ai = labels == 0, labels == 1
    return {
        "ai_true_positive": int(np.sum(verdicts_arr[ai] == "ai")),
        "ai_false_negative": int(np.sum(verdicts_arr[ai] != "ai")),
        "human_false_positive": int(np.sum(verdicts_arr[humans] == "ai")),
        "human_true_negative": int(np.sum(verdicts_arr[humans] != "ai")),
        "ai_abstained": int(np.sum(verdicts_arr[ai] == "uncertain")),
        "human_abstained": int(np.sum(verdicts_arr[humans] == "uncertain")),
    }


def _accuracy_by_length(scores: np.ndarray, labels: np.ndarray, words: np.ndarray) -> dict[str, dict]:
    """Accuracy at a fixed 0.5 cut-off inside each length bucket, with n."""
    buckets: dict[str, dict] = {}
    for bucket in ("<100", "100-250", "250-500", "500+"):
        mask = np.asarray([length_bucket(int(word)) == bucket for word in words])
        if not mask.any():
            continue
        predicted = (scores[mask] >= 0.5).astype(int)
        buckets[bucket] = {
            "n": int(mask.sum()),
            "accuracy": round(float(np.mean(predicted == labels[mask])), 4),
        }
    return buckets

def _breakdowns(frame: pd.DataFrame, ids: list[str], scores: np.ndarray, labels: np.ndarray) -> dict:
    """Per-domain / generator / length-bucket AUROC and n (AUROC is threshold-free)."""
    from sklearn.metrics import roc_auc_score

    selected = frame.set_index("id").loc[ids].reset_index()
    selected["score"] = scores
    output: dict[str, dict] = {}
    for column in ("domain", "generator"):
        if column not in selected:
            continue
        block: dict[str, dict] = {}
        for value, group in selected.groupby(column):
            entry: dict[str, float | int] = {"n": int(len(group))}
            # No per-domain FPR here: it would need the tuned threshold, and a 0.5 cut-off
            # would report a second, contradictory false-positive rate. The confusion
            # matrix already gives the FPR at the threshold the system actually uses.
            if group["label"].nunique() > 1:
                entry["roc_auc"] = round(float(roc_auc_score(group["label"], group["score"])), 4)
            block[str(value)] = entry
        output[column] = block
    buckets: dict[str, dict] = {}
    selected["bucket"] = [length_bucket(int(w)) for w in selected["words"]]
    for bucket, group in selected.groupby("bucket"):
        entry = {"n": int(len(group))}
        if group["label"].nunique() > 1:
            entry["roc_auc"] = round(float(roc_auc_score(group["label"], group["score"])), 4)
        buckets[bucket] = entry
    output["length_bucket"] = buckets
    return output


def _plots(component_scores: dict[str, dict], mode_scores: dict[str, dict], split: str) -> list[str]:
    """ROC, PR, calibration and score-distribution plots for one split."""
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve
    except Exception as exc:  # noqa: BLE001 - plotting is optional
        log.warning("plots skipped: %s", exc)
        return written

    arrays = dict(component_scores) | dict(mode_scores)
    for kind in ("roc", "pr", "calibration", "distributions"):
        fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=130)
        for name, payload in arrays.items():
            labels, scores = np.asarray(payload["labels"]), np.asarray(payload["scores"])
            if labels.size == 0 or len(set(labels.tolist())) < 2:
                continue
            if kind == "roc":
                fpr, tpr, _ = roc_curve(labels, scores)
                ax.plot(fpr, tpr, label=f"{name}", linewidth=1.2)
            elif kind == "pr":
                precision, recall, _ = precision_recall_curve(labels, scores)
                ax.plot(recall, precision, label=f"{name}", linewidth=1.2)
            elif kind == "calibration":
                bins = np.linspace(0, 1, 11)
                centres, observed = [], []
                for lo, hi in zip(bins[:-1], bins[1:]):
                    mask = (scores >= lo) & (scores < hi)
                    if mask.sum() >= 5:
                        centres.append(float(np.mean(scores[mask])))
                        observed.append(float(np.mean(labels[mask])))
                ax.plot(centres, observed, marker="o", markersize=3, label=name, linewidth=1)
            else:
                for label_value, style in ((0, ":"), (1, "-")):
                    mask = labels == label_value
                    if mask.sum():
                        ax.hist(scores[mask], bins=20, alpha=0.45, linestyle=style,
                                label=f"{name} {'AI' if label_value else 'human'}")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color="#999", linewidth=0.6)
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
            ax.set_title(f"ROC — {split}")
        elif kind == "pr":
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.set_title(f"Precision–recall — {split}")
        elif kind == "calibration":
            ax.plot([0, 1], [0, 1], color="#999", linewidth=0.6)
            ax.set_xlabel("predicted probability")
            ax.set_ylabel("observed AI rate")
            ax.set_title(f"Calibration — {split}")
        else:
            ax.set_xlabel("calibrated probability")
            ax.set_title(f"Score distributions — {split}")
        ax.legend(fontsize=5, ncol=2)
        fig.tight_layout()
        path = PLOTS_DIR / f"{kind}_{split}.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path.name)

    written.append(_confusion_plot(mode_scores, split))
    written.append(_length_plot(mode_scores, split))
    return [name for name in written if name]


def _confusion_plot(mode_scores: dict[str, dict], split: str) -> str:
    """Confusion matrix of the reported verdicts for the headline mode, as counts.

    Cells are annotated with raw counts, so the figure cannot be read as a rate without the
    denominator being visible in the same image.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - plotting is optional
        log.warning("confusion plot skipped: %s", exc)
        return ""
    candidates = [m for m in ("deep", "standard", "fast", "lite") if m in mode_scores]
    if not candidates:
        return ""
    mode = candidates[0]
    matrix = (mode_scores[mode]["metrics"] or {}).get("confusion")
    if not matrix:
        return ""
    grid = np.array(
        [
            [matrix["human_true_negative"], matrix["human_false_positive"]],
            [matrix["ai_false_negative"], matrix["ai_true_positive"]],
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(3.6, 3.0), dpi=130)
    ax.imshow(grid, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{int(grid[row, col])}", ha="center", va="center",
                    fontsize=11, color="black")
    ax.set_xticks([0, 1], ["verdict human", "verdict AI"])
    ax.set_yticks([0, 1], ["truly human", "truly AI"])
    ax.set_title(f"Confusion — {mode} on {split}\n(abstentions counted as not-AI)", fontsize=8)
    fig.tight_layout()
    path = PLOTS_DIR / f"confusion_{split}.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def _length_plot(mode_scores: dict[str, dict], split: str) -> str:
    """Accuracy by document length for every mode that produced it."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - plotting is optional
        log.warning("length plot skipped: %s", exc)
        return ""
    buckets = ["<100", "100-250", "250-500", "500+"]
    plotted = False
    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=130)
    for mode, payload in mode_scores.items():
        table = (payload["metrics"] or {}).get("accuracy_by_length") or {}
        xs = [bucket for bucket in buckets if bucket in table]
        if not xs:
            continue
        ax.plot(xs, [table[b]["accuracy"] for b in xs], marker="o", linewidth=1.2,
                label=f"{mode} (n={sum(table[b]['n'] for b in xs)})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return ""
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("document length (words)")
    ax.set_ylabel("accuracy at 0.5")
    ax.set_title(f"Accuracy by length — {split}")
    ax.legend(fontsize=6)
    fig.tight_layout()
    path = PLOTS_DIR / f"accuracy_by_length_{split}.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    seed_everything()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with timer("evaluate"):
        thresholds_raw = read_json(ARTIFACT_DIR / "thresholds.json", {}) or {}
        manifest = read_json(ARTIFACT_DIR.parent.parent / "data" / "processed" / "split_manifest.json", {}) or {}
        output: dict[str, object] = {
            "run": cfg.label,
            "data_source": manifest.get("data_source", "unknown"),
            "splits": {},
            "ablation": {},
            "plots": [],
            "notes": [],
        }
        if cfg.smoke:
            output["notes"].append(
                "SMOKE RUN: this is a correctness check on a tiny sample and must never be "
                "presented as a result."
            )
        if manifest.get("data_source") not in {"hc3", "local_csv", None}:
            output["notes"].append(
                "Data source is not the real corpus; treat every number as a pipeline check."
            )

        for split in EVAL_SPLITS:
            frame = merge(split)
            if frame.empty:
                continue
            component_scores: dict[str, dict] = {}
            for component in ("classical", "ppl_gpt2", "transformer"):
                column = f"p_{component}"
                if column not in frame:
                    continue
                mask = frame[column].notna()
                if mask.sum() < 20:
                    continue
                scores = frame.loc[mask, column].astype(float).to_numpy()
                labels = frame.loc[mask, "label"].astype(int).to_numpy()
                component_scores[component] = {
                    "metrics": _classification_metrics(scores, labels),
                    "labels": labels,
                    "scores": scores,
                }

            mode_scores: dict[str, dict] = {}
            for mode in MODES:
                path = ARTIFACT_DIR / f"stacker_{mode}.pkl"
                if not path.exists() or mode not in thresholds_raw:
                    continue
                stacker = load_stacker(path)
                started = time.perf_counter()
                scores, labels, ids = _mode_scores(frame, mode, stacker)
                latency = (time.perf_counter() - started) / max(1, len(scores)) * 1000
                if scores.size == 0:
                    continue
                mode_thresholds = ModeThresholds.from_dict(thresholds_raw[mode])
                words = frame.set_index("id").loc[ids, "words"].astype(int).to_numpy()
                metrics = _classification_metrics(scores, labels)
                if metrics:
                    metrics.update(_threshold_metrics(scores, labels, mode_thresholds, words))
                    metrics["ece"] = round(float(expected_calibration_error(scores, labels)), 4)
                    metrics["latency_ms_per_1k_texts"] = round(latency * 1000, 2)
                    metrics["components"] = list(MODE_COMPONENTS[mode])
                    metrics["breakdown"] = _breakdowns(frame, ids, scores, labels)
                    metrics["confusion"] = _confusion(scores, labels, mode_thresholds, words)
                    metrics["accuracy_by_length"] = _accuracy_by_length(scores, labels, words)
                mode_scores[mode] = {
                    "metrics": metrics, "labels": labels, "scores": scores,
                }
                log.info(
                    "%s/%s AUC=%s TPR@1%%=%s ECE=%s",
                    split, mode, metrics.get("roc_auc"), metrics.get("tpr_at_1pct_fpr"),
                    metrics.get("ece"),
                )

            output["splits"][split] = {
                "label": "held-out (out-of-domain)" if split == "test_ood" else "in-domain (optimistic)",
                "components": {k: v["metrics"] for k, v in component_scores.items()},
                "modes": {k: v["metrics"] for k, v in mode_scores.items()},
            }
            output["ablation"][split] = {
                mode: {
                    "components": list(MODE_COMPONENTS[mode]),
                    "roc_auc": mode_scores[mode]["metrics"].get("roc_auc"),
                    "tpr_at_1pct_fpr": mode_scores[mode]["metrics"].get("tpr_at_1pct_fpr"),
                    "f1": mode_scores[mode]["metrics"].get("f1"),
                }
                for mode in mode_scores
            }
            output["plots"].extend(_plots(component_scores, mode_scores, split))

        metrics_path = ARTIFACT_DIR / "metrics.json"
        write_json(metrics_path, output)
        log.info("wrote %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
