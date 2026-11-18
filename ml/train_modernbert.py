"""Fine-tune ModernBERT-base on HC3 + RAID for AI-text detection.

Why ModernBERT and not ``roberta-base`` (which ``ml/train_transformer.py`` uses):

The measured failure of the deployed detector is a *length* defect. On unseen human
text the false-positive rate is 2.4% at 60-100 words and 82.1% at 400+ words, and the
existing recipe truncates at ``max_length=256`` tokens (~190 words). So on exactly the
documents that fail, most of the text is never seen. ModernBERT supports 8192 tokens,
which removes the truncation instead of papering over it.

Memory on a 6 GB RTX 3050: ModernBERT-base is 149M params. In fp16 at 1024 tokens with
batch 8 that is roughly 2.5-3 GB of activations plus ~0.8 GB of weights, so it fits.
``--max-len`` and ``--batch-size`` are flags precisely so you can back off if it OOMs.

Usage (on your machine, inside the venv):

    python ml/train_modernbert.py --data data/processed/train.parquet \
        --out ml/artifacts/modernbert --max-len 1024 --batch-size 8 --epochs 2

    # smoke test: 200 rows, 1 epoch, a couple of minutes
    python ml/train_modernbert.py --limit 200 --epochs 1 --max-len 512 --batch-size 4

The script never fabricates a metric: if torch/CUDA/weights are unavailable it exits
non-zero with the reason rather than writing an empty artifact.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "answerdotai/ModernBERT-base"
LABELS = {0: "human", 1: "ai"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "train.parquet")
    p.add_argument("--val", type=Path, default=ROOT / "data" / "processed" / "val_stack.parquet")
    p.add_argument("--ood", type=Path, default=ROOT / "data" / "processed" / "test_ood.parquet",
                   help="held-out split used for the honest report; never trained on")
    p.add_argument("--out", type=Path, default=ROOT / "ml" / "artifacts" / "modernbert")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-len", type=int, default=1024,
                   help="8192 is the model max; 1024 covers ~99%% of this corpus")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2, help="effective batch = batch*gac")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--limit", type=int, default=0, help="cap training rows (smoke testing)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    return p.parse_args(argv)


def _require_gpu_stack() -> None:
    """Fail loudly and early; a silent CPU fallback would take hours and teach nothing."""
    try:
        import torch
        import transformers
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            "torch/transformers are missing.\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  pip install 'transformers>=4.48' accelerate datasets sentencepiece\n"
            f"  (original error: {exc})"
        )
    if not torch.cuda.is_available():
        sys.exit(
            "CUDA is not available, so training would fall back to CPU and take many hours.\n"
            "Check: python -c \"import torch; print(torch.cuda.is_available())\"\n"
            "If that prints False you installed a CPU wheel; reinstall from the cu121 index."
        )
    from packaging.version import parse as vparse

    if vparse(transformers.__version__) < vparse("4.48.0"):
        sys.exit(
            f"transformers {transformers.__version__} is too old for ModernBERT (needs >= 4.48).\n"
            "  pip install 'transformers==4.49.0'"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_gpu_stack()

    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def read(path: Path, limit: int = 0) -> pd.DataFrame:
        if not path.exists():
            sys.exit(f"missing {path} - run `python ml/prepare_data.py` first")
        frame = pd.read_parquet(path)[["text", "label"]].dropna()
        frame["label"] = frame["label"].astype(int)
        if limit:
            frame = frame.sample(n=min(limit, len(frame)), random_state=args.seed)
        return frame.reset_index(drop=True)

    train_frame = read(args.data, args.limit)
    val_frame = read(args.val)
    print(f"train {len(train_frame)} rows  val {len(val_frame)} rows")
    print(f"  train label balance: {train_frame.label.value_counts().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=LABELS, label2id={"human": 0, "ai": 1}
    )

    def encode(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len, padding=False)

    train_ds = Dataset.from_pandas(train_frame[["text", "label"]]).map(
        encode, batched=True, remove_columns=["text"]
    )
    val_ds = Dataset.from_pandas(val_frame[["text", "label"]]).map(
        encode, batched=True, remove_columns=["text"]
    )

    def metrics(eval_pred):
        logits, labels = eval_pred
        scores = logits[:, 1] - logits[:, 0]
        probs = 1 / (1 + np.exp(-scores))
        out = {"balanced_accuracy": float(balanced_accuracy_score(labels, (probs >= 0.5).astype(int)))}
        if len(set(labels.tolist())) > 1:
            out["roc_auc"] = float(roc_auc_score(labels, probs))
        return out

    steps = max(1, (len(train_ds) // (args.batch_size * args.grad_accum)))
    arguments = TrainingArguments(
        output_dir=str(args.out / "_trainer"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        fp16=args.fp16,
        logging_steps=max(10, steps // 20),
        eval_strategy="epoch",
        save_strategy="no",
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    # Honest evaluation: in-domain val AND the out-of-domain split, reported separately.
    # test_ood is 90% human, so raw accuracy there is meaningless without balanced
    # accuracy and the human false-positive rate next to it.
    report: dict[str, object] = {
        "model": args.model,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "train_rows": len(train_frame),
        "labels": LABELS,
        "splits": {},
    }

    def evaluate(frame: pd.DataFrame, name: str) -> None:
        ds = Dataset.from_pandas(frame[["text", "label"]]).map(
            encode, batched=True, remove_columns=["text"]
        )
        logits = trainer.predict(ds).predictions
        probs = 1 / (1 + np.exp(-(logits[:, 1] - logits[:, 0])))
        y = frame["label"].to_numpy()
        entry: dict[str, object] = {
            "n": int(len(y)),
            "positive_rate": float(y.mean()),
        }
        if len(set(y.tolist())) > 1:
            entry["roc_auc"] = float(roc_auc_score(y, probs))
            pred = (probs >= 0.5).astype(int)
            entry["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
            # the number that was hiding in the old report
            human = y == 0
            if human.any():
                entry["human_false_positive_rate"] = float(pred[human].mean())
        # length breakdown: this is the defect we are trying to fix
        buckets = pd.cut(frame["text"].str.split().str.len(), [0, 100, 200, 400, 100_000],
                         labels=["<100", "100-200", "200-400", "400+"])
        by_len = {}
        for label_value, group in frame.groupby(buckets, observed=True):
            mask = group.index.to_numpy()
            idx = frame.index.get_indexer(mask)
            sub_probs, sub_y = probs[idx], y[idx]
            if len(sub_y) < 10:
                continue
            entry_bucket = {"n": int(len(sub_y))}
            pred_sub = (sub_probs >= 0.5).astype(int)
            if (sub_y == 0).any():
                entry_bucket["human_fp_rate"] = float(pred_sub[sub_y == 0].mean())
            by_len[str(label_value)] = entry_bucket
        entry["by_length"] = by_len
        report["splits"][name] = entry  # type: ignore[index]

    evaluate(val_frame, "val_in_domain")
    if not args.limit:
        ood_frame = read(args.ood)
        evaluate(ood_frame, "test_ood")

    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== fine-tune report ===")
    print(json.dumps(report["splits"], indent=2))
    print(f"\nsaved to {args.out}")
    print("Next: python ml/evaluate_modernbert.py --model-dir", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
