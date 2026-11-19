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
    p.add_argument("--min-rows", type=int, default=100,
                   help="refuse to train below this many rows; catches the demo fallback")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--allow-cpu", action="store_true",
                   help="train on CPU instead of exiting; for smoke tests only")
    p.add_argument("--compile", action="store_true", default=False,
                   help="let ModernBERT torch.compile its embeddings; needs gcc and "
                        "costs minutes of warmup, for a marginal speedup")
    return p.parse_args(argv)


# Windows refuses to load an unsigned DLL when Smart App Control (or an MDM/WDAC
# policy) is enforcing. PyTorch ships several unsigned DLLs in torch/lib, so the
# failure looks like a broken install but is a security policy. Measured on a real
# machine: the PyPI CPU wheel imported fine and the cu126 wheel died on
# c10_cuda.dll with WinError 4551, so "reinstall" is the wrong advice twice over.
_CODE_INTEGRITY_MARKERS = (
    "Application Control policy has blocked",
    "An Application Control policy has blocked",
    "did not meet the Enterprise signing level",
    "WinError 4551",
    "0xc0e90002",
)

_CODE_INTEGRITY_HELP = """\
Windows Code Integrity blocked a PyTorch DLL, so CUDA cannot start.

This is Smart App Control (policy {{0283AC0F-FFF1-49AE-ADA1-8A933130CAD6}}) or a
WDAC/MDM policy refusing to load unsigned binaries. torch/lib ships unsigned DLLs,
so reinstalling, changing the CUDA tag, or upgrading torch will not clear it.
PyTorch tracks the signing gap as pytorch/pytorch#199711.

Your options, in the order worth trying:
  1. WSL2 - Linux has no Smart App Control, and the Windows NVIDIA driver already
     provides CUDA to WSL. Keeps the security feature on. See docs/SETUP_WINDOWS.md.
  2. Turn Smart App Control off (Windows Security > App and browser control).
     One way: it cannot be switched back on without reinstalling Windows.
  3. Train on CPU with --allow-cpu. Hours instead of minutes, but it runs.

To confirm the diagnosis: import torch after disabling SAC and check
torch.cuda.is_available(). If that flips to True, the policy was the cause."""


def _chain_has_code_integrity_block(exc: BaseException) -> bool:
    """Walk __cause__/__context__: the block is often wrapped by an ImportError."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(exc, "winerror", None) == 4551:
            return True
        text = str(exc)
        if any(marker in text for marker in _CODE_INTEGRITY_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__  # type: ignore[assignment]
    return False


def _require_gpu_stack(allow_cpu: bool = False) -> str:
    """Return the device to train on. Fail loudly rather than silently crawling."""
    try:
        import torch
        import transformers
    except Exception as exc:  # noqa: BLE001
        if _chain_has_code_integrity_block(exc):
            sys.exit(_CODE_INTEGRITY_HELP)
        sys.exit(
            "torch/transformers are missing.\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu126\n"
            "  pip install 'transformers==4.49.0' accelerate datasets sentencepiece\n"
            f"  (original error: {exc})"
        )
    from packaging.version import parse as vparse

    if vparse(transformers.__version__) < vparse("4.48.0"):
        sys.exit(
            f"transformers {transformers.__version__} is too old for ModernBERT (needs >= 4.48).\n"
            "  pip install 'transformers==4.49.0'"
        )
    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        print("! CUDA unavailable, training on CPU because --allow-cpu was passed.")
        print("! Expect hours, not minutes. This is a smoke test, not a real run.")
        return "cpu"
    sys.exit(
        "CUDA is not available and --allow-cpu was not passed.\n"
        "Check: python -c \"import torch; print(torch.version.cuda, torch.cuda.is_available())\"\n"
        "  available True  -> pass --allow-cpu to train on CPU anyway\n"
        "  cuda None       -> you have the PyPI CPU wheel; install from the cu126 index\n"
        "  import error    -> likely a Windows code integrity block; see the message above\n"
        "See docs/SETUP_WINDOWS.md."
    )


def columns_for_report(available) -> list[str]:
    """Columns to keep when loading a split: always text/label, plus domain if present.

    Module-level so a test can prove the loader keeps ``domain``. When this returned
    only ``["text", "label"]``, ``build_split_report``'s ``domains=None`` path was the
    only one ever taken and no report contained a per-domain breakdown.
    """
    cols = ["text", "label"]
    if "domain" in available:
        cols.append("domain")
    return cols


def _rate(pred, mask) -> float:
    """Fraction of ``pred`` that is 1 inside ``mask``; raises on an empty mask."""
    return float(pred[mask].mean())


def build_split_report(
    labels,
    probs,
    domains=None,
    word_counts=None,
) -> dict[str, object]:
    """Per-split metrics for one evaluated split.

    Pure function took this out of the training script so the reporting rules can be
    tested without a GPU. The previous version lived inline behind
    ``if "domain" in frame.columns`` and the loader never kept that column, so
    ``by_domain`` was dead code in every report the project ever produced while the
    tests (which grepped the source) passed.

    ``labels``/``probs`` are aligned positional arrays. ``domains`` and
    ``word_counts`` are optional; each enables its own breakdown.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    y = np.asarray(labels)
    p = np.asarray(probs, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"labels {y.shape} and probs {p.shape} are not aligned")

    entry: dict[str, object] = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
    }
    pred = (p >= 0.5).astype(int)
    if len(set(y.tolist())) > 1:
        entry["roc_auc"] = float(roc_auc_score(y, p))
        entry["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    # Reported for one-class splits too: a split of human text with no AI rows has no
    # AUC but its false-positive rate is exactly the number that matters.
    human = y == 0
    if human.any():
        entry["human_false_positive_rate"] = _rate(pred, human)
    ai = y == 1
    if ai.any():
        entry["ai_recall"] = _rate(pred, ai)

    # Per-domain breakdown. Aggregates across a mixed test set are not comparable
    # between runs: --mix-formal changes how many arXiv rows the held-out split
    # contains, so the same model can look better on a smaller, easier set.
    if domains is not None:
        domain_values = np.asarray(domains)
        if domain_values.shape != y.shape:
            raise ValueError("domains and labels are not aligned")
        by_domain: dict[str, dict[str, object]] = {}
        # Positional masks, not get_indexer: the latter returns -1 for misses on a
        # non-unique index and would silently index from the end of `probs`.
        for domain_name in pd.unique(domain_values):
            idx = np.flatnonzero(domain_values == domain_name)
            if len(idx) < 10:
                continue
            dom_y, dom_pred = y[idx], pred[idx]
            dom_entry: dict[str, object] = {
                "n": int(len(idx)),
                "positive_rate": float(dom_y.mean()),
            }
            if len(set(dom_y.tolist())) > 1:
                dom_entry["roc_auc"] = float(roc_auc_score(dom_y, p[idx]))
                dom_entry["balanced_accuracy"] = float(
                    balanced_accuracy_score(dom_y, dom_pred)
                )
            if (dom_y == 0).any():
                # This is the collision that made the last two runs unreadable:
                # arxiv_cv/arxiv_lg are *seen* domains, arxiv_ai and wiki_csai are
                # unseen, and averaging them together hides both facts.
                dom_entry["human_false_positive_rate"] = _rate(dom_pred, dom_y == 0)
            if (dom_y == 1).any():
                dom_entry["ai_recall"] = _rate(dom_pred, dom_y == 1)
            by_domain[str(domain_name)] = dom_entry
        entry["by_domain"] = by_domain

    if word_counts is not None:
        counts = np.asarray(word_counts)
        if counts.shape != y.shape:
            raise ValueError("word_counts and labels are not aligned")
        by_len: dict[str, dict[str, object]] = {}
        for bucket, lo, hi in (
            ("<100", 0, 100), ("100-200", 100, 200),
            ("200-400", 200, 400), ("400+", 400, np.inf),
        ):
            idx = np.flatnonzero((counts > lo) & (counts <= hi))
            if len(idx) < 10:
                continue
            sub_y, sub_pred = y[idx], pred[idx]
            bucket_entry: dict[str, object] = {"n": int(len(idx))}
            if (sub_y == 0).any():
                bucket_entry["human_fp_rate"] = _rate(sub_pred, sub_y == 0)
            by_len[bucket] = bucket_entry
        entry["by_length"] = by_len

    return entry


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _require_gpu_stack(allow_cpu=args.allow_cpu)
    if device == "cpu" and args.fp16:
        # fp16 on CPU is far slower than fp32 and NumPy has no useful fp16 path here.
        args.fp16 = False
        print("! fp16 disabled: unsupported on CPU.")

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
        available = pd.read_parquet(path)
        # Keep `domain` when the split has it. Selecting only text/label here silently
        # starved the per-domain report: `by_domain` guards on
        # `if "domain" in frame.columns`, so the branch was dead and every report
        # carried no per-domain numbers at all. The guard could never fire, and the
        # tests only asserted the branch existed, not that it ran.
        wanted = columns_for_report(available.columns)
        frame = available[wanted].dropna(subset=["text", "label"])
        frame["label"] = frame["label"].astype(int)
        if limit:
            frame = frame.sample(n=min(limit, len(frame)), random_state=args.seed)
        return frame.reset_index(drop=True)

    train_frame = read(args.data, args.limit)
    val_frame = read(args.val)
    print(f"train {len(train_frame)} rows  val {len(val_frame)} rows")
    print(f"  train label balance: {train_frame.label.value_counts().to_dict()}")

    # prepare_data.py falls back to a 60-row synthetic demo set when it cannot reach
    # a dataset source, and split_records then puts one row in train. Training on that
    # produces a number with no meaning, which is exactly what this project promises
    # not to publish. Stop and say which file is wrong instead.
    if len(train_frame) < args.min_rows:
        sys.exit(
            f"Only {len(train_frame)} training rows in {args.data} (minimum "
            f"{args.min_rows}).\n"
            "This is the synthetic demo fallback, not real data - prepare_data.py uses\n"
            "it when no dataset source could be reached, and it is not a result.\n"
            "Rebuild the splits and read what it says about the source:\n"
            "  python ml/prepare_data.py --force\n"
            "If it logs 'synthetic_demo', the downloads failed; the manifest at\n"
            "data/processed/split_manifest.json records which sources loaded."
        )
    if train_frame.label.nunique() < 2:
        sys.exit(
            f"Training data has only one class: {train_frame.label.value_counts().to_dict()}\n"
            "A classifier cannot learn from that, and any metric it reports is noise."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # ModernBERT calls torch.compile on its embeddings unless reference_compile is
    # False (modeling_modernbert.py:218, gated by config.reference_compile). On a GPU
    # that path goes through Triton, which JIT-builds a CUDA extension and needs a C
    # compiler; without one it raises
    #   InductorError: RuntimeError: Failed to find C compiler
    # The flag is consulted in forward(), so passing it to from_pretrained has no
    # effect - _maybe_set_compile() only auto-disables on cpu/mps, not cuda.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=LABELS, label2id={"human": 0, "ai": 1}
    )
    if not args.compile:
        # getattr-guarded: transformers 5.x removed this attribute along with the
        # @torch.compile decorators, and an unconditional set would create the
        # attribute on a config that no longer reads it.
        if hasattr(model.config, "reference_compile"):
            model.config.reference_compile = False
            print("torch.compile disabled for ModernBERT (no C compiler needed)")

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
    # Save the model Trainer actually trained. With fp16 the Trainer keeps a fp16
    # copy for forward passes and updates the fp32 master separately, so saving the
    # original `model` object can write untrained weights; the artefacts must match
    # the numbers reported below.
    trained = trainer.model
    trained.config.reference_compile = False
    trained.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    model = trained

    # Honest evaluation: in-domain val AND the out-of-domain split, reported separately.
    # test_ood is 90% human, so raw accuracy there is meaningless without balanced
    # accuracy and the human false-positive rate next to it.
    report: dict[str, object] = {
        "model": args.model,
        "device": device,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "train_rows": len(train_frame),
        "labels": LABELS,
        # A --limit run sees a random subset of train while still evaluating the full
        # val and OOD splits, so its numbers measure plumbing, not generalisation.
        # Record it in the artefact so no one can quote one as a result later.
        "limited": bool(args.limit),
        "note": (
            f"PLUMBING CHECK ONLY: trained on {len(train_frame)} rows via --limit. "
            "Not a result."
            if args.limit
            else "full training run"
        ),
        "splits": {},
    }

    def evaluate(frame: pd.DataFrame, name: str) -> None:
        ds = Dataset.from_pandas(frame[["text", "label"]]).map(
            encode, batched=True, remove_columns=["text"]
        )
        logits = trainer.predict(ds).predictions
        probs = 1 / (1 + np.exp(-(logits[:, 1] - logits[:, 0])))
        entry = build_split_report(
            labels=frame["label"].to_numpy(),
            probs=probs,
            domains=frame["domain"].to_numpy() if "domain" in frame.columns else None,
            word_counts=frame["text"].str.split().str.len().to_numpy(),
        )
        report["splits"][name] = entry  # type: ignore[index]

    evaluate(val_frame, "val_in_domain")
    # Always evaluate OOD, including --limit smoke runs. It is encode-only (~1 min for
    # 7484 rows) and test_ood is where this project's known failure lives: 2.4% human
    # FPR at 60-100 words rising to 82.1% at 400+. Omitting it left a smoke report
    # that looked like a success because the split that exposes the defect was absent.
    if args.ood.exists():
        evaluate(read(args.ood), "test_ood")
    else:
        print(f"! {args.ood} missing - no out-of-domain numbers in this report")

    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== fine-tune report ===")
    print(json.dumps(report["splits"], indent=2))
    print(f"\nsaved to {args.out}")
    print("artefacts: model.safetensors, tokenizer files, eval_report.json")
    if args.limit:
        print("A --limit run trains on a subset and is a plumbing check, not a result.")
    else:
        print(f"Full run: {len(train_frame)} rows, {args.epochs} epochs, "
              f"max_len {args.max_len}. Compare OOD numbers per domain, not aggregate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
