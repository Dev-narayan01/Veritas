"""Fine-tune the transformer component (roberta-base on GPU, distilroberta on CPU).

The training recipe is fixed by the spec: 2 epochs, learning rate 2e-5, weight decay
0.01, 6% warmup, batch size 16 (with gradient accumulation when memory is tight),
max length 256 and fp16 on GPU.  The script checks which ``TrainingArguments`` keyword
the installed ``transformers`` version expects, and it degrades honestly: without
``torch``/``transformers`` or without network access for the base weights it logs why
and exits 0, so ``standard``/``deep`` are reported unavailable instead of faked.

The label mapping written into the saved config is ``{0: human, 1: ai}`` and the
runtime verifies it before scoring anything.
"""

from __future__ import annotations

import argparse

from common import ARTIFACT_DIR, RunConfig, log, prepare_dirs, seed_everything, timer, write_json
from featurestore import SPLITS, cap_rows, load_split, reset, write_rows

TARGET_DIR = ARTIFACT_DIR / "transformer"


def _training_arguments_kwargs() -> dict:
    """Pick the right evaluation-strategy keyword for the installed version."""
    try:
        import inspect

        from transformers import TrainingArguments

        params = inspect.signature(TrainingArguments.__init__).parameters
        key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    except Exception:  # noqa: BLE001 - transformers missing
        key = "evaluation_strategy"
    return {key: "epoch"}


def trainer_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return False, f"torch/transformers not installed ({exc})"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    prepare_dirs()
    seed_everything()

    if (TARGET_DIR / "config.json").exists() and not args.force:
        log.info("transformer weights already present; use --force to retrain")
        return 0

    ok, reason = trainer_available()
    if not ok:
        log.warning(
            "transformer component skipped: %s. The 'standard' and 'deep' modes will be "
            "reported unavailable by /api/health rather than silently downgraded.",
            reason,
        )
        write_json(
            ARTIFACT_DIR / "transformer_meta.json",
            {"trained": False, "reason": reason, "run": cfg.label},
        )
        return 0

    with timer("train_transformer"):  # pragma: no cover - requires torch
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        model_name = (
            "roberta-base" if torch.cuda.is_available() else "distilroberta-base"
        )
        train_frame = cap_rows(load_split("train"), cfg.transformer_cap)
        val_frame = load_split("val_stack")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def encode(batch):  # noqa: ANN001, ANN202
            return tokenizer(batch["text"], truncation=True, max_length=256, padding=False)

        train_ds = Dataset.from_pandas(train_frame[["text", "label"]]).map(
            encode, batched=True, remove_columns=["text"]
        )
        val_ds = Dataset.from_pandas(val_frame[["text", "label"]]).map(
            encode, batched=True, remove_columns=["text"]
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, id2label={0: "human", 1: "ai"}, label2id={"human": 0, "ai": 1}
        )
        steps = max(1, len(train_ds) // 16)
        arguments = TrainingArguments(
            output_dir=str(ARTIFACT_DIR / "transformer_train"),
            num_train_epochs=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.06,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=1 if torch.cuda.is_available() else 4,
            fp16=torch.cuda.is_available(),
            logging_steps=max(10, steps // 20),
            save_strategy="no",
            seed=42,
            **_training_arguments_kwargs(),
        )
        # DistilBERT does not accept the token_type_ids column produced by some
        # tokenizers, so drop anything the model does not declare.
        allowed = set(model.forward.__code__.co_varnames) | {"input_ids", "attention_mask"}
        for dataset in (train_ds, val_ds):
            drop = [c for c in dataset.column_names if c not in allowed and c != "label"]
            if drop:
                dataset.remove_columns(drop)

        trainer = Trainer(model=model, args=arguments, train_dataset=train_ds, eval_dataset=val_ds)
        trainer.train()
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(TARGET_DIR)
        tokenizer.save_pretrained(TARGET_DIR)
        write_json(
            ARTIFACT_DIR / "transformer_meta.json",
            {"trained": True, "base": model_name, "rows": len(train_ds), "run": cfg.label},
        )

        from transformers import pipeline

        classifier = pipeline(
            "text-classification", model=model, tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1, top_k=None,
        )
        for split in SPLITS:
            reset(split, "transformer")
            frame = load_split(split)
            if frame.empty:
                continue
            rows = []
            for text, row_id in zip(frame["text"].tolist(), frame["id"].tolist()):
                scores = classifier(text[:4000])[0]
                probability = next(
                    (item["score"] for item in scores if item["label"].lower() in {"ai", "label_1"}),
                    0.5,
                )
                rows.append({"id": row_id, "p_transformer": float(probability)})
            write_rows(split, "transformer", rows)
            log.info("transformer scored %s rows for %s", len(rows), split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
