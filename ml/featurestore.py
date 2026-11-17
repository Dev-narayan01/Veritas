"""A tiny feature store shared by the training scripts.

Each base component writes its predictions for every split into
``ml/artifacts/features/<split>.jsonl`` one row at a time, so components can be
trained and scored independently (and a crashed run can resume).  ``train_stackers``
then reads the store: it never re-scores a base model, which keeps the ablation
comparable because every mode sees exactly the same rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import ARTIFACT_DIR, PROCESSED_DIR, RunConfig, log

FEATURE_DIR = ARTIFACT_DIR / "features"
SPLITS = ("train", "val_stack", "val_cal", "test", "test_ood")
COMPONENTS = ("classical", "ppl_gpt2", "transformer")
ZERO_SHOT = ("binoculars", "fastdetect")


def load_split(split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run prepare_data.py first")
    return pd.read_parquet(path)


ZERO_SHOT_DIR = ARTIFACT_DIR / "zeroshot"


def feature_path(split: str, component: str = "component") -> Path:
    """One file per (split, component) so components never overwrite each other."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    return FEATURE_DIR / f"{split}.{component}.jsonl"


def zeroshot_path(split: str) -> Path:
    ZERO_SHOT_DIR.mkdir(parents=True, exist_ok=True)
    return ZERO_SHOT_DIR / f"{split}.jsonl"


def write_zero_shot(split: str, rows: list[dict]) -> None:
    with zeroshot_path(split).open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_zero_shot(split: str) -> list[dict]:
    path = zeroshot_path(split)
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(split: str, component: str, rows: list[dict]) -> None:
    """Append feature rows for one split and component."""
    with feature_path(split, component).open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def reset(split: str, component: str) -> None:
    feature_path(split, component).unlink(missing_ok=True)


def reset_zero_shot(split: str) -> None:
    zeroshot_path(split).unlink(missing_ok=True)


def read_rows(split: str) -> list[dict]:
    """All component rows for a split, from every component file."""
    rows: list[dict] = []
    for path in sorted(FEATURE_DIR.glob(f"{split}.*.jsonl")):
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def merge(split: str) -> pd.DataFrame:
    """Join the split metadata with every available component and zero-shot score.

    Component features live in ``features/<split>.jsonl``; zero-shot features live in
    ``zeroshot/<split>.jsonl`` because they are computed in a separate optional step.
    Rows are joined on the record id, so a partially scored split keeps per-row gaps
    instead of losing whole columns.
    """
    base = load_split(split)
    frames = []
    for rows in (read_rows(split), read_zero_shot(split)):
        if rows:
            frames.append(pd.DataFrame(rows).groupby("id", as_index=False).first())
    if not frames:
        log.warning("no features at all for %s; stackers cannot be trained on it", split)
        return base
    merged = base
    for frame in frames:
        merged = merged.merge(frame, on="id", how="left", suffixes=("", "_feat"))
    for component in COMPONENTS:
        if f"p_{component}" not in merged:
            merged[f"p_{component}"] = None
    for component in ZERO_SHOT:
        if f"z_{component}" not in merged:
            merged[f"z_{component}"] = None
    return merged


def have_component(split: str, component: str) -> bool:
    path = feature_path(split, component)
    return path.exists() and path.stat().st_size > 2


def cap_rows(frame: pd.DataFrame, cap: int, seed: int = 42) -> pd.DataFrame:
    """Stratified cap that keeps both classes balanced."""
    if len(frame) <= cap:
        return frame
    per_label = max(1, cap // max(1, frame["label"].nunique()))
    parts = [
        group.sample(min(len(group), per_label), random_state=seed)
        for _, group in frame.groupby("label", sort=True)
    ]
    return pd.concat(parts, ignore_index=True)


def describe(cfg: RunConfig) -> None:
    log.info(
        "run=%s caps: classical=%s ppl=%s transformer=%s zeroshot=%s",
        cfg.label, cfg.classical_cap, cfg.ppl_cap, cfg.transformer_cap, cfg.zeroshot_cap,
    )
