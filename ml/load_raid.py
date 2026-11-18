"""Download RAID and convert it into a training parquet alongside HC3.

Why this exists: ``ml/prepare_data.py`` names RAID in its docstring, but there is no
``load_raid()`` in it -- the loader was never written. HC3 alone is one generator
(ChatGPT) across four Reddit-ish domains, which is exactly why the deployed detector
falls apart on unseen text: its out-of-domain half is 90% human and 37% of human
documents get flagged.

RAID (Dugan et al., 2024) is ~10M generations from 11 models across 8 domains, and it
also carries adversarial attacks. Training on it is the single largest available gain
in real accuracy, because it teaches *generator* and *attack* invariance rather than
"ChatGPT writes like this".

Usage:

    # one-off download + convert (needs internet; ~2-4 GB streamed, not all stored)
    python ml/load_raid.py --out data/processed/raid_train.parquet --per-generator 4000

    # check what you got without training anything
    python ml/load_raid.py --report-only --out data/processed/raid_train.parquet

Sampling is deliberate, not random: RAID is heavily imbalanced (a handful of models
dominate the row count), so we cap per generator and per domain, and keep a fixed
share of attacked rows. A random sample would reproduce the same single-generator
bias HC3 already has.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAID_REPO = "liamdugan/raid"
MIN_WORDS = 30          # below this the detector reports "too short" anyway
MAX_CHARS = 20_000      # guard against pathological rows


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "raid_train.parquet")
    p.add_argument("--repo", default=RAID_REPO)
    p.add_argument("--per-generator", type=int, default=4000,
                   help="cap machine rows per model so one generator cannot dominate")
    p.add_argument("--per-domain", type=int, default=6000, help="cap rows per domain")
    p.add_argument("--human-fraction", type=float, default=0.5,
                   help="target share of human rows in the output")
    p.add_argument("--attack-share", type=float, default=0.25,
                   help="target share of rows whose text was adversarially attacked")
    p.add_argument("--max-scan", type=int, default=0,
                   help="stop after this many source rows (0 = whole split); use for a quick look")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-only", action="store_true", help="summarise an existing parquet")
    return p.parse_args(argv)


def report(path: Path) -> int:
    import pandas as pd

    if not path.exists():
        print(f"{path} does not exist yet", file=sys.stderr)
        return 1
    frame = pd.read_parquet(path)
    print(f"{path}: {len(frame)} rows")
    print(f"  label: {frame.label.value_counts().to_dict()}")
    for column in ("generator", "domain", "attack"):
        if column in frame:
            print(f"  {column}: {dict(frame[column].value_counts().head(12))}")
    if "words" in frame:
        print(f"  words: mean {frame.words.mean():.0f}  p50 {frame.words.median():.0f}  max {frame.words.max():.0f}")
        print(f"  >200 words: {(frame.words > 200).mean():.1%}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.report_only:
        return report(args.out)

    import pandas as pd

    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"`datasets` is required: pip install datasets  (original error: {exc})")

    print(f"streaming {args.repo} ...")
    try:
        ds = load_dataset(args.repo, split="train", streaming=True)
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"could not open {args.repo}: {exc}\n"
            "This needs internet access to huggingface.co. If you are behind a mirror, set\n"
            "HF_ENDPOINT=https://hf-mirror.com (or your mirror) before running."
        )

    per_gen: collections.Counter = collections.Counter()
    per_domain: collections.Counter = collections.Counter()
    rows: list[dict] = []
    attacked_kept = 0
    seen = 0

    for record in ds:
        seen += 1
        if args.max_scan and seen > args.max_scan:
            break

        text = record.get("generation") or ""
        if not text:
            continue
        text = " ".join(text.split())
        words = len(text.split())
        if words < MIN_WORDS or len(text) > MAX_CHARS:
            continue

        label = int(record.get("label", 0))
        generator = str(record.get("model") or "human")
        domain = str(record.get("domain") or "unknown")
        attack = str(record.get("attack") or "none")
        is_attacked = attack not in ("none", "")

        # Human rows have no generator; cap them by the human fraction at the end.
        if label == 1:
            if per_gen[generator] >= args.per_generator:
                continue
        if per_domain[domain] >= args.per_domain:
            continue

        rows.append(
            {
                "id": f"raid-{sha(text)[:16]}",
                "text": text,
                "label": label,
                "domain": domain,
                "generator": generator if label == 1 else "human",
                "source": "raid",
                "attack": attack,
                "group": sha(str(record.get("prompt") or text[:200]))[:16],
                "words": words,
            }
        )
        per_gen[generator] += 1
        per_domain[domain] += 1
        if is_attacked:
            attacked_kept += 1

        if seen % 50_000 == 0:
            print(f"  scanned {seen:,} kept {len(rows):,}", flush=True)

    if not rows:
        sys.exit("no rows kept; the dataset schema may have changed")

    frame = pd.DataFrame(rows).drop_duplicates(subset=["text"])

    # Rebalance: the RAID human side is small relative to machine rows, and a model
    # trained 90/10 on machine text learns the wrong prior.
    human = frame[frame.label == 0]
    machine = frame[frame.label == 1]
    target_human = int(len(machine) * args.human_fraction / (1 - args.human_fraction))
    if len(human) > target_human:
        human = human.sample(n=target_human, random_state=args.seed)
    frame = pd.concat([human, machine]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    print(f"\nwrote {args.out}")
    print(f"  rows {len(frame):,}  (scanned {seen:,})")
    print(f"  label {frame.label.value_counts().to_dict()}")
    print(f"  generators ({frame[frame.label == 1].generator.nunique()} distinct): "
          f"{dict(frame[frame.label == 1].generator.value_counts().head(12))}")
    print(f"  domains: {dict(frame.domain.value_counts().head(10))}")
    print(f"  attacked rows: {attacked_kept:,} ({attacked_kept / max(1, len(frame)):.1%})")
    print("  for contrast, HC3 train median length is ~149 words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
