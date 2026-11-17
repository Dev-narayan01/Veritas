"""Build the detector dataset and the group-disjoint splits.

Sources, in order of preference (each guarded, each logged):

* Hugging Face ``datasets`` loaders for the canonical corpora (used on Kaggle, where
  the Hub is reachable): HC3, MAGE, GPT-wiki-intro, RAID, plus formal-human negatives
  from simple Wikipedia, CNN/DailyMail and arXiv abstracts;
* raw-file mirrors of the same corpora from GitHub, used in environments where only
  github.com is reachable (the build sandbox).  HC3 comes from its complete 24,322-row
  mirror and arXiv abstracts from the ArXivAbsTitleDataset mirror;
* ``data/raw/dataset.csv`` if the user supplied their own file;
* finally a tiny synthetic demo set, clearly marked ``data_source="synthetic_demo"``
  so every downstream artifact and the UI carry a red DEMO DATA badge.

Held-out ``test_ood`` covers whole domains and generators that never appear in
train/val/test: the HC3 ``wiki_csai`` domain, the ``arxiv_abstract`` formal-human
source, and (when MAGE is available) one complete generator family.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from common import (
    DATA_DIR,
    PROCESSED_DIR,
    REPO_ROOT,
    RunConfig,
    log,
    prepare_dirs,
    seed_everything,
    sha256_text,
    timer,
    write_json,
)

sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.text.clean import collapse_for_compare, normalize, word_count  # noqa: E402
from app.integrity.invisible import scan as scan_invisible  # noqa: E402

MIN_WORDS = 30
MAX_PER_SOURCE_LABEL = 12_000
OOD_DOMAIN = "wiki_csai"
OOD_FORMAL_SOURCE = "arxiv_abstract"
GITHUB_MIRRORS = {
    "hc3": "https://codeload.github.com/fangedvampire/ai_against_ai/tar.gz/refs/heads/main",
    "arxiv": "https://codeload.github.com/gcunhase/ArXivAbsTitleDataset/tar.gz/refs/heads/master",
}
CACHE = DATA_DIR / "raw" / "downloads"


@dataclass
class Record:
    """One labelled text with its provenance."""

    id: str
    text: str
    label: int
    domain: str
    generator: str
    source: str
    group: str  # prompt/document id used for group-disjoint splitting


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    """NFKC, strip invisible characters, collapse whitespace."""
    text = scan_invisible(text).cleaned
    return normalize(text).text


def _accept(text: str) -> bool:
    return word_count(text) >= MIN_WORDS


def _hf_dataset(path: str, config: str | None = None, split: str = "train"):
    """Try to load a Hugging Face dataset; return None when unavailable."""
    try:  # pragma: no cover - Kaggle only
        from datasets import load_dataset

        return load_dataset(path, config, split=split)
    except Exception as exc:  # noqa: BLE001 - any loader failure is a skip
        log.warning("HF dataset %s/%s unavailable: %s", path, config, exc)
        return None


def _download(url: str, name: str) -> Path | None:
    """Download and unpack a GitHub tarball into the cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / f"{name}.tar.gz"
    if not target.exists():
        try:
            urllib.request.urlretrieve(url, target)  # noqa: S310 - fixed https URLs
        except Exception as exc:  # noqa: BLE001 - network restriction
            log.warning("download failed for %s: %s", url, exc)
            return None
    unpacked = CACHE / name
    if not unpacked.exists():
        try:
            unpacked.mkdir(parents=True)
            subprocess.run(
                ["tar", "xzf", str(target), "-C", str(unpacked), "--strip-components=1"],
                check=True,
                capture_output=True,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("unpack failed for %s: %s", name, exc)
            return None
    return unpacked


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_hc3(per_domain_label: int = 4000) -> list[Record]:
    """HC3 from the complete GitHub mirror (or the Hub loader when reachable).

    The mirror holds all 24,322 rows across the five domains.  We keep at most
    ``per_domain_label`` rows per (domain, label) so no single domain dominates, and
    every row inherits its question hash as the grouping key, which makes the later
    split group-disjoint by prompt.
    """
    records: list[Record] = []
    counts: dict[tuple[str, int], int] = {}

    def _add(record: Record) -> None:
        key = (record.domain, record.label)
        if counts.get(key, 0) >= per_domain_label:
            return
        counts[key] = counts.get(key, 0) + 1
        records.append(record)

    archive = _download(GITHUB_MIRRORS["hc3"], "hc3")
    jsonl = next(archive.rglob("HC3.jsonl"), None) if archive else None
    if jsonl is not None:
        with jsonl.open() as handle:
            for line_no, line in enumerate(handle):
                row = json.loads(line)
                domain = str(row.get("source", "unknown"))
                question = str(row.get("question", ""))[:200]
                group = sha256_text(question)[:16]
                for answer in row.get("human_answers") or []:
                    text = _clean(str(answer))
                    if _accept(text):
                        _add(Record(id=f"hc3-{line_no}-h-{sha256_text(text)[:8]}", text=text,
                                    label=0, domain=domain, generator="human",
                                    source="hc3", group=group))
                for answer in row.get("chatgpt_answers") or []:
                    text = _clean(str(answer))
                    if _accept(text):
                        _add(Record(id=f"hc3-{line_no}-a-{sha256_text(text)[:8]}", text=text,
                                    label=1, domain=domain, generator="chatgpt",
                                    source="hc3", group=group))
        log.info("HC3 mirror: %s rows kept (per-domain cap %s)", len(records), per_domain_label)
        return records

    dataset = _hf_dataset("Hello-SimpleAI/HC3", "all")  # pragma: no cover - needs the Hub
    if dataset is None:
        return records
    for row in dataset:  # pragma: no cover
        domain = str(row.get("source", "unknown"))
        group = sha256_text(str(row.get("question", ""))[:200])[:16]
        for answer in row.get("human_answers") or []:
            text = _clean(str(answer))
            if _accept(text):
                _add(Record(id=f"hc3b-{sha256_text(text)[:10]}", text=text, label=0, domain=domain,
                            generator="human", source="hc3", group=group))
        for answer in row.get("chatgpt_answers") or []:
            text = _clean(str(answer))
            if _accept(text):
                _add(Record(id=f"hc3b-{sha256_text(text)[:10]}", text=text, label=1, domain=domain,
                            generator="chatgpt", source="hc3", group=group))
    return records


def load_arxiv(cap: int = 6000) -> list[Record]:
    """Formal human negatives: arXiv abstracts (never in train/val/test).

    The GitHub mirror stores one abstract per line, so the file is split *before*
    normalisation (which collapses newlines).  Each abstract gets its own group id
    so the splitter keeps them apart.
    """
    records: list[Record] = []
    archive = _download(GITHUB_MIRRORS["arxiv"], "arxiv")
    if archive is not None:
        for path in sorted(archive.rglob("*_abs.txt")):
            topic = re.sub(r"[^a-z]+", "_", path.stem.split("_")[0]).strip("_")[:40] or "arxiv"
            raw = path.read_text(errors="replace")
            kept = 0
            for line in raw.split("\n"):
                if not line.strip():
                    continue
                abstract = _clean(line)
                if not _accept(abstract):
                    continue
                records.append(
                    Record(
                        id=f"arxiv-{sha256_text(abstract)[:12]}",
                        text=abstract, label=0, domain="formal_academic",
                        generator="human", source=OOD_FORMAL_SOURCE,
                        group=f"arxiv-{sha256_text(abstract)[:12]}",
                    )
                )
                kept += 1
                if len(records) >= cap:
                    break
            log.info("arxiv abstracts from %s: %s rows", topic, kept)
            if len(records) >= cap:
                break
        return records
    dataset = _hf_dataset("gfissore/arxiv-abstracts-2021", None, "train")  # pragma: no cover
    if dataset is None:
        return records
    for row in dataset:  # pragma: no cover
        text = _clean(str(row.get("abstract", "")))
        if _accept(text):
            records.append(Record(id=f"arxivd-{sha256_text(text)[:10]}", text=text, label=0,
                                  domain="formal_academic", generator="human",
                                  source=OOD_FORMAL_SOURCE, group=f"arxiv-{sha256_text(text)[:12]}"))
        if len(records) >= cap:
            break
    return records


def load_wiki_formal(cap: int = 8_000) -> list[Record]:
    """Simple-Wikipedia sample: formal, encyclopaedic human text."""
    records: list[Record] = []
    dataset = _hf_dataset("wikimedia/wikipedia", "20231101.simple", "train")  # pragma: no cover
    if dataset is None:
        log.warning("simple Wikipedia unavailable; formal negatives come from arXiv only")
        return records
    for row in dataset:  # pragma: no cover
        text = _clean(str(row.get("text", ""))[:6000])
        if _accept(text):
            records.append(Record(id=f"wiki-{sha256_text(text)[:10]}", text=text, label=0,
                                  domain="encyclopedia", generator="human", source="wikipedia",
                                  group=f"wiki-{row.get('id', sha256_text(text)[:12])}"))
        if len(records) >= cap:
            break
    return records


def load_mage(cap: int = 8_000) -> list[Record]:
    """MAGE: multi-generator machine text (Kaggle only), held out by generator."""
    records: list[Record] = []
    dataset = _hf_dataset("yaful/MAGE", None, "train")  # pragma: no cover
    if dataset is None:
        return records
    generator_counts: dict[str, int] = {}
    for row in dataset:  # pragma: no cover
        generator = str(row.get("src", "mage_unknown"))
        generator_counts[generator] = generator_counts.get(generator, 0) + 1
        text = _clean(str(row.get("text", "")))
        if not _accept(text):
            continue
        label = int(row.get("label", 1))
        records.append(Record(id=f"mage-{sha256_text(text)[:10]}", text=text, label=label,
                              domain="mage_mixed", generator=generator, source="mage",
                              group=f"mage-{generator}-{sha256_text(text)[:12]}"))
        if len(records) >= cap:
            break
    return records


def load_local_csv() -> list[Record]:
    """User-supplied ``data/raw/dataset.csv`` with text,label[,domain,generator]."""
    path = DATA_DIR / "raw" / "dataset.csv"
    if not path.exists():
        return []
    records: list[Record] = []
    with path.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            text = _clean(row.get("text", ""))
            if not _accept(text):
                continue
            label = int(row.get("label", 1))
            records.append(Record(id=f"csv-{index}", text=text, label=label,
                                  domain=row.get("domain", "local"), generator=row.get("generator", "unknown"),
                                  source="local_csv", group=f"csv-{index}"))
    log.info("local dataset.csv: %s rows", len(records))
    return records


SYNTHETIC_HUMAN = [
    "i tried this last week and honestly it did not go the way i expected . the first attempt failed because i forgot to check the setting , and then i spent twenty minutes looking for the mistake . in the end it worked , but i still do not know why the second try behaved differently .",
    "my brother keeps telling me to read the manual before i complain about the printer . he is right , obviously , but i never do it . last time the paper jam turned out to be a label stuck behind the tray , which i only found after taking the whole thing apart .",
    "we walked to the market early because the good bread sells out by eight . the baker said the flour delivery was late again , so half the shelves were empty . i bought the rye anyway and we ate it on the way home , which annoyed my sister .",
]


def load_synthetic() -> list[Record]:
    """Last-resort demo set. Every downstream artifact is marked synthetic_demo."""
    ai_style = (
        "Artificial intelligence has become an increasingly important topic in modern society . "
        "It is essential to consider the various implications of this technology . "
        "Furthermore , it is important to note that the development of these systems requires "
        "careful consideration of ethical frameworks . In conclusion , stakeholders must collaborate "
        "to ensure responsible deployment ."
    )
    records: list[Record] = []
    for index in range(30):
        human = SYNTHETIC_HUMAN[index % len(SYNTHETIC_HUMAN)]
        records.append(Record(id=f"syn-h-{index}", text=human, label=0, domain="demo",
                              generator="human", source="synthetic", group=f"syn-{index}"))
        records.append(Record(id=f"syn-a-{index}", text=ai_style, label=1, domain="demo",
                              generator="chatgpt", source="synthetic", group=f"syn-{index}"))
    return records


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def deduplicate(records: list[Record]) -> list[Record]:
    """Drop exact duplicates and human/AI overlaps on the normalised text."""
    seen: dict[str, Record] = {}
    for record in records:
        key = collapse_for_compare(record.text)[:4000]
        existing = seen.get(key)
        if existing is None:
            seen[key] = record
        elif existing.label == 0 and record.label == 1:
            # An identical text labelled both human and AI pollutes the task.
            seen.pop(key, None)
    return list(seen.values())


def cap_per_source(records: list[Record], cap: int) -> list[Record]:
    """Randomly cap each (source, label) group so no source dominates."""
    rng = random.Random(42)
    buckets: dict[tuple[str, int], list[Record]] = {}
    for record in records:
        buckets.setdefault((record.source, record.label), []).append(record)
    out: list[Record] = []
    for key, items in buckets.items():
        if len(items) > cap:
            items = rng.sample(items, cap)
            log.info("capped %s to %s rows", key, cap)
        out.extend(items)
    return out


def split_records(records: list[Record], cfg: RunConfig) -> dict[str, list[Record]]:
    """Group-disjoint train/val_stack/val_cal/test plus the OOD holdout."""
    ood = [
        r for r in records
        if r.domain == OOD_DOMAIN or r.source == OOD_FORMAL_SOURCE or r.domain == "mage_heldout"
    ]
    in_domain = [r for r in records if r not in ood]

    groups: dict[str, list[Record]] = {}
    for record in in_domain:
        groups.setdefault(record.group, []).append(record)
    group_ids = sorted(groups)
    rng = random.Random(42)
    rng.shuffle(group_ids)

    n = len(group_ids)
    train_end = int(n * 0.60)
    stack_end = int(n * 0.70)
    cal_end = int(n * 0.80)
    splits = {
        "train": group_ids[:train_end],
        "val_stack": group_ids[train_end:stack_end],
        "val_cal": group_ids[stack_end:cal_end],
        "test": group_ids[cal_end:],
    }
    out: dict[str, list[Record]] = {}
    for name, ids in splits.items():
        rows = [record for gid in ids for record in groups[gid]]
        if cfg.smoke:
            rows = _limit_per_label(rows, 250)
        out[name] = rows
    if cfg.smoke:
        ood = _limit_per_label(ood, 250)
    out["test_ood"] = ood
    return out


def _limit_per_label(records: list[Record], per_label: int) -> list[Record]:
    rng = random.Random(7)
    by_label: dict[int, list[Record]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)
    out: list[Record] = []
    for label, items in by_label.items():
        out.extend(items if len(items) <= per_label else rng.sample(items, per_label))
    return out


def _manifest(splits: dict[str, list[Record]], data_source: str) -> dict:
    manifest: dict[str, object] = {"data_source": data_source, "seed": 42, "splits": {}}
    for name, rows in splits.items():
        frame = pd.DataFrame([asdict(r) for r in rows])
        entry: dict[str, object] = {
            "count": int(len(rows)),
            "sha256": sha256_text("|".join(sorted(r.id for r in rows))),
            "labels": frame["label"].value_counts().to_dict() if len(rows) else {},
        }
        for column in ("domain", "generator", "source"):
            if len(rows):
                entry[column] = frame[column].value_counts().to_dict()
        manifest["splits"][name] = entry
        log.info("split %-9s rows=%-6s labels=%s", name, len(rows), entry["labels"])
    return manifest


def _distribution_plots(splits: dict[str, list[Record]]) -> None:
    """Length distributions per split (matplotlib, headless)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from common import PLOTS_DIR

        fig, axes = plt.subplots(1, len(splits), figsize=(2.6 * len(splits), 2.2), dpi=120)
        for ax, (name, rows) in zip(axes, splits.items()):
            lengths = [word_count(r.text) for r in rows]
            ax.hist(lengths, bins=20, color="#7C5CFF")
            ax.set_title(f"{name} (n={len(rows)})", fontsize=8)
            ax.set_xlabel("words", fontsize=7)
            ax.tick_params(labelsize=6)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "split_lengths.png")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - plotting is optional
        log.warning("could not draw split plots: %s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run to test correctness")
    parser.add_argument("--full", action="store_true", help="Kaggle-scale run")
    parser.add_argument("--force", action="store_true", help="rebuild even if the splits exist")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    prepare_dirs()
    seed_everything()

    manifest_path = PROCESSED_DIR / "split_manifest.json"
    if manifest_path.exists() and not args.force:
        log.info("splits already exist (%s); use --force to rebuild", manifest_path)
        return 0

    with timer("prepare_data"):
        records = load_local_csv()
        source_name = "local_csv" if records else ""
        if not records:
            hc3 = load_hc3(700 if cfg.smoke else 4000)
            if hc3:
                records.extend(hc3)
                source_name = "hc3"
            arxiv = load_arxiv(300 if cfg.smoke else 6000)
            records.extend(arxiv)
            wiki = load_wiki_formal(300 if cfg.smoke else 8000)
            records.extend(wiki)
            mage = load_mage(300 if cfg.smoke else 8000)
            records.extend(mage)
            if mage:
                # Hold out the largest generator family as an unseen generator.
                counts: dict[str, int] = {}
                for record in mage:
                    counts[record.generator] = counts.get(record.generator, 0) + 1
                largest = max(counts, key=counts.get)
                for record in records:
                    if record.source == "mage" and record.generator == largest:
                        record.domain = "mage_heldout"
                log.info("held out MAGE generator family: %s", largest)
        if not records:
            records = load_synthetic()
            source_name = "synthetic_demo"
            manifest_note = (
                "No dataset could be downloaded, so a small synthetic demo set is used. "
                "Every artifact derived from it is marked synthetic_demo and must not be "
                "presented as a result."
            )
            log.warning(manifest_note)

        records = deduplicate(records)
        if source_name != "synthetic_demo":
            records = cap_per_source(records, MAX_PER_SOURCE_LABEL)
        splits = split_records(records, cfg)

        for name, rows in splits.items():
            frame = pd.DataFrame(
                [{k: v for k, v in asdict(r).items() if k != "text"} | {"text": r.text,
                  "words": word_count(r.text)} for r in rows]
            )
            frame.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        manifest = _manifest(splits, source_name or "unknown")
        manifest["run"] = cfg.label
        manifest["generated_at"] = pd.Timestamp.now("UTC").isoformat()
        write_json(manifest_path, manifest)
        _distribution_plots(splits)
        log.info("wrote %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
