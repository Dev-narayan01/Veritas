"""Build a transparent synthetic plagiarism benchmark.

Host documents are assembled from held-out text (test-split human documents and
arXiv abstracts) that are **not** in the corpus index, so a detected match cannot be a
self-match artefact.  Source passages with known character spans are inserted as:

* ``verbatim``      — copied unchanged,
* ``light_edit``    — 10–20% of words replaced by synonyms,
* ``reordered``     — sentences permuted but wording kept,
* ``heavy_paraphrase`` — produced by a T5 paraphraser when one is available; skipped
  and labelled as skipped otherwise.

Insertion proportions are 5%, 20% and 50% of the host length, and negatives contain no
inserted passage at all.  The benchmark is split into validation and test halves and
flagged as synthetic everywhere it is reported.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, dataclass, field

from common import ARTIFACT_DIR, PERTURBATIONS, REPO_ROOT, RunConfig, log, seed_everything, timer, write_json
from featurestore import load_split

sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.text.segment import split_sentences  # noqa: E402

OUT_DIR = ARTIFACT_DIR / "plag_benchmark"
HOST_COUNT = 320
NEGATIVE_COUNT = 120
PROPORTIONS = (0.05, 0.20, 0.50)
OBFUSCATIONS = ("verbatim", "light_edit", "reordered", "heavy_paraphrase")


@dataclass
class Case:
    """One benchmark document with ground-truth spans."""

    id: str
    host_id: str
    text: str
    obfuscation: str
    proportion: float
    spans: list[list[int]] = field(default_factory=list)
    source_texts: list[str] = field(default_factory=list)
    is_negative: bool = False


def _paraphraser():  # noqa: ANN202
    try:  # pragma: no cover - needs torch + weights
        import torch
        from transformers import pipeline

        return pipeline(
            "text2text-generation", model="humarin/chatgpt_paraphraser_on_T5-base",
            device=0 if torch.cuda.is_available() else -1,
        )
    except Exception as exc:  # noqa: BLE001 - optional model
        log.warning("heavy paraphrase unavailable (%s); that type is skipped", exc)
        return None


def _obfuscate(text: str, kind: str, paraphraser) -> str | None:  # noqa: ANN001
    if kind == "verbatim":
        return text
    if kind == "light_edit":
        return PERTURBATIONS["synonym_20"](text, "")
    if kind == "reordered":
        return PERTURBATIONS["shuffle_sentences"](text, "")
    if kind == "heavy_paraphrase":
        if paraphraser is None:
            return None
        try:  # pragma: no cover
            return paraphraser(f"paraphrase: {text}", max_length=400)[0]["generated_text"]
        except Exception as exc:  # noqa: BLE001
            log.debug("paraphrase failed: %s", exc)
            return None
    return None


def _build_case(host: dict, source: dict, kind: str, proportion: float, paraphraser) -> Case | None:  # noqa: ANN001
    passage = _obfuscate(source["text"], kind, paraphraser)
    if not passage:
        return None
    host_text: str = host["text"]
    target_words = max(40, int(len(host_text.split()) * proportion))
    passage_words = passage.split()
    if len(passage_words) > target_words:
        passage = " ".join(passage_words[:target_words])
    # Insert at a paragraph boundary near the middle so spans stay meaningful.
    sentences = split_sentences(host_text)
    insert_at = len(host_text) // 2
    if sentences:
        middle = min(range(len(sentences)), key=lambda i: abs(sentences[i].start - insert_at))
        insert_at = sentences[middle].start
    else:
        insert_at = min(insert_at, len(host_text))
    text = host_text[:insert_at] + passage + " " + host_text[insert_at:]
    start = insert_at
    return Case(
        id=f"{host['id']}-{kind}-{int(proportion * 100)}",
        host_id=str(host["id"]),
        text=text,
        obfuscation=kind,
        proportion=proportion,
        spans=[[start, start + len(passage)]],
        source_texts=[source["text"]],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    seed_everything(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with timer("make_plag_benchmark"):
        rng = random.Random(42)
        human = load_split("test")
        human = human[human["label"] == 0]
        try:
            ood = load_split("test_ood")
            formal = ood[ood["domain"] == "formal_academic"]
        except FileNotFoundError:
            formal = human.iloc[0:0]
        pool = human.to_dict("records") + formal.to_dict("records")
        rng.shuffle(pool)
        host_count = 60 if cfg.smoke else HOST_COUNT
        negative_count = 20 if cfg.smoke else NEGATIVE_COUNT
        if len(pool) < 40:
            log.warning("not enough held-out human text to build the benchmark (%s)", len(pool))
            return 0
        sources, hosts = pool[len(pool) // 2 :], pool[: len(pool) // 2]
        paraphraser = _paraphraser()

        cases: list[Case] = []
        skipped: list[str] = []
        index = 0
        for host in hosts:
            if len(cases) >= host_count:
                break
            for kind in OBFUSCATIONS:
                if kind == "heavy_paraphrase" and paraphraser is None:
                    if kind not in skipped:
                        skipped.append(kind)
                    continue
                for proportion in PROPORTIONS:
                    source = sources[index % len(sources)]
                    index += 1
                    case = _build_case(host, source, kind, proportion, paraphraser)
                    if case is not None:
                        cases.append(case)
        for position in range(negative_count):
            host = hosts[position % len(hosts)]
            cases.append(
                Case(
                    id=f"neg-{position}",
                    host_id=str(host["id"]),
                    text=str(host["text"]),
                    obfuscation="none",
                    proportion=0.0,
                    is_negative=True,
                )
            )
        rng.shuffle(cases)
        half = len(cases) // 2
        payload = {
            "synthetic": True,
            "caption": (
                "Synthetic benchmark: passages were inserted into held-out documents with "
                "known character spans. It measures the engine's ability to recover spans, "
                "not real-world plagiarism rates."
            ),
            "skipped_types": skipped,
            "obfuscations": [k for k in OBFUSCATIONS if k not in skipped],
            "proportions": list(PROPORTIONS),
            "validation": [asdict(c) for c in cases[:half]],
            "test": [asdict(c) for c in cases[half:]],
        }
        write_json(OUT_DIR / "benchmark.json", payload)
        log.info(
            "benchmark: %s validation + %s test cases (skipped %s)", half, len(cases) - half, skipped
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
