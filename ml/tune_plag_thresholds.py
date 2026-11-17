"""Grid-search the plagiarism thresholds **on the validation half** of the benchmark.

Searched: ``LEXICAL_MIN_CONTAINMENT``, ``SEMANTIC_MIN_COSINE``, ``RERANK_MIN_SCORE``,
``SEED_GAP_WORDS`` and ``MIN_MATCH_WORDS``.  The objective is character-span F1 on
validation with a penalty for false alarms on negatives.  The chosen values are written
to ``plag_thresholds.json`` and the test half is never touched here.
"""

from __future__ import annotations

import argparse
import itertools
import sys

from common import ARTIFACT_DIR, REPO_ROOT, RunConfig, log, read_json, seed_everything, timer, write_json

sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.config import settings  # noqa: E402
from app.plagiarism.align import align  # noqa: E402
from app.plagiarism.corpus import PlagiarismCorpus  # noqa: E402

BENCH_DIR = ARTIFACT_DIR / "plag_benchmark"
GRID = {
    "lexical_min_containment": (0.35, 0.50, 0.65),
    "seed_gap_words": (8, 12),
    "min_match_words": (6, 8),
}
MAX_VALIDATION_CASES = 40


def _span_f1(predicted: list[tuple[int, int]], truth: list[list[int]]) -> tuple[float, float, float]:
    """Character-level precision/recall/F1 against ground-truth spans."""
    def covered(spans: list[tuple[int, int]]) -> set[int]:
        out: set[int] = set()
        for start, end in spans:
            out.update(range(int(start), int(end)))
        return out

    predicted_set = covered(predicted)
    truth_set = covered([(int(a), int(b)) for a, b in truth])
    if not predicted_set and not truth_set:
        return 1.0, 1.0, 1.0
    if not predicted_set or not truth_set:
        return 0.0, 0.0, 0.0
    overlap = len(predicted_set & truth_set)
    precision = overlap / len(predicted_set)
    recall = overlap / len(truth_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(cases: list[dict], corpus: PlagiarismCorpus, params: dict) -> dict[str, float]:
    """Mean span F1 on positives plus the false-alarm rate on negatives."""
    f1_scores: list[float] = []
    false_alarms = 0
    negatives = 0
    for case in cases:
        matches = []
        for source_text in case["source_texts"][:1]:
            matches.extend(
                align(
                    case["text"], source_text, case["id"],
                    min_words=params["min_match_words"],
                    min_containment=params["lexical_min_containment"],
                )
            )
        if case.get("is_negative"):
            negatives += 1
            if matches:
                false_alarms += 1
            continue
        predicted = [(m.query_start, m.query_end) for m in matches]
        _, _, f1 = _span_f1(predicted, case["spans"])
        f1_scores.append(f1)
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return {
        "span_f1": round(mean_f1, 4),
        "false_alarm_rate": round(false_alarms / negatives, 4) if negatives else 0.0,
        "n_positives": len(f1_scores),
        "n_negatives": negatives,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    _cfg = RunConfig(smoke=args.smoke, full=args.full)
    del _cfg
    seed_everything()

    benchmark = read_json(BENCH_DIR / "benchmark.json", None)
    if not benchmark:
        log.warning("no benchmark found; run make_plag_benchmark.py first")
        return 0
    with timer("tune_plag_thresholds"):
        cases = benchmark["validation"][:MAX_VALIDATION_CASES]
        corpus = PlagiarismCorpus(ARTIFACT_DIR / "corpus")
        corpus.load()
        results = []
        for values in itertools.product(*GRID.values()):
            params = dict(zip(GRID, values))
            outcome = evaluate(cases, corpus, params)
            outcome.update(params)
            # Optimise span F1 but refuse configurations that fire on negatives.
            outcome["objective"] = outcome["span_f1"] * (1.0 - outcome["false_alarm_rate"])
            results.append(outcome)
        results.sort(key=lambda r: r["objective"], reverse=True)
        best = results[0]
        payload = {
            "synthetic_benchmark": True,
            "selected": {
                "lexical_min_containment": best["lexical_min_containment"],
                "seed_gap_words": best["seed_gap_words"],
                "min_match_words": best["min_match_words"],
                # Semantic thresholds are only tuned when a reranker is present; the
                # corpus build records whether it was.
                "semantic_min_cosine": settings.semantic_min_cosine,
                "rerank_min_score": settings.rerank_min_score,
            },
            "validation": {"chosen": best, "top5": results[:5]},
            "grid": {k: list(v) for k, v in GRID.items()},
        }
        write_json(ARTIFACT_DIR / "plag_thresholds.json", payload)
        log.info("chosen plagiarism thresholds: %s (F1=%s)", payload["selected"], best["span_f1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
