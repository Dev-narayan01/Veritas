"""Evaluate the plagiarism engine **on the test half** of the synthetic benchmark.

Reports character-level span precision/recall/F1 overall and per obfuscation type, the
correlation between ``similarity_percent`` and the true inserted fraction, and the
false-alarm rate on negatives.  The thresholds come from ``plag_thresholds.json``
(tuned on validation); this script never tunes anything.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from common import ARTIFACT_DIR, REPO_ROOT, RunConfig, log, read_json, seed_everything, timer, write_json
from tune_plag_thresholds import _span_f1, evaluate

sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.plagiarism.corpus import PlagiarismCorpus  # noqa: E402

BENCH_DIR = ARTIFACT_DIR / "plag_benchmark"


def _similarity(text: str, source: str, params: dict) -> float:
    from app.plagiarism.align import align

    matches = align(text, source, "source", min_words=params["min_match_words"],
                    min_containment=params["lexical_min_containment"])
    if not matches:
        return 0.0
    covered: set[int] = set()
    for match in matches:
        covered.update(range(match.query_start, match.query_end))
    return round(100.0 * len(covered) / max(1, len(text)), 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    seed_everything()

    benchmark = read_json(BENCH_DIR / "benchmark.json", None)
    thresholds = read_json(ARTIFACT_DIR / "plag_thresholds.json", None)
    if not benchmark or not thresholds:
        log.warning("benchmark or plagiarism thresholds missing; evaluation skipped")
        return 0

    with timer("eval_plagiarism"):
        params = thresholds["selected"]
        cases = benchmark["test"]
        corpus = PlagiarismCorpus(ARTIFACT_DIR / "corpus")
        corpus.load()
        cases = cases[:120] if cfg.smoke else cases

        per_type: dict[str, list[float]] = {}
        predicted_fractions: list[float] = []
        true_fractions: list[float] = []
        false_alarms = 0
        negatives = 0
        for case in cases:
            if case.get("is_negative"):
                negatives += 1
                matches = evaluate([case], corpus, params)
                if matches["false_alarm_rate"] > 0:
                    false_alarms += 1
                continue
            from app.plagiarism.align import align

            matches = align(case["text"], case["source_texts"][0], case["id"],
                            min_words=params["min_match_words"],
                            min_containment=params["lexical_min_containment"])
            precision, recall, f1 = _span_f1([(m.query_start, m.query_end) for m in matches], case["spans"])
            per_type.setdefault(case["obfuscation"], []).append(f1)
            predicted_fractions.append(_similarity(case["text"], case["source_texts"][0], params))
            true_fractions.append(case["proportion"] * 100)

        overall_f1 = float(np.mean([f for scores in per_type.values() for f in scores])) if per_type else 0.0
        payload = {
            "synthetic_benchmark": True,
            "caption": "Synthetic benchmark with known inserted spans; not a real-world rate.",
            "thresholds_used": params,
            "cases": len(cases),
            "span_f1_overall": round(overall_f1, 4),
            "span_f1_by_obfuscation": {k: round(float(np.mean(v)), 4) for k, v in per_type.items()},
            "false_alarm_rate_on_negatives": round(false_alarms / negatives, 4) if negatives else 0.0,
            "target_span_f1": 0.70,
            "target_reached": overall_f1 >= 0.70,
            "correlation_similarity_vs_true_fraction": round(
                float(np.corrcoef(predicted_fractions, true_fractions)[0, 1]), 4
            ) if len(set(true_fractions)) > 1 else None,
            "skipped_types": benchmark.get("skipped_types", []),
            "notes": [],
        }
        if not payload["target_reached"]:
            payload["notes"].append(
                "Span F1 is below the 0.70 target; this is reported honestly rather than tuned "
                "on the test half."
            )
        if payload["skipped_types"]:
            payload["notes"].append(
                "Obfuscation types skipped because the model was unavailable: "
                + ", ".join(payload["skipped_types"])
            )
        write_json(ARTIFACT_DIR / "plag_benchmark_results.json", payload)
        log.info("plagiarism benchmark: %s", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
