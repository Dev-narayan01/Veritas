"""Robustness of each mode to evasion attempts.

For AI-labelled ``test`` texts, recall is measured before and after each perturbation
at that mode's tuned ``t_ai``.  Two things make this trustworthy:

* ``changed_fraction`` records how many texts the perturbation actually altered, so a
  perturbation that silently did nothing can never produce a flattering number (the
  bug in the previous build);
* the homoglyph and invisible-character attacks are evaluated **with and without** the
  integrity layer, which is the only way to show the layer does something.

Three before/after examples per perturbation are saved for inspection.  A paraphrase
attack is attempted with a T5 paraphraser when one is available, and reported as
skipped otherwise.
"""

from __future__ import annotations

import argparse
import sys


from common import (
    ARTIFACT_DIR,
    PERTURBATIONS,
    REPO_ROOT,
    RunConfig,
    log,
    read_json,
    seed_everything,
    timer,
    write_json,
)

sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.config import MODE_COMPONENTS, MODES  # noqa: E402
from app.integrity import engine as integrity_engine  # noqa: E402
from app.detector.calibration import ModeThresholds  # noqa: E402
from app.detector.classical import load as load_classical  # noqa: E402
from app.detector.perplexity import load as load_ppl  # noqa: E402
from app.detector.stacker import feature_row, load as load_stacker  # noqa: E402
from featurestore import load_split  # noqa: E402

MAX_TEXTS = 200
EXAMPLES = 3


def _mode_probability(mode: str, text: str, classical, ppl, stacker) -> float | None:
    """Score one perturbed text with the components a mode declares."""
    probs: dict[str, float | None] = {"classical": None, "ppl_gpt2": None, "transformer": None}
    try:
        if "classical" in MODE_COMPONENTS[mode]:
            probs["classical"] = classical.predict_proba([text])[0]
        if "ppl_gpt2" in MODE_COMPONENTS[mode]:
            probs["ppl_gpt2"] = ppl.predict_proba([text])[0]
    except Exception as exc:  # noqa: BLE001 - a component failure means "cannot score"
        log.debug("scoring failed: %s", exc)
        return None
    if "transformer" in MODE_COMPONENTS[mode]:
        return None  # transformer robustness is covered by evaluate.py's OOD numbers
    if "binoculars" in MODE_COMPONENTS[mode] or "fastdetect" in MODE_COMPONENTS[mode]:
        return None
    zero_shot = {"binoculars": 0.0, "fastdetect": 0.0}
    row = feature_row(probs, zero_shot, len(text.split()), mode)
    try:
        return float(stacker.probability(row))
    except Exception:  # noqa: BLE001
        return None


def _paraphrase_attack(texts: list[str]) -> tuple[list[str] | None, str]:
    """Try to paraphrase with a small T5 model; report why not when unavailable."""
    try:  # pragma: no cover - needs torch + weights
        import torch
        from transformers import pipeline

        paraphraser = pipeline(
            "text2text-generation",
            model="humarin/chatgpt_paraphraser_on_T5-base",
            device=0 if torch.cuda.is_available() else -1,
        )
        outputs = [
            paraphraser(f"paraphrase: {text[:400]}", max_length=200, do_sample=False)[0][
                "generated_text"
            ]
            for text in texts
        ]
        return outputs, ""
    except Exception as exc:  # noqa: BLE001 - optional model
        return None, f"paraphrase attack skipped: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    seed_everything()

    with timer("robustness"):
        frame = load_split("test")
        ai_rows = frame[frame["label"] == 1].head(MAX_TEXTS)
        human_rows = frame[frame["label"] == 0].head(MAX_TEXTS)
        if ai_rows.empty:
            log.warning("no AI-labelled test rows; robustness cannot be measured")
            return 0
        thresholds_raw = read_json(ARTIFACT_DIR / "thresholds.json", {}) or {}
        classical = load_classical(ARTIFACT_DIR / "classical.pkl")
        ppl = load_ppl(ARTIFACT_DIR / "ppl_gpt2_model.pkl")

        output: dict[str, object] = {
            "run": cfg.label,
            "texts": len(ai_rows),
            "modes": {},
            "notes": [],
        }
        human_texts = human_rows["text"].tolist()
        for mode in MODES:
            if mode not in thresholds_raw:
                continue
            if MODE_COMPONENTS[mode] != ("classical",) and MODE_COMPONENTS[mode] != (
                "classical", "ppl_gpt2",
            ):
                output["notes"].append(
                    f"mode {mode} robustness not measured here: its neural components are "
                    "evaluated on the held-out split by evaluate.py"
                )
                continue
            stacker = load_stacker(ARTIFACT_DIR / f"stacker_{mode}.pkl")
            mode_thresholds = ModeThresholds.from_dict(thresholds_raw[mode])
            t_ai = mode_thresholds.t_ai

            baselines = [
                _mode_probability(mode, text, classical, ppl, stacker)
                for text in ai_rows["text"].tolist()
            ]
            baseline_hits = [p for p in baselines if p is not None and p >= t_ai]
            base_recall = len(baseline_hits) / max(1, len([p for p in baselines if p is not None]))
            entries: dict[str, dict] = {}
            for name, perturbation in PERTURBATIONS.items():
                perturbed: list[str] = []
                for index, text in enumerate(ai_rows["text"].tolist()):
                    human_reference = human_texts[index % max(1, len(human_texts))] if human_texts else ""
                    perturbed.append(perturbation(text, human_reference))
                changed = sum(1 for a, b in zip(ai_rows["text"].tolist(), perturbed) if a != b)
                changed_fraction = round(changed / len(perturbed), 4)

                def _recall(texts: list[str]) -> tuple[float, int]:
                    scored = [
                        p for p in (_mode_probability(mode, t, classical, ppl, stacker) for t in texts)
                        if p is not None
                    ]
                    hits = sum(1 for p in scored if p >= t_ai)
                    return (hits / max(1, len(scored))), len(scored)

                recall_after, scored = _recall(perturbed)
                entry: dict[str, object] = {
                    "recall_before": round(base_recall, 4),
                    "recall_after": round(recall_after, 4),
                    "changed_fraction": changed_fraction,
                    "scored": scored,
                    "examples": [
                        {"before": a[:300], "after": b[:300]}
                        for a, b in list(zip(ai_rows["text"].tolist(), perturbed))[:EXAMPLES]
                    ],
                }
                if name in {"homoglyph", "invisible_chars"}:
                    cleaned = [integrity_engine.run(t).text for t in perturbed]
                    cleaned_recall, _ = _recall(cleaned)
                    entry["recall_after_integrity_cleaning"] = round(cleaned_recall, 4)
                entries[name] = entry
                log.info(
                    "%s/%s: recall %.3f -> %.3f (changed %.0f%%)",
                    mode, name, base_recall, recall_after, changed_fraction * 100,
                )
            paraphrased, note = _paraphrase_attack(ai_rows["text"].tolist()[:50])
            if paraphrased:
                entry = {
                    "recall_after": round(
                        sum(
                            1
                            for p in (
                                _mode_probability(mode, t, classical, ppl, stacker)
                                for t in paraphrased
                            )
                            if p is not None and p >= t_ai
                        )
                        / max(1, len(paraphrased)),
                        4,
                    ),
                    "changed_fraction": round(
                        sum(1 for a, b in zip(ai_rows["text"].tolist()[:50], paraphrased) if a != b)
                        / max(1, len(paraphrased)),
                        4,
                    ),
                }
                entries["paraphrase"] = entry
            else:
                output["notes"].append(note)
            output["modes"][mode] = {"t_ai": t_ai, "baseline_recall": round(base_recall, 4),
                                     "perturbations": entries}
        write_json(ARTIFACT_DIR / "robustness.json", output)
        log.info("wrote robustness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
