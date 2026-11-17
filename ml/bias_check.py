"""False-positive-rate bias check on an ESL-style proxy.

Detectors are known to over-flag non-native English.  We cannot obtain a real
non-native corpus here, so this script builds an explicitly labelled **proxy**: human
texts simplified to the 3,000 most frequent words with shorter sentences, plus any real
essays committed under ``data/eval/esl/*.txt``.

``higher_false_positive_rate_on_proxy`` is set to true **only** when the bootstrap 95%
confidence interval of the difference excludes zero — which is exactly the defect the
previous build had (an equal-rate result reported as "higher").
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np

from common import ARTIFACT_DIR, DATA_DIR, REPO_ROOT, RunConfig, log, seed_everything, timer, write_json
from featurestore import load_split

sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.config import MODE_COMPONENTS, MODES  # noqa: E402
from app.detector.classical import load as load_classical  # noqa: E402
from app.detector.perplexity import load as load_ppl  # noqa: E402
from app.detector.calibration import ModeThresholds  # noqa: E402
from app.detector.stacker import feature_row, load as load_stacker  # noqa: E402
from common import read_json  # noqa: E402

MAX_TEXTS = 400
_WORD = re.compile(r"[A-Za-z']+")
_SENTENCE = re.compile(r"[.!?]+")


def _frequent_words(reference_texts: list[str], size: int = 3000) -> set[str]:
    counts: dict[str, int] = {}
    for text in reference_texts:
        for word in _WORD.findall(text.lower()):
            counts[word] = counts.get(word, 0) + 1
    return set(sorted(counts, key=counts.get, reverse=True)[:size])


def simplify(text: str, vocabulary: set[str]) -> str:
    """ESL-style proxy: keep frequent words, split long sentences, drop rare words."""
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    out: list[str] = []
    for sentence in sentences:
        # Break long sentences at common connectors, which is what simpler writing does.
        pieces = re.split(r",\s*(?:and|but|which|while|because|although)\s+", sentence)
        for piece in pieces:
            words = [w for w in piece.split() if w.lower().strip(".,;:!?\"'") in vocabulary
                     or not w.isalpha()]
            if len(words) >= 5:
                out.append(" ".join(words).strip() + ".")
    return " ".join(out)


def _fpr(scores: np.ndarray, t_ai: float) -> float:
    return float(np.mean(scores >= t_ai)) if scores.size else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    seed_everything()

    with timer("bias_check"):
        test = load_split("test")
        human = test[test["label"] == 0].head(MAX_TEXTS)
        if human.empty:
            log.warning("no human test rows; bias check skipped")
            return 0
        vocabulary = _frequent_words(test["text"].tolist())
        proxy_texts = [simplify(text, vocabulary) for text in human["text"].tolist()]
        kept = [(original, proxied) for original, proxied in zip(human["text"].tolist(), proxy_texts)
                if len(proxied.split()) >= 30]
        log.info("ESL proxy built for %s of %s human texts", len(kept), len(human))

        real_essays: list[str] = []
        essay_dir = DATA_DIR / "eval" / "esl"
        if essay_dir.exists():
            real_essays = [p.read_text(errors="replace") for p in sorted(essay_dir.glob("*.txt"))]
            log.info("real ESL essays found: %s", len(real_essays))

        thresholds_raw = read_json(ARTIFACT_DIR / "thresholds.json", {}) or {}
        classical = load_classical(ARTIFACT_DIR / "classical.pkl")
        ppl = load_ppl(ARTIFACT_DIR / "ppl_gpt2_model.pkl")

        output: dict[str, object] = {
            "run": cfg.label,
            "proxy_definition": (
                "human test texts simplified to the 3,000 most frequent words of the corpus, "
                "with long sentences split at common connectors. This is a PROXY for "
                "non-native writing, not real ESL data."
            ),
            "modes": {},
            "notes": [],
        }
        if real_essays:
            output["real_esl_texts"] = len(real_essays)
        for mode in MODES:
            if mode not in thresholds_raw:
                continue
            if MODE_COMPONENTS[mode] not in {("classical",), ("classical", "ppl_gpt2")}:
                output["notes"].append(
                    f"mode {mode}: bias check needs the neural components, which are not "
                    "available in this environment"
                )
                continue
            stacker = load_stacker(ARTIFACT_DIR / f"stacker_{mode}.pkl")
            t_ai = ModeThresholds.from_dict(thresholds_raw[mode]).t_ai

            def score(text: str) -> float | None:
                probs: dict[str, float | None] = {"classical": None, "ppl_gpt2": None, "transformer": None}
                try:
                    if "classical" in MODE_COMPONENTS[mode]:
                        probs["classical"] = classical.predict_proba([text])[0]
                    if "ppl_gpt2" in MODE_COMPONENTS[mode]:
                        probs["ppl_gpt2"] = ppl.predict_proba([text])[0]
                except Exception:  # noqa: BLE001
                    return None
                row = feature_row(probs, {"binoculars": 0.0, "fastdetect": 0.0},
                                  len(text.split()), mode)
                try:
                    return float(stacker.probability(row))
                except Exception:  # noqa: BLE001
                    return None

            normal_scores = np.asarray([s for s in (score(t) for t in human["text"].tolist()) if s is not None])
            proxy_scores = np.asarray([s for s in (score(t) for t in proxy_texts) if s is not None])
            if normal_scores.size < 30 or proxy_scores.size < 30:
                continue
            normal_fpr = _fpr(normal_scores, t_ai)
            proxy_fpr = _fpr(proxy_scores, t_ai)

            rng = np.random.RandomState(42)
            differences = []
            for _ in range(1000):
                normal_sample = rng.choice(normal_scores, normal_scores.size)
                proxy_sample = rng.choice(proxy_scores, proxy_scores.size)
                differences.append(_fpr(proxy_sample, t_ai) - _fpr(normal_sample, t_ai))
            low, high = np.percentile(differences, [2.5, 97.5])
            flagged = bool(high < 0 or low > 0)
            output["modes"][mode] = {
                "t_ai": t_ai,
                "normal_human_fpr": round(normal_fpr, 4),
                "proxy_human_fpr": round(proxy_fpr, 4),
                "difference": round(proxy_fpr - normal_fpr, 4),
                "difference_ci95": [round(float(low), 4), round(float(high), 4)],
                "higher_false_positive_rate_on_proxy": flagged,
                "normal_n": int(normal_scores.size),
                "proxy_n": int(proxy_scores.size),
            }
            if real_essays:
                essay_scores = np.asarray([s for s in (score(t) for t in real_essays) if s is not None])
                if essay_scores.size >= 10:
                    output["modes"][mode]["real_esl_fpr"] = round(_fpr(essay_scores, t_ai), 4)
            log.info(
                "%s: normal FPR %.3f proxy FPR %.3f CI [%.3f, %.3f] flagged=%s",
                mode, normal_fpr, proxy_fpr, low, high, flagged,
            )
        output["notes"].append(
            "Flagged only when the 95% interval of the difference excludes zero."
        )
        write_json(ARTIFACT_DIR / "bias.json", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
