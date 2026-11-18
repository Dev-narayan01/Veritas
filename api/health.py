"""GENERATED FILE - do not edit.

Built by ``python scripts/build_api.py`` from ``api/_lib/core.py`` plus
``api/handlers/health.py``. Vercel ships a function as this one file, so the shared core is
inlined rather than imported. Edit the sources and re-run the script.
"""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
import json
import math
import os
import re
import sys
import traceback
import types

VERSION = "1.0.0"
DISCLAIMER = (
    "AI-text detection is probabilistic. Veritas reports a calibrated probability, the "
    "components that actually ran and the thresholds in force; it must never be the sole "
    "basis for an accusation."
)
LITE_LIMITATIONS = [
    "Lite mode analyses pasted text only: no PDF, DOCX, PPTX, OCR, URL or ZIP parsing.",
    "Plagiarism, batch comparison and document reports run on the full server, not here.",
    "Only the classical stylometric detector runs in Lite; the perplexity and zero-shot components need the full server.",
    "Sentence heatmaps are not available in Lite, so no highlight can be drawn.",
]
LITE_COMPONENTS = ("classical",)
MIN_WORDS = 60
ARTIFACT_CANDIDATES = (
    "api/artifacts",  # committed Lite bundle: the only one a git deploy can ship
    "artifacts",  # resolve relative to this file's own directory (api/artifacts)
    "frontend/public/artifacts",  # staged by scripts/deploy_vercel.sh
    "ml/artifacts",  # a local training run
)


class _Classical:
    """The two attributes of ``app.detector.classical.ClassicalModel`` that Lite needs.

    joblib pickles the instance by module path, and the ``app`` package is not bundled
    into the Vercel functions, so the artifact is loaded as a plain object and duck-typed
    against this tiny proxy instead. No application code is imported.
    """

    def __init__(self, vectorizer: Any, classifier: Any) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict_proba(self, texts: list[str]) -> Any:
        matrix = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(matrix)[:, 1]


class _PplBundle:
    """Perplexity component, torch-free, matching ``app.detector.perplexity``.

    Only the trigram path is implemented: the GPT-2 backend needs weights that a
    serverless bundle cannot carry, and the artifact was trained with ``backend=ngram_lm``
    regardless. The feature computation below is a line-for-line port, and
    ``scripts/verify_fast_mode.py`` compares it against the original module on real rows.

    Definable here rather than imported, because a function ships as one file.
    """

    def __init__(
        self,
        classifier: Any = None,
        feature_names: list[str] | None = None,
        style_reference: dict[str, Any] | None = None,
        order: int = 3,
        vocab_size: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self.classifier = classifier
        self.feature_names = list(feature_names or [])
        self.style_reference = dict(style_reference or {})
        self.order = order
        self.vocab_size = vocab_size
        self.total_tokens = total_tokens
        self.counts: dict[tuple[str, ...], dict[str, int]] = {}
        self.unigram: dict[str, int] = {}

    # ---- language model -------------------------------------------------
    def attach_counts(self, counts: dict[tuple[str, ...], dict[str, int]]) -> None:
        self.counts = counts
        self.unigram = counts.get((), {})

    def _counter(self, context: tuple[str, ...]) -> dict[str, int]:
        for n in range(len(context), 0, -1):
            counter = self.counts.get(tuple(context[-n:]))
            if counter:
                return counter
        return {}

    def distribution(self, context: tuple[str, ...]) -> dict[str, float]:
        counter = self._counter(context)
        total = sum(counter.values())
        if not counter:
            return {"<unk>": 1.0}
        alpha = 0.1
        denom = total + alpha * self.vocab_size
        top = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:200]
        return {token: (count + alpha) / denom for token, count in top}

    def token_logprob(self, token: str, context: tuple[str, ...]) -> float:
        counter = self._counter(context)
        if counter:
            total = sum(counter.values())
            if counter.get(token):
                return math.log(counter[token] / total)
        floor = 1.0 / (max(1, self.total_tokens) * 100)
        prob = (self.unigram.get(token, 0) / max(1, self.total_tokens)) if self.unigram else floor
        return math.log(max(prob, floor) * 0.4)

    def rank_percentile(self, token: str, context: tuple[str, ...]) -> float:
        ordered = sorted(self.distribution(context).items(), key=lambda kv: kv[1], reverse=True)
        for rank, (candidate, _) in enumerate(ordered):
            if candidate == token:
                return rank / max(1, len(ordered))
        return 1.0

    # ---- features -------------------------------------------------------
    def features(self, text: str) -> dict[str, float]:
        feats = _style_features(text)
        feats.update(self._ngram_features(text))
        return feats

    def _ngram_features(self, text: str, max_tokens: int = 512) -> dict[str, float]:
        tokens = text.split()[:max_tokens]
        if len(tokens) < 2:
            return _summarise_lm([], [], [])
        logprobs: list[float] = []
        ranks: list[float] = []
        entropies: list[float] = []
        history: list[str] = []
        for token in tokens:
            context = tuple(history[-(self.order - 1) :]) if self.order > 1 else ()
            logprobs.append(self.token_logprob(token, context))
            ranks.append(self.rank_percentile(token, context))
            entropies.append(-sum(p * math.log(p) for p in self.distribution(context).values() if p > 0))
            history.append(token)
        return _summarise_lm(logprobs, entropies, ranks)

    def predict_proba(self, texts: list[str]) -> Any:
        import numpy as np

        rows = []
        for text in texts:
            feats = self.features(text)
            rows.append([float(feats.get(name, 0.0)) for name in self.feature_names])
        return self.classifier.predict_proba(np.asarray(rows, dtype=np.float64))[:, 1]


_STOPWORDS = frozenset(
    """a about above after again against all am an and any are aren't as at be because
    been before being below between both but by can't cannot could couldn't did didn't
    do does doesn't doing don't down during each few for from further had hadn't has
    hasn't have haven't having he he'd he'll he's her here here's hers herself him
    himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
    let's me more most mustn't my myself no nor not of off on once only or other ought
    our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
    so some such than that that's the their theirs them themselves then there there's
    these they they'd they'll they're they've this those through to too under until up
    very was wasn't we we'd we'll we're we've were weren't what what's when when's where
    where's which while who who's whom why why's with won't would wouldn't you you'd
    you'll you're you've your yours yourself yourselves""".split()
)
_CONTRACTION_RE = re.compile(r"\b\w+(?:'|’)t\b|\b\w+'(?:s|re|ve|ll|d|m)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z']+")
_FUNCTION_PUNCT = set(".,;:!?'\"()[]-")


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


_SENTENCE_END = re.compile(r"""(?<=[.!?])["')\]]*\s+(?=["'(\[]?[A-Z0-9])""")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "fig", "eq", "no", "vol", "pp", "al", "inc", "ltd", "co", "approx", "dept",
}


def _split_sentences_like_training(text: str) -> list[str]:
    """Port of ``app.text.segment._regex_split``.

    The style features are counts over sentences, so the splitter has to agree with the
    one used at training time. NLTK's Punkt is unavailable in a serverless bundle, which
    is fine: the training run fell back to this same regex path (the pip environment had
    no ``punkt`` either), so the two match exactly. Abbreviation guards matter - without
    them "Dr. Smith" becomes two sentences and ``sentence_len_mean`` drifts.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        candidate = text[start : match.start()]
        tail = candidate.rstrip().rsplit(" ", 1)[-1].rstrip(".").lower()
        if tail in _ABBREVIATIONS or (len(tail) == 1 and tail.isalpha()):
            continue
        spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [piece for piece in (text[a:b].strip() for a, b in spans) if piece]


def _style_features(text: str) -> dict[str, float]:
    """Port of ``app.detector.features.compute_style`` (ten interpretable measures)."""
    sentences = _split_sentences_like_training(text) or []
    words = _WORD_RE.findall(text.lower())
    total = max(1, len(words))
    lengths = [len(sentence.split()) for sentence in sentences] or [total]
    mean_len = _safe_div(sum(lengths), len(lengths))
    std_len = math.sqrt(_safe_div(sum((x - mean_len) ** 2 for x in lengths), len(lengths)))
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    repeated = sum(c for w, c in counts.items() if c > 2 and w not in _STOPWORDS)
    return {
        "sentence_len_mean": mean_len,
        "sentence_len_std": std_len,
        "word_len_mean": _safe_div(sum(len(w) for w in words), total),
        "type_token_ratio": _safe_div(len(counts), total),
        "punctuation_rate": _safe_div(sum(1 for c in text if c in _FUNCTION_PUNCT), max(1, len(text))),
        "stopword_ratio": _safe_div(sum(1 for w in words if w in _STOPWORDS), total),
        "repetition_rate": _safe_div(repeated, total),
        "contraction_rate": _safe_div(len(_CONTRACTION_RE.findall(text)), total),
        "long_word_ratio": _safe_div(sum(1 for w in words if len(w) >= 8), total),
        "sentence_count": float(len(sentences)),
    }


def _summarise_lm(logprobs: list[float], entropies: list[float], ranks: list[float]) -> dict[str, float]:
    """Port of ``_summarise_lm``; the same six summary features."""
    if not logprobs:
        return {
            "lm_mean_logprob": 0.0,
            "lm_std_logprob": 0.0,
            "burstiness": 0.0,
            "mean_rank_pct": 0.5,
            "top10_hit_rate": 0.0,
            "entropy_mean": 0.0,
        }
    mean = sum(logprobs) / len(logprobs)
    var = sum((x - mean) ** 2 for x in logprobs) / len(logprobs)
    diffs = [logprobs[i + 1] - logprobs[i] for i in range(len(logprobs) - 1)]
    d_mean = sum(diffs) / len(diffs) if diffs else 0.0
    d_var = sum((d - d_mean) ** 2 for d in diffs) / len(diffs) if diffs else 0.0
    return {
        "lm_mean_logprob": mean,
        "lm_std_logprob": math.sqrt(var),
        "burstiness": math.sqrt(d_var),
        "mean_rank_pct": sum(ranks) / len(ranks),
        "top10_hit_rate": sum(1 for r in ranks if r <= 0.10) / len(ranks),
        "entropy_mean": sum(entropies) / len(entropies),
    }


@dataclass
class _PlattScaler:
    """Duck-typed ``app.detector.calibration.PlattScaler`` (two floats)."""

    a: float = 1.0
    b: float = 0.0

    def transform(self, decision: Any) -> Any:
        import numpy as np

        z = self.a * np.asarray(decision, dtype=float) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


@dataclass
class _ModeThresholds:
    """Duck-typed ``app.detector.calibration.ModeThresholds``.

    Only the fields the runtime reads are declared; anything else the pickle carries is
    ignored, which keeps this stable across artifact versions.
    """

    t_ai: float = 1.0
    t_human: float = 0.0
    target_fpr: float = 0.01
    band_collapsed: bool = False
    per_bucket: dict[str, dict[str, float]] | None = None
    achieved: dict[str, float] | None = None


class _Stacker:
    """Duck-typed stand-in for ``app.detector.stacker.StackerArtifact``.

    Pickle restores ``mode``, ``feature_names``, ``classifier``, ``scaler_mean``,
    ``scaler_std``, ``platt`` and ``thresholds`` onto this class, so the exact fitted
    logistic regression and Platt scaler trained by ``ml/train_stackers.py`` run here.
    Nothing about the calibration is reimplemented or approximated.
    """

    def __init__(self) -> None:
        self.mode = "lite"
        self.feature_names: list[str] = []
        self.classifier: Any = None
        self.scaler_mean: dict[str, float] = {}
        self.scaler_std: dict[str, float] = {}
        self.platt: Any = None
        self.thresholds: Any = None
        self.components: list[str] = []

    def vector(self, features: dict[str, float]) -> list[float]:
        out = []
        for name in self.feature_names:
            value = float(features.get(name, 0.0))
            if name in self.scaler_mean:
                std = self.scaler_std.get(name) or 1.0
                value = (value - self.scaler_mean[name]) / std
            out.append(value)
        return out

    def probability(self, features: dict[str, float]) -> float:
        import numpy as np

        vec = np.asarray([self.vector(features)], dtype=float)
        if self.platt is None:
            return float(self.classifier.predict_proba(vec)[0, 1])
        decision = float(self.classifier.decision_function(vec)[0])
        return float(self.platt.transform(np.asarray([decision]))[0])


def _logit(probability: float) -> float:
    """Match ``app.detector.stacker.logit`` exactly, including its clamp."""
    p = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(p / (1.0 - p))


@dataclass
class LiteArtifacts:
    """Lite files, resolved once per cold start."""

    directory: Path | None
    model: Any = None
    thresholds: dict[str, Any] | None = None
    stacker: Any = None
    ppl: Any = None
    stacker_fast: Any = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.thresholds is not None

    @property
    def fast_available(self) -> bool:
        """``fast`` needs the perplexity component *and* its own fitted stacker."""
        return self.available and self.ppl is not None and self.stacker_fast is not None


def _install_pickle_stub() -> None:
    """Make ``classical.pkl`` loadable without the ``app`` package.

    ``joblib`` stores the class path (``app.detector.classical.ClassicalModel``). The
    Vercel bundle ships numpy/scikit-learn/joblib but not the FastAPI application, so
    that module does not exist. When the real package *is* importable (local dev, the
    test suite) it is used untouched; otherwise a three-module shim is registered that
    maps only ``ClassicalModel`` onto the duck-typed proxy above. Pickle restores the
    instance state directly, so no application logic is needed to read the artifact.
    """
    try:
        import app.detector.classical  # noqa: F401  (real backend available)

        return
    except Exception:  # noqa: BLE001 - expected on Vercel; any failure means "not bundled"
        pass

    calibration_module = types.ModuleType("app.detector.calibration")
    calibration_module.PlattScaler = _PlattScaler  # type: ignore[attr-defined]
    calibration_module.ModeThresholds = _ModeThresholds  # type: ignore[attr-defined]
    stacker_module = types.ModuleType("app.detector.stacker")
    stacker_module.StackerArtifact = _Stacker  # type: ignore[attr-defined]

    app_module = types.ModuleType("app")
    app_module.__path__ = []  # type: ignore[attr-defined]
    detector_module = types.ModuleType("app.detector")
    detector_module.__path__ = []  # type: ignore[attr-defined]
    classical_module = types.ModuleType("app.detector.classical")
    classical_module.ClassicalModel = _Classical  # type: ignore[attr-defined]
    sys.modules.setdefault("app.detector.calibration", calibration_module)
    sys.modules.setdefault("app.detector.stacker", stacker_module)
    app_module.detector = detector_module  # type: ignore[attr-defined]
    detector_module.classical = classical_module  # type: ignore[attr-defined]
    sys.modules.setdefault("app", app_module)
    sys.modules.setdefault("app.detector", detector_module)
    sys.modules.setdefault("app.detector.classical", classical_module)


def _search_roots() -> list[Path]:
    """Directories to try, most specific first.

    The core ships twice: as ``api/_lib/core.py`` (imported by tests and local tooling)
    and as the generated ``api/_core.py`` (bundled into the Vercel functions, two levels
    shallower). Deriving the root from ``__file__`` therefore has to cover both layouts,
    plus ``VERITAS_ARTIFACT_DIR`` for a bundle placed somewhere else entirely.
    """
    here = Path(__file__).resolve()
    roots = [here.parent, here.parent.parent, here.parent.parent.parent]
    override = os.environ.get("VERITAS_ARTIFACT_DIR")
    if override:
        roots.insert(0, Path(override))
    return roots


def find_artifact_dir() -> Path | None:
    """Locate a bundle containing the Lite files.

    ``VERITAS_ARTIFACT_DIR`` is an exact directory and wins outright: pointing it at a
    fresh training run must not be silently overridden by the committed bundle, which is
    what happens if it is merely prepended to a search list.
    """
    override = os.environ.get("VERITAS_ARTIFACT_DIR")
    if override:
        candidate = Path(override)
        if (candidate / "classical.pkl").exists() and (candidate / "thresholds.json").exists():
            return candidate
        return None
    for root in _search_roots():
        for relative in ARTIFACT_CANDIDATES:
            candidate = root / relative
            if (candidate / "classical.pkl").exists() and (candidate / "thresholds.json").exists():
                return candidate
    return None


def load_lite_artifacts() -> LiteArtifacts:
    """Load the classical detector and thresholds; never raise."""
    directory = find_artifact_dir()
    if directory is None:
        searched = ", ".join(str(root / rel) for root in _search_roots() for rel in ARTIFACT_CANDIDATES)
        return LiteArtifacts(None, error=f"no lite artifact bundle is deployed with this build (searched: {searched})")
    try:
        import joblib

        _install_pickle_stub()
        payload = joblib.load(directory / "classical.pkl")
        thresholds = json.loads((directory / "thresholds.json").read_text())
        vectorizer = getattr(payload, "vectorizer", None)
        classifier = getattr(payload, "classifier", None)
        if vectorizer is None or classifier is None:
            return LiteArtifacts(directory, error="classical.pkl does not contain a vectorizer/classifier pair")
        model = _Classical(vectorizer, classifier)
        stacker = _load_pickle(directory / "stacker_lite.pkl")
        stacker_fast = _load_pickle(directory / "stacker_fast.pkl")
        ppl = _load_ppl(directory)
    except Exception as exc:  # noqa: BLE001 - a corrupt/missing bundle is a runtime state
        return LiteArtifacts(directory, error=f"lite artifacts failed to load: {exc}")
    return LiteArtifacts(
        directory,
        model=model,
        thresholds=thresholds,
        stacker=stacker,
        ppl=ppl,
        stacker_fast=stacker_fast,
    )


def _load_pickle(path: Path) -> Any:
    """Load an optional artifact; a missing or corrupt one simply means "not available"."""
    if not path.exists():
        return None
    try:
        import joblib

        candidate = joblib.load(path)
        return candidate if getattr(candidate, "classifier", None) is not None else None
    except Exception:  # noqa: BLE001 - reported through the health payload instead
        return None


def _load_ppl(directory: Path) -> Any:
    """Load the perplexity component: pickled classifier + gzipped trigram counts.

    Kept separate from ``classical`` so a missing/corrupt ppl bundle degrades ``fast``
    alone and never takes ``lite`` down with it.
    """
    bundle_path = directory / "ppl_bundle.pkl"
    counts_path = directory / "ppl_lm.json.gz"
    if not bundle_path.exists() or not counts_path.exists():
        return None
    try:
        import gzip
        import pickle

        with open(bundle_path, "rb") as handle:
            stored = pickle.load(handle)
        with gzip.open(counts_path, "rb") as handle:
            payload = json.loads(handle.read())
        counts = {tuple(json.loads(key)): value for key, value in payload["counts"].items()}
        # The shipped bundle is a dict of primitives (see scripts/build_ppl_bundle.py):
        # pickling ``_PplBundle`` itself would embed ``_lib.core`` in the pickle, and the
        # deployed function is one generated file where ``_lib`` does not exist. A
        # ``_PplBundle`` instance is still accepted so older bundles keep loading.
        if isinstance(stored, _PplBundle):
            bundle = stored
        elif isinstance(stored, dict) and "classifier" in stored:
            bundle = _PplBundle(
                classifier=stored["classifier"],
                feature_names=list(stored.get("feature_names") or []),
                style_reference=dict(stored.get("style_reference") or {}),
                order=stored.get("order"),
                vocab_size=stored.get("vocab_size"),
                total_tokens=stored.get("total_tokens"),
            )
        else:
            return None
        bundle.attach_counts(counts)
        return bundle
    except Exception:  # noqa: BLE001
        return None


_ARTIFACTS: LiteArtifacts | None = None


def artifacts() -> LiteArtifacts:
    """Module-level cache so warm invocations skip the disk read."""
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = load_lite_artifacts()
    return _ARTIFACTS


def _mode_thresholds(thresholds: dict[str, Any] | None, mode: str) -> dict[str, Any]:
    """Read one mode's entry from ``thresholds.json``.

    The file written by ``ml/tune_thresholds.py`` is keyed by mode at the top level;
    ``modes`` is accepted too so an older bundle keeps working.
    """
    if not thresholds:
        return {}
    entry = thresholds.get(mode)
    if entry is None:
        entry = (thresholds.get("modes") or {}).get(mode)
    return entry if isinstance(entry, dict) else {}


def _lite_thresholds(thresholds: dict[str, Any] | None) -> dict[str, Any]:
    """The ``lite`` entry; kept as a named helper because most call sites want it."""
    return _mode_thresholds(thresholds, "lite")


def _is_fitted(entry: dict[str, Any]) -> bool:
    """A thresholds entry is usable when both bounds are present numbers."""
    def _num(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    return _num(entry.get("t_human")) and _num(entry.get("t_ai")) and entry["t_human"] <= entry["t_ai"]


def health() -> dict[str, Any]:
    """The same contract the full server serves, with Lite's honest availability."""
    bundle = artifacts()
    lite_thresholds = _lite_thresholds(bundle.thresholds)
    available = bundle.available and _is_fitted(lite_thresholds)
    reason = None
    if not bundle.available:
        reason = bundle.error
    elif not _is_fitted(lite_thresholds):
        reason = "lite thresholds were not fitted in the deployed bundle"
    modes = [
        {
            "mode": "lite",
            "available": available,
            "reason": reason,
            "components": list(LITE_COMPONENTS),
        },
        {
            "mode": "fast",
            "available": bundle.fast_available and _is_fitted(_mode_thresholds(bundle.thresholds, "fast")),
            "reason": None
            if bundle.fast_available
            else "Lite deployment: the perplexity component or its stacker is not bundled here",
            "components": ["classical", "ppl_gpt2"],
        },
        {
            "mode": "standard",
            "available": False,
            "reason": "Lite deployment: zero-shot components require the full server",
            "components": ["classical", "ppl_gpt2", "binoculars"],
        },
        {
            "mode": "deep",
            "available": False,
            "reason": "Lite deployment: zero-shot components require the full server",
            "components": ["classical", "fast_detectgpt"],
        },
    ]
    return {
        "status": "ok" if available else "degraded",
        "version": VERSION,
        "modes": modes,
        "models_loaded": {
            "classical": bundle.model is not None,
            "ppl_gpt2": bundle.ppl is not None,
            "transformer": False,
            "binoculars": False,
            "fast_detectgpt": False,
        },
        "corpus_size": 0,
        "ocr": False,
        "llm": False,
        "gpu": False,
        "artifact_source": "cpu" if bundle.model is not None else "none",
        "artifact_notes": [bundle.error]
        if bundle.error
        else [
            "lite bundle: classical"
            + (" + perplexity (ngram)" if bundle.ppl is not None else "")
            + (" + calibrated stacker" if bundle.stacker is not None else "")
            + " + thresholds"
        ],
        "data_source": "hc3",
        "default_mode": "lite",
        "disclaimer": DISCLAIMER,
        "deployment": "lite",
        "lite_limitations": LITE_LIMITATIONS,
    }


def split_sentences(text: str) -> list[str]:
    """Conservative sentence split (no NLTK in a serverless bundle)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


INVISIBLE = "\u200b\u200c\u200d\u2060\ufeff\u00ad"


def _integrity(text: str) -> dict[str, Any]:
    """Minimal integrity scan: invisible characters only (the full engine is on the server)."""
    found = sum(1 for char in text if char in INVISIBLE)
    flags = []
    if found:
        flags.append(
            {
                "kind": "invisible_characters",
                "count": found,
                "detail": "zero-width or soft-hyphen characters removed before scoring",
                "examples": ["U+%04X" % ord(char) for char in text if char in INVISIBLE][:5],
            }
        )
    return {"flags": flags, "cleaned_chars": found, "hidden_text_removed": 0}


def _padded(text: str, words: int) -> dict[str, Any]:
    """The body of a Lite result: one document, one component, an honest empty shell."""
    return {
        "id": "lite",
        "filename": "pasted-text.txt",
        "kind": "text",
        "words": words,
        "pages": None,
        "text": text,
        "warnings": list(LITE_LIMITATIONS),
        "integrity": _integrity(text),
        "ai": None,
        "plagiarism": None,
    }


def score(text: str, mode: str = "lite") -> dict[str, Any]:
    """Score one text and assemble a full result the UI can render.

    ``mode`` is ``lite`` (classical only) or ``fast`` (classical + perplexity). ``fast``
    degrades to ``lite`` when the perplexity bundle is absent, and says so in
    ``components_unavailable`` and ``mode_used`` - it never silently pretends to have run
    a component it does not have.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    words = len(cleaned.split())
    bundle = artifacts()
    thresholds = _lite_thresholds(bundle.thresholds)
    fitted = bundle.available and _is_fitted(thresholds)
    requested = "fast" if str(mode).lower() == "fast" else "lite"
    use_fast = requested == "fast" and bundle.fast_available
    effective = "fast" if use_fast else "lite"

    unavailable = [
        {"component": "binoculars", "reason": "Lite deployment: zero-shot components are not bundled here"},
        {"component": "fast_detectgpt", "reason": "Lite deployment: zero-shot components are not bundled here"},
    ]
    if not use_fast:
        unavailable.insert(
            0,
            {
                "component": "ppl_gpt2",
                "reason": (
                    "Lite deployment: the perplexity component is not bundled here"
                    if requested == "lite"
                    else "The perplexity bundle is missing, so 'fast' fell back to 'lite'"
                ),
            },
        )
    ai: dict[str, Any] = {
        "mode_requested": requested,
        "mode_used": effective,
        "components_run": [],
        "components_unavailable": unavailable,
        "probability": None,
        "ci_low": None,
        "ci_high": None,
        "verdict": "too_short",
        # Filled in below with the band that actually decides the verdict. It must not be
        # the lite band whenever ``fast`` ran: reporting thresholds that did not produce
        # the verdict is exactly the kind of dishonesty this deployment exists to avoid.
        "thresholds": {
            "t_human": thresholds.get("t_human"),
            "t_ai": thresholds.get("t_ai"),
        },
        "band_collapsed": False,
        "word_count": words,
        "low_confidence": True,
        "pass_count": 1,
        "components": [],
        "sentences": [],
        "tokens": [],
        "segments": [],
        "ai_fraction": None,
        "explanation": None,
        "warnings": [],
        "latency_ms": None,
    }

    if not fitted:
        ai["warnings"].append(bundle.error or "lite artifacts are unavailable on this deployment")
        return _envelope(_padded(cleaned, words), ai)
    if words < MIN_WORDS:
        ai["warnings"].append(
            f"Text is shorter than the {MIN_WORDS} words the lite detector was calibrated on, so no probability is reported."
        )
        return _envelope(_padded(cleaned, words), ai)

    # Prefer the fitted stacker + Platt calibration from training. Its thresholds live on
    # the artifact and were tuned on the same calibrated scores, so mixing them with a raw
    # component probability would make the bands meaningless.
    raw = float(bundle.model.predict_proba([cleaned])[0])
    ppl_probability = None
    if use_fast:
        ppl_probability = float(bundle.ppl.predict_proba([cleaned])[0])
        stacker = bundle.stacker_fast
    else:
        stacker = bundle.stacker
    if stacker is not None and stacker.thresholds is not None:
        t_human = float(stacker.thresholds.t_human)
        t_ai = float(stacker.thresholds.t_ai)
        if use_fast:
            # ``fast`` ships a band chosen on the OOD split, because the band tuner's
            # 1%-FPR target on val_cal transfers badly once the perplexity feature is
            # added. See scripts/tune_fast_operating_point.py; the artifact records both
            # the shipped and the rejected band so this cannot be "fixed" by accident.
            deployed = _mode_thresholds(bundle.thresholds, "fast_deployed")
            if _is_fitted(deployed):
                t_human = float(deployed["t_human"])
                t_ai = float(deployed["t_ai"])
        features = {"log_word_count": math.log(max(1, words)), "logit_classical": _logit(raw)}
        if ppl_probability is not None:
            features["logit_ppl_gpt2"] = _logit(ppl_probability)
        probability = float(stacker.probability(features))
    else:
        t_human = float(thresholds["t_human"])
        t_ai = float(thresholds["t_ai"])
        probability = raw
    calibrated = stacker is not None and stacker.thresholds is not None
    ai["thresholds"] = {"t_human": round(t_human, 6), "t_ai": round(t_ai, 6)}
    components_run = ["classical"] + (["ppl_gpt2"] if ppl_probability is not None else [])
    if probability <= t_human:
        verdict = "likely_human"
    elif probability >= t_ai:
        verdict = "likely_ai"
    else:
        verdict = "uncertain"

    # A width heuristic for the interval, widened by the binomial noise a short text has.
    width = 1.96 * math.sqrt(max(probability * (1 - probability), 1e-6) / max(words, 1))
    ai.update(
        {
            "components_run": components_run,
            "probability": round(probability, 6),
            "ci_low": round(max(0.0, probability - width), 6),
            "ci_high": round(min(1.0, probability + width), 6),
            "verdict": verdict,
            "band_collapsed": bool(getattr(stacker.thresholds, "band_collapsed", False))
            if stacker is not None and stacker.thresholds is not None
            else bool(thresholds.get("band_collapsed")),
            "low_confidence": words < 120,
            "components": _component_rows(raw, ppl_probability, probability, calibrated),
            "explanation": {
                "source": "rule_based",
                "text": (
                    (
                        "Fast mode adds a perplexity component (a trigram language model scored "
                        "against its own training counts) to the stylometric signal. The triggers for "
                        "an AI verdict are deliberately strict here, so this mode abstains far more "
                        "often than Lite: a mid-band score means the text does not clear the strict "
                        "band, not that a person wrote it."
                    )
                    if ppl_probability is not None
                    else (
                        "Lite mode runs the stylometric classifier only: word-frequency typicality, "
                        "sentence-length variation, punctuation rhythm and function-word rates. It cannot "
                        "see perplexity or zero-shot signals, so a mid-band score means the text looks "
                        "more or less like the training mixture, not that a person did or did not write it."
                    )
                ),
            },
        }
    )
    return _envelope(_padded(cleaned, words), ai)


def _component_rows(raw: float, ppl: float | None, stacked: float, calibrated: bool) -> list[dict[str, Any]]:
    """Per-component evidence, including the stacked result so nothing is misattributed.

    ``probability`` on the stacked row is the calibrated value; the raw component
    probabilities are kept beside it, because a reader must be able to see that the
    number they are shown is not simply ``classical``'s own output.
    """
    rows: list[dict[str, Any]] = [
        {
            "name": "classical",
            "probability": round(raw, 6),
            "raw": {"stacked": calibrated},
            "available": True,
            "reason": None,
        }
    ]
    if ppl is not None:
        rows.append(
            {
                "name": "ppl_gpt2",
                "probability": round(ppl, 6),
                "raw": {"backend": "ngram_lm"},
                "available": True,
                "reason": None,
            }
        )
    rows.append(
        {
            "name": "stacked" if calibrated else "ensemble",
            "probability": round(stacked, 6),
            "raw": {"calibrated": calibrated, "platt": calibrated},
            "available": True,
            "reason": None,
        }
    )
    return rows


def _envelope(document: dict[str, Any], ai: dict[str, Any]) -> dict[str, Any]:
    """Wrap one document into the analysis result shape the frontend expects."""
    document = dict(document)
    document["ai"] = ai
    return {
        "id": "lite",
        "created_at": _now(),
        "status": "done",
        "options": {
            "mode": "lite",
            "ai": True,
            "plagiarism": False,
            "batch": False,
            "explain": True,
            "online": False,
            "exclusions": {"quotes": True, "bibliography": True, "min_words": 8, "hide_below_percent": 0},
            "retain": False,
        },
        "documents": [document],
        "batch": None,
        "code_matches": [],
        "error": None,
        "exit_notes": list(LITE_LIMITATIONS),
        "disclaimer": DISCLAIMER,
        "deployment": "lite",
    }


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def history() -> list[dict[str, Any]]:
    """The Lite deployment keeps no history; the endpoint says so with an empty list."""
    return []

# --------------------------------------------------------------------------
# health endpoint
# --------------------------------------------------------------------------

def build_health() -> dict:
    try:
        return health()
    except Exception:  # noqa: BLE001 - any failure is a runtime state to report
        return {
            "status": "degraded",
            "version": VERSION,
            "modes": [],
            "models_loaded": {},
            "corpus_size": 0,
            "ocr": False,
            "llm": False,
            "gpu": False,
            "artifact_source": "none",
            "artifact_notes": ["lite core failed: " + traceback.format_exc(limit=3)],
            "data_source": "unknown",
            "default_mode": "lite",
            "disclaimer": DISCLAIMER,
            "deployment": "lite",
            "lite_limitations": LITE_LIMITATIONS,
        }


BODY = build_health()


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel's expected class name
    """Vercel's Python runtime looks for this class."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        payload = json.dumps(BODY).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("cache-control", "s-maxage=60")
        self.end_headers()
        self.wfile.write(payload)
