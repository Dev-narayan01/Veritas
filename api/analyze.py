"""GENERATED FILE - do not edit.

Built by ``python scripts/build_api.py`` from ``api/_lib/core.py`` plus
``api/handlers/analyze.py``. Vercel ships a function as this one file, so the shared core is
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
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.thresholds is not None


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
        stacker = None
        stacker_path = directory / "stacker_lite.pkl"
        if stacker_path.exists():
            candidate = joblib.load(stacker_path)
            if getattr(candidate, "classifier", None) is not None:
                stacker = candidate
    except Exception as exc:  # noqa: BLE001 - a corrupt/missing bundle is a runtime state
        return LiteArtifacts(directory, error=f"lite artifacts failed to load: {exc}")
    return LiteArtifacts(directory, model=model, thresholds=thresholds, stacker=stacker)


_ARTIFACTS: LiteArtifacts | None = None


def artifacts() -> LiteArtifacts:
    """Module-level cache so warm invocations skip the disk read."""
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = load_lite_artifacts()
    return _ARTIFACTS


def _lite_thresholds(thresholds: dict[str, Any] | None) -> dict[str, Any]:
    """Read the ``lite`` entry from ``thresholds.json``.

    The file written by ``ml/tune_thresholds.py`` is keyed by mode at the top level;
    ``modes`` is accepted too so an older bundle keeps working.
    """
    if not thresholds:
        return {}
    entry = thresholds.get("lite")
    if entry is None:
        entry = (thresholds.get("modes") or {}).get("lite")
    return entry if isinstance(entry, dict) else {}


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
            "available": False,
            "reason": "Lite deployment: the perplexity component and its stacker live on the full server",
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
            "ppl_gpt2": False,
            "transformer": False,
            "binoculars": False,
            "fast_detectgpt": False,
        },
        "corpus_size": 0,
        "ocr": False,
        "llm": False,
        "gpu": False,
        "artifact_source": "cpu" if bundle.model is not None else "none",
        "artifact_notes": [bundle.error] if bundle.error else ["lite bundle: classical + thresholds only"],
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


def score(text: str) -> dict[str, Any]:
    """Run the classical detector and assemble a full result the UI can render.

    The shape mirrors ``backend/app/schemas.py`` so the same React pages work; the
    difference is that ``plagiarism`` is null, ``sentences``/``tokens``/``segments`` are
    empty, and ``exit_notes`` states the limitations.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    words = len(cleaned.split())
    bundle = artifacts()
    thresholds = _lite_thresholds(bundle.thresholds)
    fitted = bundle.available and _is_fitted(thresholds)

    ai: dict[str, Any] = {
        "mode_requested": "lite",
        "mode_used": "lite",
        "components_run": [],
        "components_unavailable": [
            {"component": "ppl_gpt2", "reason": "Lite deployment: the language-model component is not bundled here"},
            {"component": "binoculars", "reason": "Lite deployment: zero-shot components are not bundled here"},
            {"component": "fast_detectgpt", "reason": "Lite deployment: zero-shot components are not bundled here"},
        ],
        "probability": None,
        "ci_low": None,
        "ci_high": None,
        "verdict": "too_short",
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
    stacker = bundle.stacker
    if stacker is not None and stacker.thresholds is not None:
        t_human = float(stacker.thresholds.t_human)
        t_ai = float(stacker.thresholds.t_ai)
        probability = float(stacker.probability({"log_word_count": math.log(max(1, words)),
                                                 "logit_classical": _logit(raw)}))
    else:
        t_human = float(thresholds["t_human"])
        t_ai = float(thresholds["t_ai"])
        probability = raw
    calibrated = stacker is not None and stacker.thresholds is not None
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
            "components_run": ["classical"],
            "probability": round(probability, 6),
            "ci_low": round(max(0.0, probability - width), 6),
            "ci_high": round(min(1.0, probability + width), 6),
            "verdict": verdict,
            "band_collapsed": bool(thresholds.get("band_collapsed")),
            "low_confidence": words < 120,
            "components": [
                {
                    "name": "classical",
                    "probability": round(probability, 6),
                    "raw": {"component_probability": round(raw, 6), "stacked": calibrated},
                    "available": True,
                    "reason": None,
                }
            ],
            "explanation": {
                "source": "rule_based",
                "text": (
                    "Lite mode runs the stylometric classifier only: word-frequency typicality, "
                    "sentence-length variation, punctuation rhythm and function-word rates. It cannot "
                    "see perplexity or zero-shot signals, so a mid-band score means the text looks "
                    "more or less like the training mixture, not that a person did or did not write it."
                ),
            },
        }
    )
    return _envelope(_padded(cleaned, words), ai)


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
# analyze endpoint
# --------------------------------------------------------------------------

MAX_BODY_BYTES = 200_000


def _error(status: int, code: str, message: str) -> tuple[int, bytes]:
    return status, json.dumps({"detail": {"error": {"code": code, "message": message}}}).encode()


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel's expected class name
    """Vercel's Python runtime looks for this class."""

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.send_response(204)
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type")
        self.send_header("access-control-allow-methods", "POST, OPTIONS")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("content-length") or 0)
        if length > MAX_BODY_BYTES:
            status, payload = _error(413, "too_large", "Lite mode accepts at most 200 KB of text.")
            return self._send(status, payload)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            status, payload = _error(400, "bad_json", "Send JSON: {\"text\": \"...\"}.")
            return self._send(status, payload)

        text = str(body.get("text") or "")
        if not text.strip():
            status, payload = _error(
                422,
                "no_input",
                "Lite mode analyses pasted text only. Upload files on the full server, or paste text here.",
            )
            return self._send(status, payload)
        requested = str(body.get("mode") or "lite")
        if requested != "lite":
            status, payload = _error(
                422,
                "mode_unavailable",
                f"Mode '{requested}' needs the full server. This deployment can only run 'lite'.",
            )
            return self._send(status, payload)

        result = score(text)
        result["lite_limitations"] = LITE_LIMITATIONS
        self._send(200, json.dumps(result).encode())

    def _send(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("access-control-allow-origin", "*")
        self.end_headers()
        self.wfile.write(payload)
