"""``/api/health``: the single source of truth about what can run here.

The frontend disables modes from this response, and the UI prints the same reasons
to the user.  Nothing is reported as available unless its files actually loaded.
"""

from __future__ import annotations


from fastapi import APIRouter

from ..artifacts import registry
from ..config import DISCLAIMER, MODES, settings
from ..logging_setup import get_logger
from ..schemas import HealthResponse, ModeAvailability
from ..pipeline import get_corpus

log = get_logger(__name__)
router = APIRouter(tags=["health"])

VERSION = "1.0.0"


def _ocr_available() -> bool:
    from ..parsers.image_ocr import ocr_available

    return ocr_available()


def _llm_available() -> bool:
    """True when a local Ollama server answers (never required)."""
    try:
        import requests

        response = requests.get(f"{settings.ollama_url}/api/tags", timeout=1.5)
        return response.ok
    except Exception:  # noqa: BLE001 - optional local service
        return False


def build_health() -> HealthResponse:
    """Compose the health payload (also used by the smoke-test scripts)."""
    registry.load()
    modes = [
        ModeAvailability(
            mode=state.mode,
            available=state.available,
            reason=state.reason,
            components=state.components,
        )
        for state in registry.all_modes()
    ]
    models_loaded: dict[str, bool] = {}
    for component in ("classical", "ppl_gpt2", "transformer", "binoculars", "fastdetect"):
        ok, _ = registry.component_available(component)
        models_loaded[component] = ok
    try:
        corpus_size = get_corpus().stats().documents
    except Exception as exc:  # noqa: BLE001 - health must never fail
        log.warning("corpus stats failed: %s", exc)
        corpus_size = 0
    try:
        import torch

        gpu = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        gpu = False

    status = "ok" if any(m.available for m in modes) else "degraded"
    if registry.status in {"loading", "idle"}:
        status = "loading"
    return HealthResponse(
        status=status,  # type: ignore[arg-type]
        version=VERSION,
        modes=modes,
        models_loaded=models_loaded,
        corpus_size=corpus_size,
        ocr=_ocr_available(),
        llm=_llm_available(),
        gpu=gpu,
        artifact_source=registry.source,  # type: ignore[arg-type]
        artifact_notes=registry.notes,
        data_source=registry.data_source,
        default_mode=registry.default_mode() or settings.default_mode,
        disclaimer=DISCLAIMER,
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report status, per-mode availability and optional capabilities."""
    return build_health()


@router.get("/version")
def version() -> dict[str, str]:
    return {"version": VERSION, "modes": ",".join(MODES)}
