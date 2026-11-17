"""Availability probe for the optional zero-shot components.

Importing ``torch`` and a 0.5B model is expensive, so the probe only answers
"could this run here?" — it checks that the libraries import and the model files
are cached locally or downloadable.  The actual load happens lazily on first use
inside the component itself.
"""

from __future__ import annotations

from functools import lru_cache

from ..logging_setup import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=4)
def zero_shot_available(name: str) -> tuple[bool, str | None]:
    """Return (available, reason) for ``binoculars`` or ``fastdetect``."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return False, f"{name} needs torch and transformers ({exc.__class__.__name__})"
    try:
        from ..config import settings

        model = settings.binoculars_observer if name == "binoculars" else settings.fastdetect_model
        from huggingface_hub import try_to_load_from_cache
        import os

        cached = try_to_load_from_cache(model, "config.json")
        if cached is os.sep or cached is None:
            # Not cached: allowed only when a network token path exists.
            if not settings.hub_repo:
                return (
                    False,
                    f"{name} weights for {model} are not cached locally and "
                    "no model repository is configured",
                )
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        log.debug("zero-shot probe for %s inconclusive: %s", name, exc)
    return True, None
