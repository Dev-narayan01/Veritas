"""GET /api/health on the Lite deployment."""

from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler


def _build() -> dict:
    """Build the health payload, reporting import/load failures instead of crashing.

    A serverless function that raises at import time answers every request with an opaque
    500. The site then shows "server unreachable", which hides the actual cause. Building
    the body lazily means a broken bundle still produces a readable ``degraded`` payload
    that names the failure, and the UI shows the server's own reason.
    """
    try:
        from _core import health
    except Exception:  # noqa: BLE001 - any import failure is a runtime state to report
        return {
            "status": "degraded",
            "version": "unknown",
            "modes": [],
            "models_loaded": {},
            "corpus_size": 0,
            "ocr": False,
            "llm": False,
            "gpu": False,
            "artifact_source": "none",
            "artifact_notes": ["lite core failed to import: " + traceback.format_exc(limit=3)],
            "data_source": "unknown",
            "default_mode": "lite",
            "disclaimer": "Lite analysis is unavailable on this deployment.",
            "deployment": "lite",
            "lite_limitations": [],
        }
    return health()


BODY = _build()


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
