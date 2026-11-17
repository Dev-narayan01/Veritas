"""Request handling for GET /api/health.

The payload is built defensively: if the inlined core cannot produce it, the function
still answers 200 with a readable ``degraded`` body naming the failure. A function that
raises at import time answers every request with an opaque 500, and the site then shows
"server unreachable", which hides the cause.
"""

from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler


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
