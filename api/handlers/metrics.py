"""Request handling for GET /api/metrics.

Lite ships no evaluation artifacts. Rather than a page of invented numbers, the endpoint
says so, and the UI renders that as "metrics unavailable on this deployment" with a link
to the full server. ``test_ood`` stays in the payload shape so the Evaluation page's
ordering logic is unchanged.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler

BODY = {
    "artifact_source": "none",
    "data_source": "unknown",
    "metrics": None,
    "manifest": {"deployment": "lite"},
    "robustness": None,
    "bias": None,
    "cross_generator": None,
    "baseline": None,
    "plag_thresholds": None,
    "plag_benchmark_results": None,
    "corpus_stats": None,
    "embedding_projection": None,
    "plots": [],
    "plots_base": "/api/metrics/plots",
    "thresholds": None,
    "manifest_notes": [
        "This is the Lite deployment. Evaluation artifacts are served by the full analysis "
        "server, which evaluates on held-out data; nothing is fabricated here."
    ],
    "deployment": "lite",
}


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel's expected class name
    """Vercel's Python runtime looks for this class."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        payload = json.dumps(BODY).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("access-control-allow-origin", "*")
        self.end_headers()
        self.wfile.write(payload)
