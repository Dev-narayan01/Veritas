"""GET /api/health on the Lite deployment."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler

from _lib.core import health

BODY = health()


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel's expected class name
    """Vercel's Python runtime looks for this class."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        import json

        payload = json.dumps(BODY).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("cache-control", "s-maxage=60")
        self.end_headers()
        self.wfile.write(payload)
