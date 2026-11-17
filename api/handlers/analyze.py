"""Request handling for POST /api/analyze: one pasted text, classical detector only.

The full server accepts multipart uploads, URLs, multiple files and options. Lite accepts
JSON ``{"text": "...", "mode": "lite"}`` and refuses everything else with a clear message
rather than silently doing less than the user asked for.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler


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
