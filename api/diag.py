"""TEMPORARY diagnostic for the deployed Python bundle. Delete after debugging."""

from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler


def _dirs() -> dict:
    out = {}
    for path in ("/var/task", "/var/task/api", "/var"):
        try:
            out[path] = sorted(os.listdir(path))[:40]
        except Exception as exc:  # noqa: BLE001
            out[path] = f"error: {exc}"
    return out


def _modules() -> dict:
    out = {}
    for name in ("joblib", "numpy", "sklearn", "fastapi", "scipy"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "imported")
        except Exception as exc:  # noqa: BLE001
            out[name] = f"FAILED: {exc}"
    return out


BODY = {
    "python": sys.version,
    "cwd": os.getcwd(),
    "sys_path": sys.path,
    "dirs": _dirs(),
    "modules": _modules(),
    "error": None,
}
try:
    BODY["artifacts"] = os.listdir(os.path.join(os.path.dirname(__file__), "artifacts"))
except Exception:  # noqa: BLE001
    BODY["artifacts"] = traceback.format_exc(limit=2)


class handler(BaseHTTPRequestHandler):  # noqa: N801
    def do_GET(self) -> None:  # noqa: N802
        payload = json.dumps(BODY, indent=1).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(payload)
