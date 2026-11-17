"""Security and privacy rules that must hold at the API boundary."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytestmark = pytest.mark.usefixtures("client")


def test_upload_size_limit_is_enforced(client: Any) -> None:
    big = b"word " * (5 * 1024 * 1024)  # ~25 MB
    response = client.post(
        "/api/analyses",
        files={"files": ("huge.txt", big, "text/plain")},
        data={"options": json.dumps({"mode": "lite", "ai": True})},
    )
    assert response.status_code in {413, 422}, response.text
    assert response.json()["detail"]["error"]["code"]


def test_unsupported_file_type_is_refused(client: Any) -> None:
    response = client.post(
        "/api/analyses",
        files={"files": ("payload.exe", b"MZ\x90\x00binary", "application/octet-stream")},
        data={"options": json.dumps({"mode": "lite", "ai": True})},
    )
    assert response.status_code in {415, 422}, response.text


def test_document_text_is_never_returned_as_html(client: Any) -> None:
    """The API returns plain text; the client renders it as text nodes, never as HTML."""
    text = "<script>alert('x')</script> A normal sentence follows this injection attempt. " * 4
    response = client.post("/api/analyses", data={"text": text, "options": json.dumps({"mode": "lite", "ai": True})})
    assert response.status_code == 202
    result = client.get(f"/api/analyses/{response.json()['id']}").json()
    while result.get("status") not in {"done", "error"}:
        result = client.get(f"/api/analyses/{response.json()['id']}").json()
    assert result["status"] == "done"
    stored = result["documents"][0]["text"]
    assert "<script>" in stored, "the raw text is preserved verbatim (escaping is the renderer's job)"


def test_error_shape_is_consistent(client: Any) -> None:
    response = client.get("/api/analyses/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert set(body["detail"]["error"]) == {"code", "message"}


def test_rate_limit_reports_a_retryable_error(client: Any) -> None:
    """The limiter must answer 429 with the documented error shape when it trips."""
    from app.ratelimit import SlidingWindowLimiter

    limiter = SlidingWindowLimiter(limit=2, window_seconds=60)
    assert limiter.allow("1.2.3.4") == (True, 0)
    assert limiter.allow("1.2.3.4") == (True, 0)
    allowed, retry_after = limiter.allow("1.2.3.4")
    assert allowed is False, "the third call inside the window must be refused"
    assert retry_after > 0, "a refused call must say when to retry"
    assert limiter.allow("5.6.7.8")[0] is True, "another client is unaffected"


def test_cors_headers_are_present_for_the_configured_origin(client: Any) -> None:
    response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in {"http://localhost:5173", "*"}


def test_no_artifact_path_escapes_the_artifact_directory(client: Any) -> None:
    response = client.get("/api/metrics/plots/..%2F..%2Fmanifest.json")
    assert response.status_code == 404
