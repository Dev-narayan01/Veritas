"""HTTP contract tests. They run the real app with the real (CPU) artifacts."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

pytestmark = pytest.mark.usefixtures("client")


def wait_done(client: Any, analysis_id: str, timeout: float = 120.0) -> dict[str, Any]:
    """Poll an analysis until it is done, failing loudly on error or timeout."""
    deadline = time.time() + timeout
    payload: dict[str, Any] = {}
    while time.time() < deadline:
        response = client.get(f"/api/analyses/{analysis_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] == "done":
            return payload
        assert payload["status"] != "error", payload
        time.sleep(0.4)
    raise AssertionError(f"analysis {analysis_id} did not finish: {payload.get('status')}")


def submit(client: Any, text: str, **options: Any) -> str:
    """Create a text-only analysis and return its id."""
    payload = {
        "mode": "lite",
        "ai": True,
        "plagiarism": True,
        "batch": False,
        "explain": True,
        "online": False,
        "retain": True,
        "exclusions": {"quotes": False, "bibliography": False, "min_words": 3, "hide_below_percent": 0},
    }
    payload.update(options)
    response = client.post("/api/analyses", data={"text": text, "options": json.dumps(payload)})
    assert response.status_code == 202, response.text
    return response.json()["id"]


def test_health_reports_real_components(client: Any) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["modes"], "health must list the four modes"
    assert {mode["mode"] for mode in payload["modes"]} == {"lite", "fast", "standard", "deep"}
    assert payload["corpus_size"] >= 1, "the corpus size must be the real number loaded"
    for mode in payload["modes"]:
        if mode["available"]:
            assert mode["components"], "an available mode must name its components"
            assert mode["reason"] is None
        else:
            assert mode["reason"], "an unavailable mode must say why"
    assert payload["artifact_source"] in {"kaggle", "cpu", "none"}


def test_version_and_root(client: Any) -> None:
    version = client.get("/api/version").json()
    assert version["version"]
    root = client.get("/")
    assert root.status_code == 200


def test_analysis_reports_mode_and_components(client: Any, human_text: str) -> None:
    payload = wait_done(client, submit(client, human_text))
    document = payload["documents"][0]
    ai = document["ai"]
    assert ai["mode_used"] == "lite"
    assert ai["components_run"], "the result must say which components ran"
    assert isinstance(ai["components_unavailable"], list)
    assert ai["probability"] is None or 0.0 <= ai["probability"] <= 1.0
    assert ai["thresholds"]["t_human"] is not None and ai["thresholds"]["t_ai"] is not None
    assert ai["verdict"] in {"likely_human", "likely_ai", "uncertain", "too_short"}
    assert payload["disclaimer"]
    # Sentence spans are consistent with the returned text.
    for sentence in ai["sentences"]:
        assert 0 <= sentence["start"] < sentence["end"] <= len(document["text"])


def test_unavailable_components_are_named(client: Any, human_text: str) -> None:
    payload = wait_done(client, submit(client, human_text))
    ai = payload["documents"][0]["ai"]
    for item in ai["components_unavailable"]:
        assert item["component"] and item["reason"], "an unavailable component needs a reason"
    # In this environment torch is absent, so bins must not be listed as having run.
    ran = set(ai["components_run"])
    assert not ran & {"binoculars", "fast_detectgpt", "transformer"}, ran


def test_plagiarism_reports_real_corpus_size(client: Any, human_text: str) -> None:
    payload = wait_done(client, submit(client, human_text))
    plagiarism = payload["documents"][0]["plagiarism"]
    stats = client.get("/api/corpus/stats").json()
    assert plagiarism["corpus_size"] == stats["documents"]
    assert 0.0 <= plagiarism["similarity_percent"] <= 100.0
    assert 0.0 <= plagiarism["similarity_percent_filtered"] <= plagiarism["similarity_percent"] + 1e-6
    for match in plagiarism["matches"]:
        start, end = match["query_span"]
        assert 0 <= start < end <= len(payload["documents"][0]["text"])
        assert match["words"] >= 1


def test_exclusions_reduce_the_filtered_percentage(client: Any, human_text: str) -> None:
    default = wait_done(client, submit(client, human_text))
    strict = wait_done(
        client,
        submit(
            client,
            human_text,
            exclusions={"quotes": True, "bibliography": True, "min_words": 25, "hide_below_percent": 5},
        ),
    )
    before = default["documents"][0]["plagiarism"]
    after = strict["documents"][0]["plagiarism"]
    assert after["similarity_percent_filtered"] <= before["similarity_percent_filtered"] + 1e-6
    assert len(after["matches"]) <= len(before["matches"])


def test_unknown_mode_is_rejected_or_downgraded(client: Any, human_text: str) -> None:
    response = client.post(
        "/api/analyses",
        data={"text": human_text, "options": json.dumps({"mode": "ultra", "ai": True})},
    )
    assert response.status_code in {202, 422}
    if response.status_code == 202:
        payload = wait_done(client, response.json()["id"])
        assert payload["documents"][0]["ai"]["mode_used"] in {"lite", "fast", "standard", "deep"}


def test_empty_input_is_rejected(client: Any) -> None:
    response = client.post("/api/analyses", data={"options": json.dumps({"mode": "lite"})})
    assert response.status_code == 422
    assert "error" in response.json()["detail"]


def test_short_text_is_reported_as_too_short(client: Any) -> None:
    payload = wait_done(client, submit(client, "Too short."))
    ai = payload["documents"][0]["ai"]
    assert ai["verdict"] == "too_short"
    assert ai["probability"] is None, "a document that is too short must not get a probability"


def test_retain_false_deletes_the_stored_text(client: Any, human_text: str) -> None:
    payload = wait_done(client, submit(client, human_text, retain=False))
    assert payload["documents"][0]["text"] == ""
    assert payload["documents"][0]["ai"]["sentences"], "analysis must still be visible"


def test_exports_are_real_files(client: Any, human_text: str) -> None:
    analysis_id = submit(client, human_text)
    wait_done(client, analysis_id)

    export_json = client.get(f"/api/analyses/{analysis_id}/export.json")
    assert export_json.status_code == 200
    assert export_json.json()["documents"], "the JSON export must contain the documents"

    export_csv = client.get(f"/api/analyses/{analysis_id}/export.csv")
    assert export_csv.status_code == 200
    header = export_csv.text.splitlines()[0].lower()
    assert "file" in header and "verdict" in header
    # All three sections are always present, even when a section has no rows.
    for section in ("sentence_start", "source_title", "corpus_size"):
        assert section in export_csv.text, section

    report = client.get(f"/api/analyses/{analysis_id}/report.pdf")
    assert report.status_code == 200
    assert report.content[:4] == b"%PDF"


def test_history_lists_and_deletes(client: Any, human_text: str) -> None:
    analysis_id = submit(client, human_text)
    wait_done(client, analysis_id)

    items = client.get("/api/analyses").json()
    assert isinstance(items, list), "history is a plain list of summaries"
    assert any(item["id"] == analysis_id for item in items)

    assert client.delete(f"/api/analyses/{analysis_id}").status_code in {200, 204}
    assert client.get(f"/api/analyses/{analysis_id}").status_code == 404


def test_events_stream_reaches_done(client: Any, human_text: str) -> None:
    analysis_id = submit(client, human_text)
    wait_done(client, analysis_id)
    events = client.get(f"/api/analyses/{analysis_id}/events").text
    assert "progress" in events, events[:400]
    assert "done" in events, "the SSE stream must signal completion"


def test_url_parser_refuses_private_addresses(client: Any) -> None:
    response = client.post(
        "/api/analyses",
        data={"url": "http://127.0.0.1:8000/", "options": json.dumps({"mode": "lite"})},
    )
    assert response.status_code == 422
    body = json.dumps(response.json()).lower()
    assert "not public" in body or "private" in body or "local" in body


def test_samples_endpoint_returns_two_documents(client: Any) -> None:
    samples = client.get("/api/samples").json()
    assert len(samples) >= 2
    assert all(sample["text"].strip() for sample in samples)


def test_file_upload_round_trip(client: Any, tmp_copy: "callable") -> None:
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "ml" / "bundled" / "samples" / "sample_ai.txt"
    path = tmp_copy(source)
    with path.open("rb") as handle:
        response = client.post(
            "/api/analyses",
            files={"files": (path.name, handle, "text/plain")},
            data={"options": json.dumps({"mode": "lite", "ai": True, "plagiarism": False})},
        )
    assert response.status_code == 202, response.text
    payload = wait_done(client, response.json()["id"])
    document = payload["documents"][0]
    assert document["filename"] == path.name
    assert document["words"] > 20
    assert document["plagiarism"] is None
    assert document["ai"]["probability"] is not None


def test_batch_comparison_of_two_documents(client: Any, tmp_copy: "callable") -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "ml" / "bundled" / "samples"
    first = tmp_copy(root / "sample_human.txt")
    second = tmp_copy(root / "sample_ai.txt", "second.txt")
    files = []
    handles = []
    try:
        for path in (first, second):
            handle = path.open("rb")
            handles.append(handle)
            files.append(("files", (path.name, handle, "text/plain")))
        response = client.post(
            "/api/analyses",
            files=files,
            data={"options": json.dumps({"mode": "lite", "ai": True, "plagiarism": True, "batch": True})},
        )
        assert response.status_code == 202, response.text
    finally:
        for handle in handles:
            handle.close()

    payload = wait_done(client, response.json()["id"])
    batch = payload["batch"]
    assert batch is not None, "batch comparison was requested but not returned"
    assert len(batch["matrix"]) == 2
    assert batch["matrix"][0][0] == pytest.approx(100.0, abs=0.01)
    assert len(batch["pairs"]) == 1


def test_metrics_expose_ood_first_and_never_fabricate(client: Any) -> None:
    payload = client.get("/api/metrics").json()
    assert payload["artifact_source"] in {"kaggle", "cpu", "none"}
    splits = (payload.get("metrics") or {}).get("splits") or {}
    assert "test_ood" in splits
    assert (payload.get("metrics") or {}).get("notes") is not None
    baseline = payload.get("baseline")
    if baseline is not None:
        assert "third" in json.dumps(baseline).lower() or baseline.get("available") is False


def test_plot_that_does_not_exist_is_404(client: Any) -> None:
    assert client.get("/api/metrics/plots/nope.png").status_code == 404


def test_unavailable_mode_names_the_blocking_component(client: Any) -> None:
    """The reason must name what is actually missing.

    In a CPU bundle both the neural weights *and* the stackers for the heavy modes are absent.
    Reporting "stacker missing" would send a reader looking in the wrong place; the component
    is the blocker they can act on.
    """
    modes = {mode["mode"]: mode for mode in client.get("/api/health").json()["modes"]}
    for name in ("standard", "deep"):
        entry = modes[name]
        if entry["available"]:
            continue
        assert entry["reason"], f"{name} is unavailable and must say why"
        assert not entry["reason"].startswith("stacker"), (
            f"{name} should blame the component it cannot run, not its stacker: {entry['reason']}"
        )
        assert "transformer" in entry["reason"] or "torch" in entry["reason"]
