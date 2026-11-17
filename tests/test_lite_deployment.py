"""The Vercel Lite functions (``api/``) must be honest about what they can do."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "api"


@pytest.fixture(scope="module")
def core() -> "object":
    sys.path.insert(0, str(API))
    from _lib import core as lite_core

    return lite_core


def test_health_declares_only_lite_and_explains_the_rest(core: "object") -> None:
    payload = core.health()  # type: ignore[attr-defined]
    modes = {mode["mode"]: mode for mode in payload["modes"]}
    assert set(modes) == {"lite", "fast", "standard", "deep"}
    assert payload["deployment"] == "lite"
    for mode in ("fast", "standard", "deep"):
        assert modes[mode]["available"] is False
        assert modes[mode]["reason"], f"{mode} must say why it is unavailable"
    assert payload["corpus_size"] == 0, "Lite has no corpus and must not claim one"
    assert payload["lite_limitations"]


def test_score_returns_the_full_result_shape(core: "object") -> None:
    text = (ROOT / "ml" / "bundled" / "samples" / "sample_human.txt").read_text()
    result = core.score(text)  # type: ignore[attr-defined]
    assert result["status"] == "done"
    assert result["deployment"] == "lite"
    document = result["documents"][0]
    ai = document["ai"]
    assert ai["components_run"] == ["classical"]
    assert ai["components_unavailable"]
    assert 0.0 <= ai["probability"] <= 1.0
    assert ai["thresholds"]["t_human"] is not None and ai["thresholds"]["t_ai"] is not None
    assert ai["verdict"] in {"likely_human", "likely_ai", "uncertain"}
    # Lite cannot draw highlights, so it must not pretend to: no sentences, no matches.
    assert ai["sentences"] == []
    assert ai["tokens"] == []
    assert document["plagiarism"] is None
    assert result["exit_notes"]


def test_short_text_gets_no_probability(core: "object") -> None:
    result = core.score("Too short to say anything.")  # type: ignore[attr-defined]
    ai = result["documents"][0]["ai"]
    assert ai["verdict"] == "too_short"
    assert ai["probability"] is None
    assert ai["components_run"] == []


def test_invisible_characters_are_reported(core: "object") -> None:
    text = (ROOT / "ml" / "bundled" / "samples" / "sample_ai.txt").read_text()
    dirty = text[:400] + "\u200b" + text[400:]
    result = core.score(dirty)  # type: ignore[attr-defined]
    document = result["documents"][0]
    flags = [flag for flag in document["integrity"]["flags"] if flag["kind"] == "invisible_characters"]
    assert flags, "an invisible character must be reported, not silently ignored"
    assert flags[0]["count"] >= 1
    assert flags[0]["examples"]


def test_joblib_artifact_loads_without_the_app_package(core: "object") -> None:
    """A Lite cold start must not need the FastAPI application to import the model."""
    bundle = core.artifacts()  # type: ignore[attr-defined]
    assert bundle.available, bundle.error
    assert hasattr(bundle.model, "predict_proba")


def _run_handler(module_name: str, method: str, body: bytes | None = None) -> tuple[int, dict]:
    """Instantiate a Vercel handler and call one HTTP method against a fake socket."""
    spec = importlib.util.spec_from_file_location(f"vercel_{module_name}", API / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.path.insert(0, str(API))
    # A generated function defines its own dataclasses, and dataclasses resolve their
    # module through sys.modules, so the module has to be registered before it executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    handler_class = module.handler
    instance = handler_class.__new__(handler_class)
    captured: dict[str, object] = {}

    import io

    instance.wfile = io.BytesIO()
    instance.headers = {"content-length": str(len(body or b""))}
    instance.rfile = io.BytesIO(body or b"")
    headers: dict[str, str] = {}

    def fake_send_response(code: int) -> None:
        captured["status"] = code

    def fake_send_header(key: str, value: str) -> None:
        headers[key] = value

    instance.send_response = fake_send_response  # type: ignore[attr-defined]
    instance.send_header = fake_send_header  # type: ignore[attr-defined]
    instance.end_headers = lambda: None  # type: ignore[attr-defined]
    getattr(instance, f"do_{method}")()
    payload = json.loads(instance.wfile.getvalue().decode() or "{}")
    return int(captured["status"]), payload  # type: ignore[arg-type]


def test_health_function_serves_the_same_contract(core: "object") -> None:
    status, payload = _run_handler("health", "GET")
    assert status == 200
    assert payload["modes"][0]["mode"] == "lite"
    assert payload["deployment"] == "lite"


def test_analyze_function_rejects_files_and_unknown_modes() -> None:
    status, payload = _run_handler("analyze", "POST", json.dumps({"mode": "deep", "text": "hello"}).encode())
    assert status == 422
    assert payload["detail"]["error"]["code"] == "mode_unavailable"

    status, payload = _run_handler("analyze", "POST", json.dumps({}).encode())
    assert status == 422
    assert payload["detail"]["error"]["code"] == "no_input"


def test_analyze_function_scores_pasted_text() -> None:
    text = (ROOT / "ml" / "bundled" / "samples" / "sample_ai.txt").read_text()
    status, payload = _run_handler("analyze", "POST", json.dumps({"text": text, "mode": "lite"}).encode())
    assert status == 200
    assert payload["documents"][0]["ai"]["components_run"] == ["classical"]


def test_metrics_function_says_it_has_none() -> None:
    status, payload = _run_handler("metrics", "GET")
    assert status == 200
    assert payload["metrics"] is None
    assert payload["deployment"] == "lite"
    assert payload["manifest_notes"], "the page needs a sentence explaining why it is empty"


def test_samples_function_returns_original_text() -> None:
    status, payload = _run_handler("samples", "GET")
    assert status == 200
    assert len(payload) == 2
    assert all(len(sample["text"].split()) > 60 for sample in payload)


def test_generated_functions_are_not_stale() -> None:
    """``api/<name>.py`` is built from the core plus ``api/handlers/<name>.py``.

    Vercel ships a function as its entrypoint file alone, so the Lite core is inlined
    into each one. The generated files are committed (a git deploy packages the
    repository before any build step runs), which means they can go stale. This test
    rebuilds them and fails on any difference.
    """
    import subprocess

    names = ("health", "samples", "metrics", "analyze")
    before = {name: (API / f"{name}.py").read_bytes() for name in names}
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_api.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for name in names:
        after = (API / f"{name}.py").read_bytes()
        assert after == before[name], f"api/{name}.py was stale; run: python scripts/build_api.py"
        assert b"GENERATED FILE - do not edit" in after


def test_functions_are_self_contained() -> None:
    """Vercel's builder uploads only the entrypoint file.

    Neither a sibling package (``from _lib.core import ...``) nor a sibling module
    (``from _core import ...``) survives deployment: both raised ModuleNotFoundError in
    production while the build log stayed green. The generated functions must import
    nothing from this repository.
    """
    for name in ("health", "samples", "metrics", "analyze"):
        source = (API / f"{name}.py").read_text()
        for forbidden in ("from _lib", "from _core", "import _core", "from app"):
            assert forbidden not in source, f"api/{name}.py must be self-contained"
        assert "class handler(BaseHTTPRequestHandler)" in source


def test_the_committed_lite_bundle_is_present_and_loadable() -> None:
    """A git deploy ships only tracked files.

    Vercel packages functions before any build step runs, so ``frontend/public/artifacts``
    (git-ignored, staged by the deploy script) is absent from a git-based deploy. That is
    why the same endpoints answered "no lite artifact bundle is deployed with this build"
    in production while working locally. ``api/artifacts`` is the committed copy.
    """
    bundle = API / "artifacts"
    for name in ("classical.pkl", "thresholds.json"):
        assert (bundle / name).is_file(), f"missing api/artifacts/{name}"
    assert (bundle / "classical.pkl").stat().st_size < 2_000_000, "keep the Lite bundle small"

    # The function finds it the way it runs in production: from api/<name>.py.
    spec = importlib.util.spec_from_file_location("vercel_health_bundle", API / "health.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    payload = module.BODY
    assert payload["artifact_source"] == "cpu", payload["artifact_notes"]
    assert payload["models_loaded"]["classical"] is True
    assert payload["status"] == "ok"


def test_no_dependency_file_shadows_requirements_txt() -> None:
    """Vercel prefers pyproject.toml over requirements.txt when both sit by the functions.

    Adding ``api/pyproject.toml`` for a lint setting silently changed the deployed
    dependency set: the build log switched from "Installing required dependencies from
    api/requirements.txt" to uv.lock, and the functions then failed with
    ``No module named 'joblib'`` even though the code was fine.
    """
    for shadow in ("pyproject.toml", "Pipfile", "Pipfile.lock"):
        assert not (API / shadow).exists(), f"api/{shadow} would override api/requirements.txt"
    assert (API / "requirements.txt").is_file()
