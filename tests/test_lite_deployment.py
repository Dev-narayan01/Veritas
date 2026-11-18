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


def test_health_reports_each_mode_honestly(core: "object") -> None:
    """Whatever is available is advertised; whatever is not says why.

    ``fast`` became available once the torch-free perplexity component was bundled, so the
    old "only lite" assertion is gone. The invariant that matters is that an unavailable
    mode always carries a reason and an available one never claims components it lacks.
    """
    payload = core.health()  # type: ignore[attr-defined]
    modes = {mode["mode"]: mode for mode in payload["modes"]}
    assert set(modes) == {"lite", "fast", "standard", "deep"}
    assert payload["deployment"] == "lite"
    assert modes["lite"]["available"] is True
    for name in ("standard", "deep"):
        assert modes[name]["available"] is False
        assert modes[name]["reason"], f"{name} must say why it is unavailable"
    fast = modes["fast"]
    assert fast["available"] == payload["models_loaded"]["ppl_gpt2"], (
        "fast must be available exactly when the perplexity component is"
    )
    if not fast["available"]:
        assert fast["reason"], "an unavailable fast mode must say why"
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


def test_metrics_function_never_invents_numbers() -> None:
    """Either the real evaluation bundle is served, or the page says it is empty.

    Both are honest; a placeholder number would not be. The frontend reads
    ``metrics.splits[split].modes[mode]`` with ``roc_auc``/``tpr_at_1pct_fpr``/``ece``
    keys, so when data exists those keys must be present.
    """
    status, payload = _run_handler("metrics", "GET")
    assert status == 200
    assert payload["deployment"] == "lite"
    assert payload["manifest_notes"], "the page always needs a sentence explaining the numbers"

    if payload["metrics"] is None:
        assert payload["artifact_source"] == "none"
        return

    assert payload["artifact_source"] == "cpu"
    splits = payload["metrics"]["splits"]
    assert "test" in splits and "test_ood" in splits
    for split in ("test", "test_ood"):
        for mode in ("lite", "fast"):
            entry = splits[split]["modes"][mode]
            for key in ("roc_auc", "tpr_at_1pct_fpr", "ece"):
                assert isinstance(entry[key], (int, float)), f"{split}/{mode} missing {key}"
    assert payload["thresholds"]["t_human"] < payload["thresholds"]["t_ai"]


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
    # 5 MB char-ngram model vs Vercel's 500 MB unzipped bundle limit: plenty of headroom,
    # but a runaway artifact (e.g. accidentally committing the 26 MB ppl model, which Lite
    # never loads) should still fail here rather than at deploy time.
    assert (bundle / "classical.pkl").stat().st_size < 20_000_000, "keep the Lite bundle small"

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
    # The fitted stacker carries the Platt calibration and its own thresholds; without it
    # the runtime falls back to raw component scores, which are not calibrated.
    assert (bundle / "stacker_lite.pkl").is_file(), "ship the fitted lite stacker"
    assert module.BODY["artifact_notes"], "health must describe the bundle it loaded"


def test_lite_uses_the_calibrated_stacker_when_present() -> None:
    """A stacked probability must be reported as such, not silently swapped for raw."""
    sys.path.insert(0, str(API))
    from _lib import core

    bundle = core.artifacts()
    if bundle.stacker is None:
        pytest.skip("committed bundle has no stacker; it reports raw component scores")
    text = (ROOT / "ml" / "bundled" / "samples" / "sample_ai.txt").read_text()
    ai = core.score(text, mode="lite")["documents"][0]["ai"]
    rows = {row["name"]: row for row in ai["components"]}
    assert "stacked" in rows, "the calibrated result must be its own row"
    stacked = rows["stacked"]
    assert stacked["raw"]["calibrated"] is True
    classical = rows["classical"]["probability"]
    assert classical != ai["probability"], "a raw component score must not be passed off as the verdict"
    assert ai["probability"] == pytest.approx(stacked["probability"], abs=1e-6)
    assert ai["thresholds"]["t_ai"] == pytest.approx(bundle.stacker.thresholds.t_ai, abs=1e-6)


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


def test_fast_mode_runs_both_components_when_bundled() -> None:
    """``fast`` = classical + perplexity; it must report both, not claim one."""
    sys.path.insert(0, str(API))
    from _lib import core

    bundle = core.artifacts()
    if not bundle.fast_available:
        pytest.skip("perplexity bundle not present")
    text = (ROOT / "ml" / "bundled" / "samples" / "sample_ai.txt").read_text()
    ai = core.score(text, mode="fast")["documents"][0]["ai"]
    assert ai["mode_requested"] == "fast"
    assert ai["mode_used"] == "fast"
    assert set(ai["components_run"]) == {"classical", "ppl_gpt2"}
    names = {row["name"] for row in ai["components"]}
    assert {"classical", "ppl_gpt2"} <= names
    ppl_row = next(row for row in ai["components"] if row["name"] == "ppl_gpt2")
    assert ppl_row["raw"]["backend"] == "ngram_lm", "the deployed component is the n-gram one"


def test_fast_mode_degrades_to_lite_instead_of_failing() -> None:
    """A missing perplexity bundle must not turn a request into an error."""
    sys.path.insert(0, str(API))
    from _lib import core

    original = core._ARTIFACTS
    try:
        bundle = core.artifacts()
        core._ARTIFACTS = core.LiteArtifacts(
            bundle.directory,
            model=bundle.model,
            thresholds=bundle.thresholds,
            stacker=bundle.stacker,
            ppl=None,
            stacker_fast=None,
        )
        text = (ROOT / "ml" / "bundled" / "samples" / "sample_ai.txt").read_text()
        ai = core.score(text, mode="fast")["documents"][0]["ai"]
        assert ai["mode_requested"] == "fast"
        assert ai["mode_used"] == "lite", "it must say it fell back, not pretend"
        assert ai["components_run"] == ["classical"]
        reasons = {row["component"]: row["reason"] for row in ai["components_unavailable"]}
        assert "fell back" in reasons["ppl_gpt2"]
    finally:
        core._ARTIFACTS = original


def test_fast_ships_the_ood_chosen_band_not_the_tuned_one() -> None:
    """The tuned fast band is measurably harmful out-of-domain; guard the shipped one.

    On 464 held-out OOD rows the tuned band (t_human 0.93 / t_ai 0.978) fires on 87% of
    texts and is wrong on 108 of them; the shipped band commits on 13% and is wrong on 3.
    Restoring the tuned band would make the deployment look more capable while being
    less trustworthy, so this test pins the choice.
    """
    sys.path.insert(0, str(API))
    from _lib import core

    thresholds = core.artifacts().thresholds or {}
    deployed = thresholds.get("fast_deployed")
    if not deployed:
        pytest.skip("no fast_deployed band in this bundle")
    assert "reason" in deployed and len(deployed["reason"]) > 40
    assert deployed["measured"]["wrong"] <= thresholds["fast_rejected_tuned"]["measured"]["wrong"]
    assert deployed["t_ai"] >= thresholds["fast_rejected_tuned"]["t_ai"]
    assert deployed["t_human"] <= thresholds["fast_rejected_tuned"]["t_human"]


def test_ppl_port_matches_the_trained_implementation() -> None:
    """The deployed n-gram port is hand-written; drift here silently changes verdicts."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_fast_mode.py")],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if "perplexity bundle not present" in result.stderr or result.returncode == 3:
        pytest.skip("perplexity bundle not present")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "IDENTICAL" in result.stdout
