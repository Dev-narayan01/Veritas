"""Orchestrate the whole ML pipeline and produce the artifact bundle.

``python ml/run_all.py [--smoke|--full] [--force] [--steps a,b]``

Every step is a standalone script, so a failed run can resume: a step is skipped when
its output file already exists unless ``--force`` is given.  At the end the script
writes ``ml/artifacts/manifest.json`` (code version, data source, split hash,
components, creation time), copies the training plots and corpus index into the
artifact directory and zips everything into ``veritas_artifacts.zip``.  When
``HF_TOKEN`` and ``VERITAS_HUB_REPO`` are set it also pushes the bundle to the Hub.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import ARTIFACT_DIR, REPO_ROOT, RunConfig, log, read_json, timer, write_json

STEPS: list[tuple[str, str, str]] = [
    # (module, output that marks completion relative to ml/artifacts, label)
    ("prepare_data", "data/processed/split_manifest.json", "data + group-disjoint splits"),
    ("train_classical", "artifacts/classical.pkl", "classical TF-IDF detector"),
    ("train_perplexity_model", "artifacts/ppl_gpt2_model.pkl", "token statistics detector"),
    ("train_transformer", "artifacts/transformer_meta.json", "fine-tuned transformer"),
    ("compute_zeroshot_features", "artifacts/zeroshot_meta.json", "zero-shot features"),
    ("train_stackers", "artifacts/stacker_lite.pkl", "mode stackers"),
    ("tune_thresholds", "artifacts/thresholds.json", "Platt calibration + thresholds"),
    ("evaluate", "artifacts/metrics.json", "evaluation + ablation"),
    ("eval_cross_generator", "artifacts/cross_generator.json", "leave-one-generator-out"),
    ("eval_baseline_detector", "artifacts/baseline.json", "third-party baseline"),
    ("robustness", "artifacts/robustness.json", "perturbation robustness"),
    ("bias_check", "artifacts/bias.json", "ESL-proxy bias check"),
    ("embedding_projection", "artifacts/embedding_projection.json", "3D projection"),
    ("build_corpus", "artifacts/corpus/meta.json", "plagiarism corpus index"),
    ("make_plag_benchmark", "artifacts/plag_benchmark/benchmark.json", "synthetic benchmark"),
    ("tune_plag_thresholds", "artifacts/plag_thresholds.json", "plagiarism thresholds"),
    ("eval_plagiarism", "artifacts/plag_benchmark_results.json", "plagiarism evaluation"),
]

# Steps that are allowed to conclude "not possible here" without failing the run.
SOFT_STEPS = {
    "train_transformer",
    "compute_zeroshot_features",
    "eval_cross_generator",
    "eval_baseline_detector",
    "make_plag_benchmark",
}


def _code_version() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - git may be absent
        commit, dirty = "", ""
    return {"commit": commit or "unknown", "dirty": "yes" if dirty else "no"}


def _step_done(module: str, marker: str, force: bool) -> bool:
    path = REPO_ROOT / marker
    if force:
        return False
    return path.exists()


def _run_step(module: str, cfg: RunConfig, force: bool) -> int:
    script = REPO_ROOT / "ml" / f"{module}.py"
    if not script.exists():
        log.warning("step %s has no script at %s; skipping", module, script)
        return 0
    command = [sys.executable, str(script)]
    if cfg.smoke:
        command.append("--smoke")
    elif cfg.full:
        command.append("--full")
    if force:
        command.append("--force")
    log.info("running: %s", " ".join(command))
    completed = subprocess.run(command, cwd=REPO_ROOT / "ml", check=False)
    return completed.returncode


def write_manifest(cfg: RunConfig, steps: dict[str, dict]) -> dict:
    """Assemble manifest.json: everything a consumer needs to judge the bundle."""
    split_manifest = read_json(REPO_ROOT / "data" / "processed" / "split_manifest.json", {}) or {}
    data_source = split_manifest.get("data_source", "unknown")
    stacker_meta = read_json(ARTIFACT_DIR / "stacker_meta.json", {}) or {}
    transformer_meta = read_json(ARTIFACT_DIR / "transformer_meta.json", {}) or {}
    zeroshot_meta = read_json(ARTIFACT_DIR / "zeroshot_meta.json", {}) or {}
    ppl_meta = read_json(ARTIFACT_DIR / "ppl_meta.json", {}) or {}
    components = ["classical"]
    if ppl_meta.get("backend"):
        components.append("ppl_gpt2")
    if transformer_meta.get("trained"):
        components.append("transformer")
    for name in ("binoculars", "fastdetect"):
        if (zeroshot_meta.get(name) or {}).get("available"):
            components.append(name)
    manifest = {
        "product": "Veritas",
        "code_version": _code_version(),
        "data_source": data_source,
        "split_sha256": (split_manifest.get("splits") or {}).get("test_ood", {}).get("sha256"),
        "components_trained": components,
        "modes_fitted": [m for m, entry in (stacker_meta.get("modes") or {}).items() if entry.get("fitted")],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run": "smoke" if cfg.smoke else ("full" if cfg.full else "cpu"),
        "source": "kaggle" if (cfg.full and not cfg.smoke) else "cpu",
        "smoke": cfg.smoke,
        "ppl_backend": ppl_meta.get("backend", "unknown"),
        "steps": steps,
        "notes": [],
    }
    if cfg.smoke:
        manifest["notes"].append("Smoke run: numbers are a pipeline check, not results.")
    if data_source == "synthetic_demo":
        manifest["notes"].append(
            "DEMO DATA: the corpus could not be downloaded, so this bundle must not be used "
            "for any published claim."
        )
    write_json(ARTIFACT_DIR / "manifest.json", manifest)
    return manifest


def bundle(zip_path: Path) -> Path:
    """Zip the artifact directory (plots and corpus index included)."""
    import shutil

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    base = zip_path.with_suffix("")
    if base.exists():
        shutil.rmtree(base)
    shutil.copytree(ARTIFACT_DIR, base, ignore=shutil.ignore_patterns("features", "zeroshot", "*_train"))
    archive = shutil.make_archive(str(base), "zip", base)
    log.info("bundle written to %s", archive)
    shutil.rmtree(base, ignore_errors=True)
    return Path(archive)


def push_to_hub(archive: Path) -> None:
    """Upload to ``VERITAS_HUB_REPO`` when a token is available."""
    import os
    import tempfile
    import zipfile

    token = os.environ.get("HF_TOKEN", "")
    repo = os.environ.get("VERITAS_HUB_REPO", "")
    if not token or not repo:
        log.warning("HF_TOKEN or VERITAS_HUB_REPO missing: skipping the Hub upload")
        return
    try:  # pragma: no cover - network + credentials
        from huggingface_hub import HfApi, create_repo

        create_repo(repo, repo_type="model", exist_ok=True, token=token)
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(tmp)
            HfApi().upload_folder(folder_path=tmp, repo_id=repo, repo_type="model", token=token)
        log.info("artifacts uploaded to %s", repo)
    except Exception as exc:  # noqa: BLE001 - upload is best effort
        log.error("hub upload failed: %s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU correctness run")
    parser.add_argument("--full", action="store_true", help="Kaggle GPU scale")
    parser.add_argument("--force", action="store_true", help="redo completed steps")
    parser.add_argument("--steps", default="", help="comma-separated subset of step modules")
    parser.add_argument("--no-bundle", action="store_true")
    args = parser.parse_args(argv)
    cfg = RunConfig(smoke=args.smoke, full=args.full)
    selected = {s.strip() for s in args.steps.split(",") if s.strip()} or None

    results: dict[str, dict] = {}
    with timer(f"run_all ({cfg.label})"):
        for module, marker, label in STEPS:
            if selected and module not in selected:
                continue
            if _step_done(module, marker, args.force):
                log.info("skip %-26s (already complete)", module)
                results[module] = {"label": label, "status": "skipped"}
                continue
            log.info("step %-26s %s", module, label)
            code = _run_step(module, cfg, args.force)
            results[module] = {"label": label, "status": "ok" if code == 0 else f"exit {code}"}
            if code != 0:
                if module in SOFT_STEPS:
                    log.warning("step %s could not run here; continuing (soft step)", module)
                    continue
                log.error("step %s failed with exit code %s; stopping", module, code)
                write_manifest(cfg, results)
                return code

    manifest = write_manifest(cfg, results)
    if not args.no_bundle:
        archive = bundle(REPO_ROOT / "dist" / "veritas_artifacts.zip")
        push_to_hub(archive)
    log.info(
        "finished. data_source=%s components=%s modes=%s",
        manifest["data_source"], manifest["components_trained"], manifest["modes_fitted"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
