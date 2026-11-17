"""Install a downloaded artifact bundle into ``ml/artifacts`` after verifying it.

    python ml/adopt_artifacts.py dist/veritas_artifacts.zip
    python ml/adopt_artifacts.py ~/Downloads/veritas_artifacts.zip --keep

Verification refuses to adopt a bundle that would make the site dishonest:

* ``manifest.json`` must exist and name the product;
* every step it claims to have run must have produced its output file;
* a bundle whose ``run`` is ``smoke`` is refused unless ``--allow-smoke`` is given, because
  smoke numbers must never reach the live site.

The previous artifacts are moved to ``ml/artifacts.previous`` (or kept alongside with
``--keep``) so adoption is reversible.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts"
BACKUP_DIR = REPO_ROOT / "ml" / "artifacts.previous"

STEP_OUTPUTS = {
    "prepare_data": None,  # data lives outside the bundle
    "train_classical": "classical.pkl",
    "train_perplexity_model": "ppl_gpt2_model.pkl",
    "train_transformer": "transformer_meta.json",
    "compute_zeroshot_features": "zeroshot_meta.json",
    "train_stackers": "stacker_lite.pkl",
    "tune_thresholds": "thresholds.json",
    "evaluate": "metrics.json",
    "robustness": "robustness.json",
    "bias_check": "bias.json",
    "embedding_projection": "embedding_projection.json",
    "build_corpus": "corpus/meta.json",
    "make_plag_benchmark": "plag_benchmark/benchmark.json",
    "eval_plagiarism": "plag_benchmark_results.json",
}


def _display(path: Path) -> str:
    """Path relative to the repo when possible, absolute otherwise (tests use temp dirs)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def extract(archive: Path, target: Path) -> Path:
    """Extract, tolerating a bundle zipped with or without its top folder."""
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    if (target / "manifest.json").exists():
        return target
    for child in target.iterdir():
        if child.is_dir() and (child / "manifest.json").exists():
            return child
    raise SystemExit(f"{archive} does not contain a manifest.json")


def verify(directory: Path, allow_smoke: bool) -> dict:
    """Check the bundle and return its manifest, or exit with a readable reason."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("missing manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("product") != "Veritas":
        raise SystemExit(f"manifest is for {manifest.get('product')!r}, not Veritas")
    if manifest.get("run") == "smoke" and not allow_smoke:
        raise SystemExit(
            "this bundle is a smoke run: its numbers exist to prove the pipeline executes, "
            "not to be shown as results. Re-run ml/run_all.py without --smoke, or pass "
            "--allow-smoke if you are only testing the deployment."
        )

    missing: list[str] = []
    for step, relative in STEP_OUTPUTS.items():
        status = (manifest.get("steps") or {}).get(step, {}).get("status")
        if status != "ok" or relative is None:
            continue
        if not (directory / relative).exists():
            missing.append(f"{step} → {relative}")
    if missing:
        raise SystemExit("manifest claims these outputs but they are absent: " + ", ".join(missing))

    for required in ("thresholds.json", "metrics.json"):
        if not (directory / required).exists():
            raise SystemExit(f"incomplete bundle: {required} is missing")
    return manifest


def adopt(directory: Path, keep_previous: bool) -> None:
    """Copy the bundle into place, keeping the old one for rollback."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if any(ARTIFACT_DIR.iterdir()):
        if keep_previous:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            for item in ARTIFACT_DIR.iterdir():
                if item.name == ".gitkeep":
                    continue
                destination = BACKUP_DIR / item.name
                if destination.is_dir():
                    shutil.rmtree(destination)
                shutil.move(str(item), destination)
            print(f"previous artifacts moved to {_display(BACKUP_DIR)}")
        else:
            shutil.rmtree(BACKUP_DIR, ignore_errors=True)
            shutil.move(str(ARTIFACT_DIR), str(BACKUP_DIR))
            ARTIFACT_DIR.mkdir()
            print(f"previous artifacts moved to {_display(BACKUP_DIR)} (pass --keep to merge instead)")
    for item in directory.iterdir():
        destination = ARTIFACT_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="veritas_artifacts.zip from Kaggle or a local run")
    parser.add_argument("--keep", action="store_true", help="merge into ml/artifacts instead of replacing it")
    parser.add_argument("--allow-smoke", action="store_true", help="accept a smoke bundle (deployment testing only)")
    parser.add_argument("--target", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()

    if not args.archive.exists():
        raise SystemExit(f"no such file: {args.archive}")
    with tempfile.TemporaryDirectory() as tmp:
        root = extract(args.archive, Path(tmp))
        manifest = verify(root, args.allow_smoke)
        adopt(root, args.keep)
    print(
        "adopted bundle: "
        f"source={manifest.get('source')} run={manifest.get('run')} "
        f"modes={manifest.get('modes_fitted')} components={manifest.get('components_trained')}"
    )
    print("restart the backend (or wait for the artifact warm-up) to load it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
