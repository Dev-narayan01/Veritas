"""Artifact adoption must refuse anything that would make the site dishonest."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

import adopt_artifacts  # noqa: E402  (needs the sys.path entry)


def _bundle(tmp_path: Path, manifest: dict, files: dict[str, str] | None = None) -> Path:
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        for name, content in (files or {"thresholds.json": "{}", "metrics.json": "{}"}).items():
            zf.writestr(name, content)
    return archive


def _manifest(**overrides) -> dict:
    base = {
        "product": "Veritas",
        "run": "full",
        "source": "kaggle",
        "steps": {"train_classical": {"status": "ok"}},
    }
    base.update(overrides)
    return base


def _extract(archive: Path, target: Path) -> Path:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    return target


def test_accepts_a_complete_non_smoke_bundle(tmp_path: Path) -> None:
    archive = _bundle(
        tmp_path,
        _manifest(),
        {"thresholds.json": "{}", "metrics.json": "{}", "classical.pkl": ""},
    )
    manifest = adopt_artifacts.verify(_extract(archive, tmp_path / "ok"), allow_smoke=False)
    assert manifest["source"] == "kaggle"


def test_refuses_a_smoke_bundle(tmp_path: Path) -> None:
    archive = _bundle(tmp_path, _manifest(run="smoke"))
    with zipfile.ZipFile(archive) as zf:
        extracted = tmp_path / "out"
        zf.extractall(extracted)
    with pytest.raises(SystemExit) as error:
        adopt_artifacts.verify(extracted, allow_smoke=False)
    assert "smoke" in str(error.value).lower()


def test_accepts_a_smoke_bundle_only_with_the_flag(tmp_path: Path) -> None:
    archive = _bundle(
        tmp_path,
        _manifest(run="smoke"),
        {"thresholds.json": "{}", "metrics.json": "{}", "classical.pkl": ""},
    )
    extracted = _extract(archive, tmp_path / "out2")
    manifest = adopt_artifacts.verify(extracted, allow_smoke=True)
    assert manifest["run"] == "smoke"


def test_refuses_a_foreign_product(tmp_path: Path) -> None:
    archive = _bundle(tmp_path, _manifest(product="OtherThing"))
    extracted = tmp_path / "out3"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extracted)
    with pytest.raises(SystemExit) as error:
        adopt_artifacts.verify(extracted, allow_smoke=True)
    assert "veritas" in str(error.value).lower()


def test_refuses_a_manifest_that_claims_missing_outputs(tmp_path: Path) -> None:
    archive = _bundle(tmp_path, _manifest(steps={"train_classical": {"status": "ok"}}))
    extracted = tmp_path / "out4"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extracted)
    with pytest.raises(SystemExit) as error:
        adopt_artifacts.verify(extracted, allow_smoke=False)
    assert "classical.pkl" in str(error.value)


def test_refuses_a_bundle_without_thresholds(tmp_path: Path) -> None:
    archive = _bundle(tmp_path, _manifest(steps={}), {"metrics.json": "{}"})
    extracted = tmp_path / "out5"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extracted)
    with pytest.raises(SystemExit) as error:
        adopt_artifacts.verify(extracted, allow_smoke=False)
    assert "thresholds.json" in str(error.value)


def test_adopt_keeps_a_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adoption must be reversible: the previous artifacts are moved aside, not deleted."""
    source = tmp_path / "bundle"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(_manifest()))
    (source / "metrics.json").write_text("{}")

    target = tmp_path / "artifacts"
    target.mkdir()
    (target / "old.json").write_text('{"previous": true}')
    backup = tmp_path / "previous"
    monkeypatch.setattr(adopt_artifacts, "ARTIFACT_DIR", target)
    monkeypatch.setattr(adopt_artifacts, "BACKUP_DIR", backup)
    adopt_artifacts.adopt(source, keep_previous=True)
    assert (target / "metrics.json").exists()
    assert (backup / "old.json").exists()
