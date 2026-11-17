"""Shared pytest fixtures.

Each test session runs against a throwaway data directory and database so the real
``data/`` tree (uploads, history) is never touched, and the artifacts in ``ml/artifacts``
are used read-only through the normal registry.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

BUNDLED = ROOT / "ml" / "bundled" / "samples"


@pytest.fixture(scope="session")
def app_env(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Point the settings at a temporary data directory before the app is imported."""
    data_dir = tmp_path_factory.mktemp("veritas-data")
    os.environ["VERITAS_DATA_DIR"] = str(data_dir)
    os.environ["VERITAS_DB_URL"] = f"sqlite:///{(data_dir / 'test.sqlite3').as_posix()}"
    os.environ.setdefault("VERITAS_ARTIFACT_DIR", "ml/artifacts")
    return data_dir


@pytest.fixture(scope="session")
def client(app_env: Path) -> Iterator["object"]:
    """A TestClient with the app's lifespan executed (artifacts + DB ready)."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def human_text() -> str:
    """A bundled human sample that is long enough to analyse."""
    return (BUNDLED / "sample_human.txt").read_text()


@pytest.fixture()
def ai_text() -> str:
    """A bundled AI-style sample that is long enough to analyse."""
    return (BUNDLED / "sample_ai.txt").read_text()


@pytest.fixture()
def tmp_copy(tmp_path: Path) -> "callable":
    """Copy a file into the test's temp dir and return the new path."""

    def copy(source: Path, name: str | None = None) -> Path:
        target = tmp_path / (name or source.name)
        shutil.copyfile(source, target)
        return target

    return copy
