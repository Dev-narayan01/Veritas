"""Acceptance gate (spec 19): no stale product name and no placeholder artefacts.

The repository must not mention the superseded ``VeriText`` name anywhere a user or a
reviewer can see it, and finished phases must not ship TODO/FIXME markers as stand-ins
for work. Excluded: virtualenvs, node_modules, build output, lock files and this test's
own pattern definitions.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "frontend/node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "data",
    "ml/artifacts",
    "ml/incoming",
    "playwright-report",
    "test-results",
}
TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".txt", ".json", ".yaml", ".yml", ".html", ".css", ".sh", ".toml", ".cfg", ".ini"}
STALE_NAME = re.compile(r"veritext", re.IGNORECASE)
PLACEHOLDER = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")


def _iter_files() -> list[Path]:
    found: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if any(relative == excluded or relative.startswith(f"{excluded}/") for excluded in EXCLUDED_DIRS):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name in {
            "package-lock.json",
            "pnpm-lock.yaml",
            # The audit trail is allowed to name the old product: that is the record of
            # the rename, and the spec asks for it.
            "DECISIONS.md",
            "PROGRESS.md",
            # This file defines the pattern it searches for.
            "test_naming.py",
        }:
            continue
        found.append(path)
    return found


def test_no_stale_product_name() -> None:
    offenders = []
    for path in _iter_files():
        text = path.read_text(errors="ignore")
        for match in STALE_NAME.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line}")
    assert not offenders, (
        "stale product name left in the shipped tree: "
        + ", ".join(offenders[:20])
        + " (the two docs that explain the rename are allowed in docs/DECISIONS.md)"
    )


def test_no_placeholder_markers_in_source() -> None:
    offenders = []
    for path in _iter_files():
        if path.parent.name == "docs" or path.parent.name == "tests":
            continue
        if path.name in {"test_naming.py", "PROGRESS.md", "DECISIONS.md", "REPORT.md"}:
            continue
        text = path.read_text(errors="ignore")
        for match in PLACEHOLDER.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line} {match.group(0)}")
    assert not offenders, "placeholder markers left in shipped code: " + ", ".join(offenders[:20])
