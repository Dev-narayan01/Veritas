#!/usr/bin/env python3
"""Inline the Lite core into each Vercel function, producing self-contained entrypoints.

Why this exists
---------------
Vercel's Python builder packages a function as *its entrypoint file alone*. Neither a
sibling package (``api/_lib/``) nor a sibling module (``api/_core.py``) is uploaded, so
``from _lib.core import ...`` and ``from _core import ...`` both fail at request time:

    ModuleNotFoundError: No module named '_core'

The build log stays green either way, which is what made this hard to see. The layout the
builder does handle reliably is one self-contained file per function.

So ``api/_lib/core.py`` stays the single source of truth (readable, testable, imported
directly by the test suite), and ``api/handlers/<name>.py`` holds the request handling for
each endpoint. This script splices the two into ``api/<name>.py``, the committed
entrypoint that Vercel actually ships.

Run ``python scripts/build_api.py`` after editing anything under ``api/_lib`` or
``api/handlers``. ``tests/test_lite_deployment.py`` fails if the committed output is stale.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "api"
CORE = API / "_lib" / "core.py"
HANDLERS = API / "handlers"
FUNCTIONS = ("health", "samples", "metrics", "analyze")

BANNER = '"""GENERATED FILE - do not edit.\n\nBuilt by ``python scripts/build_api.py`` from ``api/_lib/core.py`` plus\n``api/handlers/{name}.py``. Vercel ships a function as this one file, so the shared core is\ninlined rather than imported. Edit the sources and re-run the script.\n"""\n'

_STDLIB_IMPORT = re.compile(r"^(?:import [\w.]+|from [\w.]+ import [^()\n]+)$")


def _strip_docstring(source: str) -> str:
    if not source.startswith('"""'):
        return source
    end = source.find('"""', 3)
    return source[end + 3 :].lstrip("\n") if end != -1 else source


def _split_imports(source: str) -> tuple[list[str], str]:
    """Separate the module's imports from its body, so the two can be merged."""
    imports: list[str] = []
    body: list[str] = []
    for line in source.splitlines():
        if not body and (
            _STDLIB_IMPORT.match(line) or line.startswith(("import ", "from ")) or not line.strip()
        ):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                imports.append(line)
            continue
        body.append(line)
    return imports, "\n".join(body).lstrip("\n")


def _merge_imports(groups: list[list[str]]) -> str:
    """Union the import lines, keeping ``__future__`` first and the rest sorted."""
    seen: dict[str, None] = {}
    futures: dict[str, None] = {}
    for group in groups:
        for line in group:
            target = futures if line.startswith("from __future__") else seen
            target.setdefault(line, None)
    ordered = list(futures) + [""] + sorted(seen)
    while ordered and not ordered[-1].strip():
        ordered.pop()
    return "\n".join(ordered)


def build(name: str) -> str:
    core = CORE.read_text(encoding="utf-8")
    handler = (HANDLERS / f"{name}.py").read_text(encoding="utf-8")
    core_imports, core_body = _split_imports(_strip_docstring(core))
    handler_imports, handler_body = _split_imports(_strip_docstring(handler))
    return (
        BANNER.format(name=name)
        + "\n"
        + _merge_imports([core_imports, handler_imports])
        + "\n\n"
        + core_body
        + "\n\n"
        + "# --------------------------------------------------------------------------\n"
        + f"# {name} endpoint\n"
        + "# --------------------------------------------------------------------------\n\n"
        + handler_body.rstrip("\n")
        + "\n"
    )


def main() -> int:
    if not CORE.is_file():
        print(f"missing {CORE}", file=sys.stderr)
        return 1
    names = [p.stem for p in sorted(HANDLERS.glob("*.py"))] if HANDLERS.is_dir() else list(FUNCTIONS)
    for name in names:
        target = API / f"{name}.py"
        target.write_text(build(name), encoding="utf-8")
        print(f"built api/{name}.py ({target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
