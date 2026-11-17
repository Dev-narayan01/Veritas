#!/usr/bin/env bash
# One command that runs every gate and reports what passed, what failed and what was skipped.
#
#   bash scripts/finalize.sh            # tests + build + smoke
#   bash scripts/finalize.sh --deploy   # also push the Space (needs .env.deploy)
#
# Exits non-zero on the first failing gate. Skipped gates are listed at the end so a missing
# browser or a missing GPU can never be mistaken for a pass.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
FAILED=0
SKIPPED=()

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

step "python tests"
if ! "$PY" -m pytest tests -q; then FAILED=1; fi

step "ruff"
if ! "$PY" -m ruff check backend ml api tests; then FAILED=1; fi

step "frontend (typecheck, vitest, build, playwright)"
if bash scripts/check_frontend.sh 2>&1 | tee /tmp/veritas-frontend.log | tail -20; then
  :
else
  FAILED=1
fi
grep -q "SKIPPED: no Playwright" /tmp/veritas-frontend.log && SKIPPED+=("Playwright smoke (no browser binary; see docs/HUMAN_STEPS.md §6)")

step "smoke (live API)"
if ! bash scripts/smoke.sh; then FAILED=1; fi

step "honesty checks"
"$PY" - <<'PY'
import json
import sys
from pathlib import Path

artifacts = Path("ml/artifacts")
problems = []
if not (artifacts / "manifest.json").exists():
    problems.append("no artifact bundle: run ml/run_all.py")
else:
    manifest = json.loads((artifacts / "manifest.json").read_text())
    if manifest.get("product") != "Veritas":
        problems.append("manifest product is not Veritas")
    if manifest.get("run") == "smoke":
        print("NOTE: artifacts are from a smoke run; the site will show the banner and")
        print("      adopt_artifacts.py will refuse them. Run ml/run_all.py (no --smoke).")
    modes = manifest.get("modes_fitted") or []
    missing = {"lite", "fast", "standard", "deep"} - set(modes)
    if missing:
        print(f"NOTE: modes not fitted in this bundle: {sorted(missing)} — /api/health will say why")
for name in ("metrics.json", "thresholds.json", "robustness.json", "bias.json"):
    if not (artifacts / name).exists():
        problems.append(f"missing artifact {name}")
if problems:
    print("FAIL: " + "; ".join(problems))
    sys.exit(1)
print("PASS: artifacts present and self-consistent")
PY
[ $? -ne 0 ] && FAILED=1

if [ "${1:-}" = "--deploy" ]; then
  step "deploy Space"
  bash scripts/deploy_space.sh && bash scripts/deploy_space.sh --push || FAILED=1
fi

step "summary"
if [ "$FAILED" -eq 0 ]; then
  echo "ALL GATES PASSED"
else
  echo "SOME GATES FAILED (see above)"
fi
for item in "${SKIPPED[@]:-}"; do
  [ -n "$item" ] && echo "SKIPPED: $item"
done
exit "$FAILED"
