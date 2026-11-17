#!/usr/bin/env bash
# Frontend gate: typecheck, unit tests, production build, and (when a browser is
# installed) the Playwright smoke suite. A skipped browser is reported as SKIPPED so a
# missing browser can never be mistaken for a passing end-to-end run.
set -uo pipefail
cd "$(dirname "$0")/../frontend"

echo "== typecheck =="
npm run typecheck || exit 1

echo "== unit tests =="
npm run test || exit 1

echo "== build =="
npm run build || exit 1

echo "== e2e =="
if compgen -G "${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}/chromium-*" > /dev/null; then
  npx playwright test || exit 1
else
  echo "SKIPPED: no Playwright chromium in \${PLAYWRIGHT_BROWSERS_PATH:-~/.cache/ms-playwright}."
  echo "         Install it locally with 'cd frontend && npx playwright install chromium'."
fi
