#!/usr/bin/env bash
# Deploy the frontend to Vercel from inside frontend/.
#
#   scripts/deploy_vercel.sh            # deploy a preview
#   scripts/deploy_vercel.sh --prod     # deploy to production
#
# The project keeps the existing Vercel project name (`veritas`); the old project and the
# old Space are never deleted. VITE_API_BASE_URL must point at the Space URL, or the site
# will run in Lite mode against api/*.py.
set -euo pipefail
cd "$(dirname "$0")/../frontend"

if [ -f ../.env.deploy ]; then
  set -a
  # shellcheck disable=SC1091
  . ../.env.deploy
  set +a
fi

: "${VITE_API_BASE_URL:?set VITE_API_BASE_URL in .env.deploy (the Space URL)}"
echo "building with VITE_API_BASE_URL=$VITE_API_BASE_URL"

npm ci
npm run build

if command -v vercel >/dev/null 2>&1; then
  if [ "${1:-}" = "--prod" ]; then
    vercel deploy --prod --yes --cwd .
  else
    vercel deploy --yes --cwd .
  fi
else
  echo "the vercel CLI is not installed here; the build in dist/ is ready to upload" >&2
  echo "install it with 'npm i -g vercel' (or deploy through the Vercel dashboard)" >&2
  exit 1
fi
