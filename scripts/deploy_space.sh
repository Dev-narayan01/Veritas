#!/usr/bin/env bash
# Prepare (and optionally push) the Hugging Face Space deployment.
#
#   scripts/deploy_space.sh                 # build the staging folder only
#   scripts/deploy_space.sh --push          # push with huggingface_hub (needs HF_TOKEN)
#
# Reads HF_USERNAME, SPACE_NAME (default veritas-api), HF_TOKEN and VERITAS_HUB_REPO from
# the environment or .env.deploy (gitignored). Never prints a token.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env.deploy ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.deploy
  set +a
fi

: "${HF_USERNAME:?set HF_USERNAME in .env.deploy}"
SPACE_NAME="${SPACE_NAME:-veritas-api}"
ARCHIVE="${ARCHIVE:-dist/veritas_artifacts.zip}"
STAGE="dist/space"

if [ ! -f "$ARCHIVE" ]; then
  echo "missing $ARCHIVE — run 'python ml/run_all.py' first (or download it from the Hub)" >&2
  exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE"
cp backend/Dockerfile "$STAGE/Dockerfile"
cp backend/requirements.txt "$STAGE/requirements.txt"
cp backend/space_README.md "$STAGE/README.md"
cp -r backend/app "$STAGE/app"
mkdir -p "$STAGE/ml"
cp -r ml/bundled "$STAGE/ml/bundled"

# Unpack the artifact bundle so the Space boots without a Hub download.
mkdir -p "$STAGE/artifacts"
mkdir -p "$STAGE/tmp" && unzip -q -o "$ARCHIVE" -d "$STAGE/tmp"
if [ -f "$STAGE/tmp/manifest.json" ]; then
  cp -r "$STAGE/tmp/." "$STAGE/artifacts/"
elif [ -d "$STAGE/tmp/artifacts" ]; then
  cp -r "$STAGE/tmp/artifacts/." "$STAGE/artifacts/"
else
  echo "the bundle does not look like an artifact directory" >&2
  exit 1
fi
rm -rf "$STAGE/tmp"

cat > "$STAGE/.env.example" <<'ENV'
# Copy to the Space's Settings ▸ Variables and secrets. Never commit real values.
VERITAS_ARTIFACT_DIR=/app/artifacts
VERITAS_DATA_DIR=/data/state
VERITAS_HUB_REPO=
CORS_ORIGINS=https://veritas.vercel.app,http://localhost:5173
ENV

echo "staged Space files in $STAGE ($(du -sh "$STAGE" | cut -f1))"
echo "artifact source: $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("source"))' "$STAGE/artifacts/manifest.json" 2>/dev/null || echo unknown)"

if [ "${1:-}" != "--push" ]; then
  echo "dry run: re-run with --push to upload to $HF_USERNAME/$SPACE_NAME"
  exit 0
fi

: "${HF_TOKEN:?set HF_TOKEN in .env.deploy to push}"
.venv/bin/python - "$STAGE" "$HF_USERNAME/$SPACE_NAME" <<'PY'
import sys
from huggingface_hub import HfApi

folder, repo_id = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="space")
print(f"pushed to https://huggingface.co/spaces/{repo_id}")
PY
