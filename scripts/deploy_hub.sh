#!/usr/bin/env bash
# Publish the artifact bundle to the Hugging Face Hub model repo
# (<HF_USERNAME>/veritas-detector) and print the URL. Needs HF_TOKEN in .env.deploy.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env.deploy ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.deploy
  set +a
fi

: "${HF_USERNAME:?set HF_USERNAME in .env.deploy}"
: "${HF_TOKEN:?set HF_TOKEN in .env.deploy}"
REPO="${VERITAS_HUB_REPO:-$HF_USERNAME/veritas-detector}"
ARCHIVE="${ARCHIVE:-dist/veritas_artifacts.zip}"
[ -f "$ARCHIVE" ] || { echo "missing $ARCHIVE — run ml/run_all.py first" >&2; exit 1; }

.venv/bin/python - "$ARCHIVE" "$REPO" <<'PY'
import sys

from huggingface_hub import HfApi

archive, repo_id = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=archive, path_in_repo="veritas_artifacts.zip", repo_id=repo_id, repo_type="model")
print(f"https://huggingface.co/{repo_id}")
PY
