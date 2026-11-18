#!/usr/bin/env bash
# WSL2 / Linux setup for the ModernBERT fine-tune, from a bare Ubuntu to a passing
# smoke test. Handles the four things that actually break on a fresh machine:
#
#   1. main is a placeholder - it holds only README.md, and the code is on a branch.
#      A plain `git clone` therefore gives you one file and every script "is missing".
#   2. WSL now ships Python 3.14, and numpy 1.26.4 / pyarrow 17.0.0 have no wheels
#      past cp312, so pip falls back to compiling and dies with "Unknown compiler(s)".
#   3. /mnt/c is a Windows drive seen from Linux: correct, but 10-50x slower on I/O.
#   4. data/processed/ is gitignored, so no clone has it; ml/prepare_data.py builds it.
#
# Usage, from anywhere:
#   bash setup_wsl.sh                 # clone/update ~/Veritas and set it up
#   bash setup_wsl.sh --dir ~/work/V  # somewhere else
#   bash setup_wsl.sh --in-place      # already in a checkout; just configure it
#
# Safe to re-run.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Dev-narayan01/Veritas.git}"
# main holds only README.md, so the branch is not optional.
BRANCH="${BRANCH:-arena/772d7f78-veritas}"
TARGET="${HOME}/Veritas"
IN_PLACE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) TARGET="$2"; shift 2 ;;
    --in-place) IN_PLACE=1; shift ;;
    --branch) BRANCH="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\n=== %s\n' "$*"; }
warn() { printf '!! %s\n' "$*" >&2; }

# --- 0. where are we ---------------------------------------------------------
say "Target: $TARGET"

case "$TARGET" in
  /mnt/*)
    warn "Target is under /mnt, a Windows drive seen from Linux."
    warn "It works but I/O is 10-50x slower. Prefer the Linux filesystem (\$HOME)."
    ;;
esac

# --- 1. clone or update ------------------------------------------------------
# The branch matters more than the clone: main has one file.
if [ "$IN_PLACE" = "1" ]; then
  if [ ! -d .git ]; then
    warn "--in-place given but $(pwd) is not a git checkout"
    exit 1
  fi
  TARGET="$(pwd)"
  say "Configuring existing checkout at $TARGET"
else
  if [ -d "$TARGET/.git" ]; then
    say "Updating existing checkout"
    git -C "$TARGET" fetch --quiet origin
    git -C "$TARGET" switch --quiet "$BRANCH"
    git -C "$TARGET" pull --quiet --ff-only origin "$BRANCH"
  else
    if [ -d "$TARGET" ]; then
      # Usually a clone made before the --branch fix, holding only README.md.
      warn "$TARGET exists but is not a git checkout. Removing it."
      rm -rf "$TARGET"
    fi
    say "Cloning $BRANCH (main only has README.md, so the branch is explicit)"
    git clone --quiet --branch "$BRANCH" "$REPO_URL" "$TARGET"
  fi
fi

cd "$TARGET"

# Prove we got code rather than the placeholder. This is the check that turns a
# confusing "No such file or directory" 30 seconds later into one clear sentence.
if [ ! -f scripts/setup_wsl.sh ] || [ ! -d ml ]; then
  warn "This checkout has no ml/ or scripts/ - it is the placeholder tree."
  warn "Current branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  warn "Fix: git switch $BRANCH"
  exit 1
fi
echo "branch $(git rev-parse --abbrev-ref HEAD), $(git rev-parse --short HEAD)"

# --- 2. an interpreter the pins support --------------------------------------
# Linux wheel tags, checked on PyPI:
#   numpy 1.26.4    cp39 cp310 cp311 cp312
#   pyarrow 17.0.0  cp38 cp39 cp310 cp311 cp312
# 3.13 and 3.14 have neither, so pip compiles from source and there is no compiler.
say "Resolving an interpreter the pins support (3.10-3.12, prefer 3.12)"

venv_ver_of() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none; }

# uv's installer does not reliably write ~/.local/bin/env, so find the binary and
# put its directory on PATH instead of sourcing a file that may not exist.
UV_BIN=""
find_uv() {
  [ -n "$UV_BIN" ] && return 0
  for c in uv "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if command -v "$c" >/dev/null 2>&1; then UV_BIN="$(command -v "$c")"; return 0; fi
    if [ -x "$c" ]; then UV_BIN="$c"; return 0; fi
  done
  command -v curl >/dev/null 2>&1 || return 1
  echo "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
  for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    [ -x "$c" ] && { UV_BIN="$c"; return 0; }
  done
  return 1
}

PYTHON=""
EXISTING_VENV=""
if [ -x ".venv/bin/python" ]; then
  vv="$(venv_ver_of ".venv/bin/python")"
  case "$vv" in
    3.12|3.11|3.10)
      EXISTING_VENV=".venv"
      echo "reusing existing .venv ($vv)"
      ;;
    *) echo "existing .venv is $vv, which the pins do not support; will rebuild" ;;
  esac
fi

if [ -z "$EXISTING_VENV" ]; then
  for cand in python3.12 python3.11 python3.10 python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    ver="$(venv_ver_of "$cand")"
    case "$ver" in
      3.12|3.11|3.10) PYTHON="$cand"; echo "using $cand ($ver)"; break ;;
    esac
  done
fi

# uv can build a supported CPython on a system whose only python3 is 3.14, which is
# the normal case on current Ubuntu. Install it if the distro has nothing usable.
if [ -z "$EXISTING_VENV" ] && [ -z "$PYTHON" ]; then
  have="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none)"
  say "No usable system Python (found $have); using uv"

  find_uv
  # --seed puts pip into the venv. uv venvs have no pip by default, and the next
  # few lines call 'python -m pip', which would die with "No module named pip".
  "$UV_BIN" venv --seed --python 3.12 .venv
  EXISTING_VENV=".venv"
  echo "created .venv (seeded with pip)"
fi

if [ -n "$EXISTING_VENV" ]; then
  say "Activating $EXISTING_VENV"
else
  say "Creating .venv with $PYTHON"
  [ -e .venv ] && rm -rf .venv
  "$PYTHON" -m venv .venv || {
    warn "venv failed; install the venv module for $PYTHON:"
    warn "  sudo apt install -y ${PYTHON}-venv"
    exit 1
  }
fi

# shellcheck disable=SC1091
. .venv/bin/activate
echo "python $(python --version 2>&1)"

# A venv can legitimately have no pip: uv builds them that way, and some distros
# ship a python3-venv without ensurepip. Detect instead of assuming, because the
# failure mode ("No module named pip") reads like a broken environment.
PIP_MODE=""
if python -m pip --version >/dev/null 2>&1; then
  PIP_MODE="pip"
elif python -m ensurepip --upgrade >/dev/null 2>&1 && python -m pip --version >/dev/null 2>&1; then
  PIP_MODE="pip"
  echo "pip was missing; installed it with ensurepip"
elif find_uv; then
  PIP_MODE="uv"
  echo "pip is unavailable in this venv; installing with uv pip instead"
else
  warn "No pip in .venv and no uv to fall back on."
  warn "Recreate it: uv venv --seed --python 3.12 .venv"
  exit 1
fi

# Single entry point so the rest of the script does not care which one is in use.
pip_install() {
  case "$PIP_MODE" in
    pip) python -m pip install --quiet "$@" ;;
    uv)  "$UV_BIN" pip install --quiet --python "$PWD/.venv/bin/python" "$@" ;;
  esac
}

[ "$PIP_MODE" = "pip" ] && python -m pip install --quiet --upgrade pip || true

# --- 3. GPU before a 500 MB download ----------------------------------------
# Discovering the driver is not passing the GPU through after installing torch is
# the waste this ordering prevents.
say "Checking GPU passthrough"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  warn "nvidia-smi does not report a GPU inside WSL."
  warn "Needs a recent NVIDIA Windows driver (>= 470). Check nvidia-smi.exe in PowerShell."
fi

# --- 4. dependencies --------------------------------------------------------
say "Installing backend requirements"
pip_install -r backend/requirements-dev.txt

say "Installing ml requirements"
pip_install -r ml/requirements.txt

say "Installing ModernBERT-capable transformers"
pip_install "transformers==4.49.0" accelerate sentencepiece

# --- 5. verify torch --------------------------------------------------------
say "Verifying torch"
CUDA_OK="$(python -c 'import torch; print(int(torch.cuda.is_available()))' 2>/dev/null || echo import_failed)"
if [ "$CUDA_OK" != "1" ]; then
  warn "CUDA not available (status: $CUDA_OK). Reinstalling torch from PyPI."
  case "$PIP_MODE" in
    pip) python -m pip uninstall -y --quiet torch || true ;;
    uv)  "$UV_BIN" pip uninstall --python "$PWD/.venv/bin/python" torch >/dev/null 2>&1 || true ;;
  esac
  pip_install torch
  CUDA_OK="$(python -c 'import torch; print(int(torch.cuda.is_available()))' 2>/dev/null || echo import_failed)"
fi

if [ "$CUDA_OK" = "1" ]; then
  python - <<'PY'
import torch, transformers
print(f"torch         {torch.__version__}")
print(f"cuda build    {torch.version.cuda}")
print(f"device        {torch.cuda.get_device_name(0)}")
print(f"vram          {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
print(f"transformers  {transformers.__version__}")
PY
else
  warn "CUDA unavailable after reinstall - that is a driver issue, not Python."
  warn "Smoke test will still work: python ml/train_modernbert.py --allow-cpu ..."
fi

# --- 6. data ----------------------------------------------------------------
# data/processed/ is gitignored, so no clone contains it. Training fails with
# FileNotFoundError until prepare_data has run once.
say "Checking data/processed"
if [ -f data/processed/train.parquet ]; then
  echo "already present"
else
  echo "not present (it is gitignored, so no clone contains it)"
  echo "build it with: python ml/prepare_data.py"
fi

cat <<EOF

=== Done. Next:
  cd $TARGET
  source .venv/bin/activate
  export HF_HOME="\$HOME/hf_cache"      # keep caches on Linux storage, not /mnt/d
  python ml/prepare_data.py             # ~1 min, writes data/processed/
  python ml/train_modernbert.py --limit 200 --epochs 1 --max-len 512 --batch-size 2 \\
      --out ml/artifacts/modernbert_smoke
EOF
