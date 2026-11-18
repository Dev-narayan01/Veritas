#!/usr/bin/env bash
# One-shot Linux/WSL2 environment for the ModernBERT fine-tune.
#
# Why this exists: on Windows, Smart App Control refuses to load PyTorch's unsigned
# CUDA DLLs (WinError 4551, policy {0283AC0F-FFF1-49AE-ADA1-8A933130CAD6}), and
# reinstalling does not clear it. Linux has no such policy, and the PyPI wheel there
# is the CUDA build (539 MB) rather than the CPU build Windows gets (127 MB).
#
# The Windows NVIDIA driver already provides CUDA to WSL2, so no driver install here.
#
#   bash scripts/setup_wsl.sh
#
# Safe to re-run: it reuses the venv and only reinstalls torch if CUDA is missing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-.venv}"
say() { printf '\n=== %s\n' "$*"; }
warn() { printf '!! %s\n' "$*" >&2; }

# --- 0. refuse the slow filesystem ------------------------------------------
# /mnt/c goes through a translation layer: PyTorch imports crawl and dataloading
# stalls. It works, it is just 10-50x slower, so say it once and carry on.
case "$REPO_ROOT" in
  /mnt/*)
    warn "This checkout is on $REPO_ROOT (a Windows drive mounted into Linux)."
    warn "It works, but I/O is very slow. For real training clone inside Linux:"
    warn "  git clone https://github.com/Dev-narayan01/Veritas.git ~/Veritas"
    ;;
esac

# --- 1. python --------------------------------------------------------------
say "Checking Python"
if ! command -v python3 >/dev/null; then
  warn "python3 not found. sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
  exit 1
fi
PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "python3 $PY_VER"
case "$PY_VER" in
  3.10|3.11|3.12|3.13) ;;
  *) warn "Python $PY_VER is untested here. 3.12 is what the pins were verified against." ;;
esac

# --- 2. GPU visible? --------------------------------------------------------
# Check this before installing 500 MB of torch: if the driver is not passing the
# GPU through, no Python package will fix it.
say "Checking GPU passthrough"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  warn "nvidia-smi is not reporting a GPU inside WSL."
  warn "On Windows: WSL2 needs a recent NVIDIA driver (>= 470). Check 'nvidia-smi.exe' in PowerShell."
  warn "Continuing anyway - install will succeed, training will not."
fi

# --- 3. venv ----------------------------------------------------------------
say "Creating/reusing venv at $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" || {
    warn "venv failed. Usually missing: sudo apt install -y python3-venv"
    exit 1
  }
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip

# --- 4. dependency order matters -------------------------------------------
# ml/requirements.txt pulls sentence-transformers, which depends on torch. On
# Windows that pins the CPU wheel in place. On Linux the PyPI wheel IS the CUDA
# build, so the order is harmless here - but it is kept consistent with the docs.
say "Installing backend requirements"
python -m pip install --quiet -r backend/requirements-dev.txt

say "Installing ml requirements (this pulls torch)"
python -m pip install --quiet -r ml/requirements.txt

say "Installing ModernBERT-capable transformers"
python -m pip install --quiet "transformers==4.49.0" accelerate sentencepiece

# --- 5. verify, and repair only if needed ----------------------------------
say "Verifying torch CUDA"
CUDA_OK="$(python - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
PY
)"

if [ "$CUDA_OK" != "1" ]; then
  warn "torch is importable but CUDA is unavailable. Reinstalling torch."
  python -m pip uninstall -y --quiet torch
  python -m pip install --quiet torch
  CUDA_OK="$(python - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
PY
)"
fi

if [ "$CUDA_OK" = "1" ]; then
  python - <<'PY'
import torch, transformers
print(f"torch         {torch.__version__}")
print(f"cuda build    {torch.version.cuda}")
print(f"available     {torch.cuda.is_available()}")
print(f"device        {torch.cuda.get_device_name(0)}")
print(f"vram          {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
print(f"transformers  {transformers.__version__}")
PY
else
  warn "CUDA still unavailable after reinstall. Something is wrong at the driver"
  warn "level, not the Python level. Check 'nvidia-smi' on the Windows side."
  warn "You can still smoke-test with: python ml/train_modernbert.py --allow-cpu ..."
fi

# --- 6. caches --------------------------------------------------------------
cat <<'EOF'

=== Cache locations
Set these in this shell before downloading data (add to ~/.bashrc to persist).
Keep them INSIDE Linux: /mnt/... would be slow, and your D: drive is a Windows path.

  export HF_HOME="$HOME/hf_cache"
  export TRANSFORMERS_CACHE="$HOME/hf_cache/transformers"
  export HF_DATASETS_CACHE="$HOME/hf_cache/datasets"

=== Next steps
  source .venv/bin/activate
  python ml/prepare_data.py
  python ml/train_modernbert.py --limit 200 --epochs 1 --max-len 512 --batch-size 2 \
      --out ml/artifacts/modernbert_smoke
EOF
