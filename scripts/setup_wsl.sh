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
    warn "================================================================"
    warn " This checkout is under /mnt - a Windows drive seen from Linux."
    warn " Everything works, but file I/O is 10-50x slower, and this script"
    warn " will install ~3 GB of packages onto that slow path."
    warn ""
    warn " Do this instead:"
    warn "   git clone https://github.com/Dev-narayan01/Veritas.git ~/Veritas"
    warn "   cd ~/Veritas && bash scripts/setup_wsl.sh"
    warn ""
    warn " Continuing in 10s. Ctrl-C to stop."
    warn "================================================================"
    sleep 10
    ;;
esac

# --- 1. python --------------------------------------------------------------
# The pins decide this, not preference. Linux wheel tags, checked on PyPI:
#   numpy 1.26.4      cp39 cp310 cp311 cp312          <- no cp313, no cp314
#   pyarrow 17.0.0    cp38 cp39 cp310 cp311 cp312     <- ditto
# So 3.13 and 3.14 have no wheels for two hard dependencies and pip falls back to
# compiling from source, which needs a C compiler a fresh WSL does not have. The
# symptom is a 20-line meson "Unknown compiler(s)" error that looks like a broken
# build rather than a wrong interpreter, which is why this is a hard stop.
say "Resolving an interpreter the pins support (3.10-3.12, prefer 3.12)"

# Keep these in sync with the case statements below: 'case x in $VAR)' does not
# treat | inside a variable as alternation, so the lists cannot be factored out.
PYTHON=""          # system interpreter to build a venv from, if needed
EXISTING_VENV=""   # set when a usable venv already exists, from uv or earlier runs

venv_ver_of() {
  "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none
}

# An existing venv wins over the system interpreter. This is what makes the uv
# route work: uv can build a 3.12 venv on a machine whose only system python is
# 3.14, and re-running this script then picks that venv up instead of refusing.
if [ -x "$VENV/bin/python" ]; then
  vv="$(venv_ver_of "$VENV/bin/python")"
  case "$vv" in
    3.12|3.11|3.10)
      EXISTING_VENV="$VENV"
      echo "reusing existing venv at $VENV ($vv)"
      ;;
    *)
      echo "existing venv is $vv, which the pins do not support; will rebuild"
      ;;
  esac
fi

if [ -z "$EXISTING_VENV" ]; then
  for cand in python3.12 python3.11 python3.10 python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    ver="$(venv_ver_of "$cand")"
    case "$ver" in
      3.12|3.11|3.10)
        PYTHON="$cand"
        echo "using $cand ($ver)"
        break
        ;;
    esac
  done
fi

if [ -z "$EXISTING_VENV" ] && [ -z "$PYTHON" ]; then
  have="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none)"
  warn "No supported Python found. This system's python3 is $have."
  warn ""
  warn "numpy 1.26.4 and pyarrow 17.0.0 have wheels only up to cp312, so $have"
  warn "would compile both from source. That is the meson error you just saw."
  warn ""
  warn "Option A - install 3.12 from the distro:"
  warn "  sudo apt update && sudo apt install -y python3.12 python3.12-venv"
  warn ""
  warn "Option B - uv, which downloads its own CPython and needs nothing from apt."
  warn "           This works even on very new releases where apt has no 3.12:"
  warn "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  warn "  source \$HOME/.local/bin/env"
  warn "  uv venv --python 3.12 $VENV"
  warn "  bash scripts/setup_wsl.sh          # picks the venv up and continues"
  warn ""
  exit 1
fi

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
if [ -n "$EXISTING_VENV" ]; then
  say "Using existing venv at $EXISTING_VENV"
else
  say "Creating venv at $VENV"
  # A venv built by an unsupported interpreter cannot be upgraded in place.
  [ -e "$VENV" ] && rm -rf "$VENV"
  "$PYTHON" -m venv "$VENV" || {
    warn "venv failed. Missing the venv module for $("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])'):"
    warn "  sudo apt install -y python3.12-venv"
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
