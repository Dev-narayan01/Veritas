# Windows setup (PowerShell)

The failure everyone hits first on Windows is not the repo — it is **Python 3.15**.

```
Preparing metadata (pyproject.toml) ... error
Unknown compiler(s): [['icl'], ['cl'], ['cc'], ...
error: metadata-generation-failed
╰─> numpy
```

That is pip saying: *there is no pre-built wheel for your Python version, so I tried to
compile numpy from source, and you have no C compiler.* The pins in this repo are not at
fault; Python 3.15 is newer than any of them.

## Wheel support, checked against PyPI

| package | supports | Windows wheel for 3.12 | 3.13 | 3.15 |
|---|---|---|---|---|
| `numpy==1.26.4` | 3.9–3.12 | yes | **no** | **no** |
| `pandas==2.2.3` | 3.9–3.12 | yes | no | no |
| `scikit-learn==1.5.2` | 3.9–3.12 | yes | no | no |
| `torch` (latest 2.15.0) | 3.11–3.15 | yes | yes | yes |
| `transformers==4.49.0` | pure python | yes | yes | yes |

So: **use Python 3.12.** Every pin in `backend/requirements.txt` (23 of them, verified)
installs as a wheel on cp312-win_amd64 with no compiler. 3.13 breaks numpy/pandas/sklearn
because those exact pins predate it.

## 1. Install Python 3.12

```powershell
winget install -e --id Python.Python.3.12
```

Confirm the launcher sees it (you should get 3.12.x, not 3.15):

```powershell
py -0p
py -3.12 --version
```

`py -0p` lists every interpreter with its path. If 3.15 is listed first, that is fine —
we will address 3.12 explicitly everywhere.

## 2. Create the venv with 3.12, not `python`

This is the step that matters. `python -m venv` used your default interpreter, which was
3.15 — that is why the venv inherited the problem.

```powershell
cd C:\Users\DEV NARAYAN\Downloads\llm_final\Veritas

# delete the broken venv from before
Remove-Item -Recurse -Force .venv

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version          # must print Python 3.12.x
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. PowerShell has no `&&`

Every `&&` in the docs is a bash-ism and this PowerShell version rejects it. Use `;`
to run two commands in sequence:

```powershell
# instead of: pip install -r a && pip install -r b
python -m pip install -r backend/requirements-dev.txt
python -m pip install -r ml/requirements.txt
```

(Or use `pwsh` 7+, where `&&` works.)

## 4. `pip` is blocked — always use `python -m pip`

Your machine has an Application Control policy blocking `pip.exe`:

```
Program 'pip.exe' failed to run: An Application Control policy has blocked this file
```

You already worked around it correctly. Keep doing that; every command below uses
`python -m pip`. No need to fight the policy.

## 5. Install the CPU stack

```powershell
python -m pip install --upgrade pip
python -m pip install -r backend/requirements-dev.txt
python -m pip install -r ml/requirements.txt
```

Verify the interpreter actually got the wheels:

```powershell
python -c "import numpy, pandas, sklearn, pyarrow; print(numpy.__version__, pandas.__version__, sklearn.__version__)"
```

Expect `1.26.4 2.2.3 1.5.2`.

## 6. GPU stack

`pip install torch --index-url .../whl/cu121` first failed with **"from versions: none"**
because that venv was Python 3.15, and `download.pytorch.org` has no cp315 build. Under
3.12 the same index resolves (`torch 2.14.1+cu126`) — and you do not need cu121
specifically; match the tag to your driver.

**But resolving is not the same as loading.** On a machine with Smart App Control
enforced, the cu126 wheel downloads 2.6 GB and then `import torch` dies with
`WinError 4551 ... Application Control policy has blocked this file`. That is section 6b,
and it is the reason this whole page warns you to verify with `import torch` and not just
with pip's success message.

### PyPI's Windows `torch` is CPU-only — this catches everyone

Measured wheel sizes for torch 2.15.0:

| wheel | size |
|---|---|
| `torch-2.15.0-cp312-cp312-win_amd64.whl` (PyPI) | **121 MB** |
| `torch-2.15.0-cp312-cp312-manylinux_2_28_x86_64.whl` | **514 MB** |

A CUDA build is ~400 MB bigger because it bundles the CUDA runtime. On **Linux**, PyPI's
`torch` is the CUDA build; on **Windows** it is the CPU build. Installing from PyPI gets
you a torch that reports `Torch not compiled with CUDA enabled`.

Worse, `ml/requirements.txt` pulls `sentence-transformers`, which depends on `torch`, so
that install puts the **CPU wheel** in first. A later `pip install torch --upgrade` then
says "already satisfied" and changes nothing.

**Correct order — note that torch comes last:**

```powershell
python -m pip install -r backend/requirements-dev.txt
python -m pip install -r ml/requirements.txt          # pulls CPU torch, on purpose
python -m pip install "transformers==4.49.0" accelerate sentencepiece

# now replace the CPU torch with a CUDA build
python -m pip uninstall -y torch
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Pick the index from `nvidia-smi` (top right shows `CUDA Version: 12.x`). If the index
has no matching build:

```powershell
python -m pip index versions torch --index-url https://download.pytorch.org/whl/cu126
```

then move to `cu124` / `cu128`, or loosen the version.

### Verify the build, not just the import

`torch.cuda.get_device_name(0)` raises `AssertionError: Torch not compiled with CUDA
enabled` when you have the CPU wheel. Check the build attribute first — it fails fast and
says why:

```powershell
python -c "import torch; print('version:', torch.__version__); print('cuda build:', torch.version.cuda); print('available:', torch.cuda.is_available())"
```

You need `cuda build: 12.x` (**not `None`**) and `available: True`.
`cuda build: None` means the CPU wheel is still installed. If `import torch` raises
`WinError 4551` instead, stop — that is section 6b, and reinstalling will not help.

Then check the GPU is visible — but only *after* `import torch` is known to succeed, or you
will read a code-integrity block as a missing GPU:

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

You want `NVIDIA GeForce RTX 3050`.

### 6b. If `import torch` fails with WinError 4551

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading "...\torch\lib\c10_cuda.dll" or one of its dependencies.
```

This is **Smart App Control** (policy `{0283AC0F-FFF1-49AE-ADA1-8A933130CAD6}`,
CodeIntegrity event 3077). It requires DLLs to be signed or Microsoft-reputed, and
PyTorch ships several unsigned DLLs in `torch/lib`. Upstream: `pytorch/pytorch#199711`.

Reinstalling cannot fix it, and changing the CUDA tag will not either — the policy judges
each file by hash. Which files pass is not predictable: on the machine this was measured
on, the **PyPI CPU wheel imported fine** and the **cu126 wheel did not**.

Confirm:

```powershell
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' | Select-Object VerifiedAndReputablePolicyState
# 1 = enforcing, 0 = off, 2 = evaluation

Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-CodeIntegrity/Operational'; Id=3077} -MaxEvents 5 | Format-List TimeCreated, Message
```

`CiTool.exe --list-policies` returns access denied without elevation, so the registry
value and the event log are the practical checks.

Your options:

| option | keeps SAC on? | cost |
|---|---|---|
| **WSL2** (recommended) | yes | clone into `~/Veritas`, not `/mnt/c/...` — cross-filesystem I/O is slow |
| Turn SAC off | no, and **it cannot be turned back on without reinstalling Windows** | Windows Security ▸ App and browser control |
| `--allow-cpu` | yes | hours instead of minutes; fine for a 200-row smoke test |

If the machine is managed by an employer or school, the policy is theirs; ask for an
exclusion rather than working around it.

### 6c. WSL2 — the option that keeps Smart App Control on

Linux has no Smart App Control, and the PyPI wheel there is a different artifact:

| wheel | size | CUDA |
|---|---|---|
| `torch-2.15.0-cp312-cp312-manylinux_2_28_x86_64.whl` | **539 MB** | yes |
| `torch-2.15.0-cp312-cp312-win_amd64.whl` | **127 MB** | no |

So on Linux a plain `python -m pip install torch` gives you the GPU build with no index
URL and no 2.6 GB download. WSL2 also gets CUDA from the Windows driver, so there is no
driver to install inside Linux.

Check WSL is present, then open it:

```powershell
wsl --status
wsl
```

Inside Linux, **clone into the Linux filesystem, not `/mnt/c`**. `/mnt/c` works but goes
through a translation layer that makes torch imports extremely slow and stalls
dataloading. It is the single biggest performance mistake available here.

```bash
sudo apt update && sudo apt install -y python3 python3-venv git
git clone https://github.com/Dev-narayan01/Veritas.git ~/Veritas
cd ~/Veritas
bash scripts/setup_wsl.sh
```

`scripts/setup_wsl.sh` checks the GPU passthrough first (installing 539 MB of torch
before discovering the driver is not passing the GPU through wastes the download),
creates the venv, installs in the right order, and verifies CUDA at the end.

Point the caches at Linux storage — `D:` is a Windows path and would be slow:

```bash
export HF_HOME="$HOME/hf_cache"
```

Add that to `~/.bashrc` if you want it to persist across shells.

**Files already built on Windows** (`data/processed/*.parquet`) are reachable at
`/mnt/c/Users/DEV NARAYAN/Downloads/llm_final/Veritas/data/processed/` — copy them in
rather than rebuilding the RAID download:

```bash
cp -r "/mnt/c/Users/DEV NARAYAN/Downloads/llm_final/Veritas/data/processed" ~/Veritas/data/
```

WSL2 has no systemd by default and its default memory cap is 50% of Windows RAM — on an
8 GB machine that is 4 GB, which is workable for this model but tight. Cap it explicitly
in `C:\Users\DEV NARAYAN\.wslconfig` if you see the kernel OOM-killer:

```ini
[wsl2]
memory=6GB
swap=8GB
```

Restart with `wsl --shutdown` after editing that file.

Then confirm ModernBERT can load at all (~600 MB download, once). In WSL use the bash
spelling of the same thing:

```powershell
python -c "
import transformers, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
print('transformers', transformers.__version__)
tok = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
m = AutoModelForSequenceClassification.from_pretrained('answerdotai/ModernBERT-base', num_labels=2)
print('ModernBERT loaded, params:', sum(p.numel() for p in m.parameters()))
"
```

If `transformers` is below 4.48 this raises an `AutoModelForSequenceClassification`
error — ModernBERT support landed in 4.48, which is why the pin is 4.49.0.

## 7. Data and the fine-tune

```powershell
python ml/prepare_data.py

python ml/load_raid.py --out data/processed/raid_train.parquet --per-generator 4000

# smoke test first: catches OOM and schema problems in ~3 minutes
python ml/train_modernbert.py --limit 200 --epochs 1 --max-len 512 --batch-size 4 --out ml/artifacts/modernbert_smoke

# the real run
python ml/train_modernbert.py --data data/processed/raid_train.parquet --out ml/artifacts/modernbert --max-len 1024 --batch-size 8 --grad-accum 2 --epochs 3
```

## Windows-specific gotchas

**Path length.** Long venv paths plus deep `ml/artifacts/...` output can exceed the old
260-character limit. If you see `FileNotFoundError` on a deep path:

```powershell
# once, as administrator
Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1
```

**Datasets cache location.** `datasets` writes to `%USERPROFILE%\.cache\huggingface`. RAID
is multi-GB; point it somewhere with room before you start — but confirm the drive exists
first, or the failure looks unrelated to the cache:

```powershell
Test-Path D:\                # if False, use a path on C:
$env:HF_HOME = "D:\hf_cache"   # or "C:\hf_cache"
```

**Page file / RAM.** You have 8 GB. Windows will swap hard during tokenization. Close
Chrome and anything else heavy before the real run; the smoke test tells you whether you
have headroom before you commit hours.

**Full path in `py -0p`.** If `py -3.12` is not recognised after installing, the launcher
may not have refreshed. Close and reopen PowerShell, or invoke the interpreter directly:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
```

## Quick sanity sequence

```powershell
python --version                                    # 3.12.x
python -c "import numpy; print(numpy.__version__)"  # 1.26.4
python -c "import torch; print(torch.cuda.is_available())"   # True
py -m pytest -q                                     # 123 passed, 3 skipped
```
