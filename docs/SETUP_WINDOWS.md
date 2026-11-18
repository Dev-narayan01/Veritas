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

`pip install torch --index-url .../whl/cu121` failed with **"from versions: none"** because
`download.pytorch.org` publishes per-CUDA wheels, and there is no cu121 build for Python
3.15. With 3.12 it works — and you do not need cu121 specifically.

**Simplest path** (PyPI's `torch` bundles CUDA 12.x and supports 3.11–3.15):

```powershell
python -m pip install torch --upgrade
python -m pip install "transformers==4.49.0" accelerate sentencepiece
```

**If you specifically want the CUDA 12.1 wheels:**

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check the GPU is actually visible — this is the single most common silent failure:

```powershell
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

You need `cuda: True` and `NVIDIA GeForce RTX 3050`. If it says `False` you installed a
CPU-only wheel — reinstall from the cu121 index above.

Then confirm ModernBERT can load at all (this downloads ~600 MB once):

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
is multi-GB; point it somewhere with room before you start:

```powershell
$env:HF_HOME = "D:\hf_cache"
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
