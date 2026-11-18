# Running Veritas locally (with your RTX 3050)

Everything below is verified against this repo: Python 3.11, Node 22, the exact
`requirements.txt` files, and the real `ml/run_all.py` step names.

One honest framing before you start, because it decides how you spend your evening:

**GPT-2 will not fix your accuracy problem.** The measured defect is a length/domain
shift — on unseen human text the false-positive rate climbs from 2.4% at 100 words to
82% at 400+ words. A better perplexity backend does not touch that; it can make it
worse, because GPT-2 is *more* confident on fluent long prose. GPT-2 gets you
`standard`/`deep` modes and an honest "transformer" component. The accuracy work is the
length fix, and that needs no GPU at all.

So this guide has **two tracks**. Track A is the project running (15 min). Track B is
the GPU step (another 15 min, optional).

---

## 0. Get the code

The repo is public, so no token and no zip from me:

```bash
git clone https://github.com/Dev-narayan01/Veritas.git
cd Veritas
```

> **Read this before you assume something is missing.** `data/`, `ml/artifacts/` and
> `models/` are **git-ignored deliberately** — training data and trained bundles never
> get committed. A fresh clone has *no* `data/processed/*.parquet` and *no*
> `ml/artifacts/*.pkl`. That is normal. Track A-1 rebuilds them.

Two branch notes: the work from our session is on `arena/772d7f78-veritas`, and `main`
does not have it. Use the branch if you want the current state:

```bash
git switch arena/772d7f78-veritas     # optional; main works too
```

---

## Track A — run the project

### A-1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r backend/requirements-dev.txt
```

No CUDA, no `torch` yet. This is the CPU path and it is what the deployed site runs.

### A-2. Get the training data

`ml/prepare_data.py` looks for the HC3 mirror and **downloads it if absent**. That
mirror is a GitHub tarball, which works from your machine:

```bash
python ml/prepare_data.py
```

Expect ~70 MB of `data/raw/downloads/hc3/data/HC3.jsonl` and ~15 MB of parquet under
`data/processed/`. If GitHub is slow, any mirror of HC3 works — just put a file named
`HC3.jsonl` at `data/raw/downloads/hc3/data/HC3.jsonl` and the loader will find it.

Verify the splits look right before training on them:

```bash
python -c "
import pandas as pd
for s in ['train','val_stack','val_cal','test','test_ood']:
    d = pd.read_parquet(f'data/processed/{s}.parquet')
    print(f'{s:10} {len(d):6d} rows  AI={int(d.label.sum()):5d}')
"
```

Expected: `train 12338`, `val_stack 2080`, `val_cal 2045`, `test 4168`, `test_ood 7484`.

> **Important, and it will bite you otherwise.** `test_ood` is **90.1% human** — 6,744
> human rows and only 740 ChatGPT rows, and the 5,988 arxiv rows contain **zero** AI
> text. A system that answers "human" every time scores 90.1% on this split. Always
> report balanced accuracy alongside raw accuracy, or you will chase a base rate.

### A-3. Train the models (CPU, ~10–15 min)

```bash
python ml/run_all.py --force
```

Steps that write artifacts, in order: `prepare_data` → `train_classical` →
`train_perplexity_model` → `train_transformer` → `compute_zeroshot_features` →
`train_stackers` → `tune_thresholds` → `evaluate` → and the evaluation extras.

Several steps need `torch`/`transformers` or `sentence-transformers` and will log that
they are **skipped**. That is by design — missing components are reported through
`/api/health`, never hidden. If you want them, see Track B. To retrain a single step:

```bash
python ml/run_all.py --force --steps train_classical,train_stackers
```

### A-4. Run the API

```bash
cd backend
../.venv/bin/uvicorn app.main:app --reload --port 8000
```

Check it before touching the UI:

```bash
curl -s localhost:8000/api/health | python -m json.tool | head -40
```

You want `models_loaded.classical: true`. `ppl_gpt2` is true if the n-gram bundle was
built; `transformer`/`binoculars` stay false until Track B.

### A-5. Run the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to :8000
```

No file uploads? The full server supports them; the Vercel Lite functions do not.

### A-6. Confirm nothing is broken

```bash
python -m pytest -q                     # 123 passed, 3 skipped
cd frontend && npx vitest run           # 66 passed
```

---

## Track B — the GPU step (optional)

Your 3050 has 6 GB VRAM. GPT-2 small is 124M params ≈ 500 MB fp32 and fits easily.

### B-1. Install torch + transformers

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.2
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True NVIDIA GeForce RTX 3050`. If it prints `False`, you installed a CPU wheel;
reinstall with the CUDA index above.

### B-2. Download the weights once

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('gpt2')
AutoModelForCausalLM.from_pretrained('gpt2')
print('gpt2 cached')
"
```

### B-3. Retrain the perplexity component on GPT-2

`ml/train_perplexity_model.py` already prefers GPT-2 automatically: it calls
`_prefer_gpt2()` (line ~44) and falls back to the n-gram backend if torch or the weights
are unavailable. With B-1 and B-2 done:

```bash
python ml/train_perplexity_model.py --force
python ml/train_stackers.py --force          # the ppl features changed
python ml/evaluate.py                        # new metrics.json
```

Check it actually switched backends:

```bash
python -c "
import json; m=json.load(open('ml/artifacts/ppl_meta.json'))
print('backend =', m.get('backend'))
"
```

You want `gpt2`. If it says `ngram_lm`, the import failed — read the log line
`GPT-2 backend not usable here (...)`; it names the reason.

### B-4. The one code change you will need

`scripts/build_ppl_bundle.py` **deliberately refuses a `gpt2` artifact**:

```python
if getattr(artifact, "backend", None) != "ngram_lm":
    print(f"refusing to bundle backend={artifact.backend!r}: only ngram_lm is torch-free")
```

That check exists so nobody ships a 500 MB transformer to a 10-second serverless
function by accident. Two legitimate paths:

- **Local only:** skip the bundle step entirely. The full backend loads
  `ml/artifacts/ppl_gpt2_model.pkl` directly in Track A-4, so GPT-2 works locally
  without touching that guard.
- **Deploy:** do not ship GPT-2. Extract the 16 numeric features once into a cache and
  ship the cache. I can write that script when you want it — it keeps Vercel torch-free
  and keeps the cold start under a second.

---

## The accuracy work (no GPU, do this first)

The defect, measured on your own `test_ood`:

| words | n | human false-positive rate |
|---|---|---|
| 60–100 | 2,170 | 2.4% |
| 100–150 | 2,284 | 7.5% |
| 150–200 | 1,182 | 13.9% |
| 200–250 | 221 | 32.6% |
| 250–300 | 117 | 54.7% |
| 400+ | 56 | **82.1%** |

Long human prose accumulates stylistic evidence until it crosses `t_ai`. Two fixes, in
order of payoff:

1. **Aggregate style features instead of summing them.** Rates, per-sentence means and
   dispersion rather than counts. This is the actual fix.
2. **Wire the per-length thresholds you already compute.** `tune_thresholds.py` writes
   `thresholds.json → lite.per_bucket` (e.g. `"100-250": {t_ai: 0.967, t_human: 0.928}`)
   and `score()` in `api/_lib/core.py` **never reads it** — it uses the flat band only.
   That is a half-built fix sitting on disk.

To reproduce my measurements, cache the OOD scores once (~3.5 min), then every experiment
is seconds:

```python
# save as scripts/cache_ood_scores.py, run once
import sys; sys.path.insert(0, 'api')
from _lib import core
import pandas as pd
d = pd.read_parquet('data/processed/test_ood.parquet')
rows = []
for _, r in d.iterrows():
    rows.append({
        'p_lite': core.score(r['text'], mode='lite')['documents'][0]['ai']['probability'],
        'p_fast': core.score(r['text'], mode='fast')['documents'][0]['ai']['probability'],
        'y': int(r['label']), 'src': r['source'], 'words': int(r['words']),
    })
pd.DataFrame(rows).to_parquet('/tmp/ood_scores.parquet')
```

Then evaluate any threshold or feature change against that frame.

---

## Two traps I hit that will cost you an hour each

**1. `ppt`/`punkt` sentence splitting changes three features.** The port in
`api/_lib/core.py` reproduces `backend/app/text/segment.py::_regex_split` exactly,
abbreviation guard included, because punkt was unavailable during training. If you
install NLTK's `punkt` locally, training will use a *different* splitter than the one
the deployed port matches, and `sentence_len_mean` drifts by ~12 units. Keep
`scripts/verify_fast_mode.py` green — it exits non-zero on divergence.

**2. `VERITAS_ARTIFACT_DIR` shadows the bundle.** Tests point it at `ml/artifacts`, so a
local run can silently read the *training* artifacts instead of the ones you just built
into `api/artifacts`. Check `echo $VERITAS_ARTIFACT_DIR` when numbers look wrong.

---

## Ship a change back

```bash
git switch -c my-accuracy-fix
# edit, test
python -m pytest -q && (cd frontend && npx vitest run)
git add -A && git commit -m "..." && git push -u origin my-accuracy-fix
```

Then either open a PR, or tell me the branch name and I will deploy and measure it
against `test_ood` for you.
