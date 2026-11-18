# Runbook: ModernBERT + HC3/RAID, fully local

Decisions locked with you: **ModernBERT-base**, **HC3 + RAID**, **no Vercel**.
Nothing in this document deploys anything. Nothing is posted anywhere.

---

## Set Vercel aside first (one-time, in your browser)

This repo's Vercel project is **git-connected**, so any push to `main` auto-deploys.
Do this once and the question never comes up again:

> Vercel → project **veritas** → Settings → Git → **Disconnect Git Repository**

The existing site keeps serving; it just stops rebuilding on push. I will not call the
deploy connector again in this session.

---

## What you are actually fixing (read this or you'll optimise the wrong thing)

Measured on your own `test_ood` split with the current deployed model:

| words | human false-positive rate |
|---|---|
| 60–100 | 2.4% |
| 100–150 | 7.5% |
| 150–200 | 13.9% |
| 200–250 | 32.6% |
| 250–300 | 54.7% |
| 400+ | **82.1%** |

And the cause is partly mechanical: the existing recipe truncates at `max_length=256`
(≈190 words), which cuts **32.8%** of your training rows. At `max_len=1024` only **0.31%**
are truncated. ModernBERT at a longer window is not a luxury here — it is the fix.

Also worth stating plainly: `test_ood` is **90.1% human**, so raw accuracy there is
meaningless (always answering "human" scores 90.1%). Everything below reports **balanced
accuracy** and **human false-positive rate**. Treat those as the real numbers.

---

## Step 0 — get the code

```bash
git clone https://github.com/Dev-narayan01/Veritas.git
cd Veritas
git switch arena/772d7f78-veritas      # has the new scripts; main does not
```

## Step 1 — Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r backend/requirements-dev.txt
pip install -r ml/requirements.txt
```

## Step 2 — GPU stack (exact versions that support ModernBERT)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.49.0" accelerate sentencepiece
```

`transformers < 4.48` **cannot** load ModernBERT — it will fail with an
`AutoModelForSequenceClassification` error. 4.49.0 is the mature pin; 5.x changes APIs
for no benefit here.

Verify before going further:

```bash
python -c "import torch, transformers; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); \
print('transformers', transformers.__version__)"
```

You need `cuda True NVIDIA GeForce RTX 3050`. If `cuda` is `False` you installed a CPU
wheel — reinstall from the `cu121` index above.

## Step 3 — data

```bash
python ml/prepare_data.py                  # HC3, ~70 MB download, writes data/processed/
```

Sanity check the splits:

```bash
python -c "
import pandas as pd
for s in ['train','val_stack','val_cal','test','test_ood']:
    d = pd.read_parquet(f'data/processed/{s}.parquet')
    print(f'{s:10} {len(d):6d} rows  AI={int(d.label.sum()):5d}')
"
```

Expect `train 12338 / val_stack 2080 / val_cal 2045 / test 4168 / test_ood 7484`.

## Step 4 — RAID (the part that fixes unseen data)

```bash
python ml/load_raid.py --out data/processed/raid_train.parquet --per-generator 4000
```

This streams ~10M rows and keeps a **capped, rebalanced** subset — capping per generator
is deliberate: RAID is dominated by a few models, and an uncapped sample reproduces the
single-generator bias HC3 already has. Expect a multi-GB download and 10–30 minutes.

Inspect what you got before training on it:

```bash
python ml/load_raid.py --report-only --out data/processed/raid_train.parquet
```

You want **many distinct generators** (not 6+), several domains, and a non-trivial
attacked share. If `generator` shows one or two values, the schema changed — tell me and
I'll adjust the loader rather than let you train on a broken mix.

## Step 5 — smoke test first (2–3 minutes, catches OOM and schema bugs)

Never start a multi-hour run without this:

```bash
python ml/train_modernbert.py --limit 200 --epochs 1 --max-len 512 --batch-size 4 \
    --out ml/artifacts/modernbert_smoke
```

It should finish and print a report. If it OOMs here, drop `--batch-size` to 2.

## Step 6 — the real fine-tune

Start at 1024 tokens / batch 8. On a 3050 this is roughly 3 epochs × ~12k rows.

```bash
python ml/train_modernbert.py \
    --data data/processed/raid_train.parquet \
    --val  data/processed/val_stack.parquet \
    --ood  data/processed/test_ood.parquet \
    --out  ml/artifacts/modernbert \
    --max-len 1024 --batch-size 8 --grad-accum 2 --epochs 3
```

If you hit `CUDA out of memory`, work down in this order:
`--batch-size 4` → `--max-len 768` → `--max-len 512`. Each is still far better than the
current 256. The script writes `eval_report.json` with balanced accuracy, human FPR and
the **by-length breakdown** — that last table is how you tell whether the defect is
actually fixed.

## Step 7 — compare against the current model honestly

Do not trust a single run. The report gives you the numbers; the comparison that matters:

| metric | current (classical + ngram) | your new model | target |
|---|---|---|---|
| balanced accuracy on `test_ood` | 0.896 | ? | > 0.90 |
| human FPR, 400+ words | 82.1% | ? | < 20% |
| human FPR, 100–150 words | 7.5% | ? | ≤ 7.5% |

If balanced accuracy improves but the long-document FPR doesn't move, the model learned
length-independent features and the problem is still the *threshold*, not the model.

---

## Honest expectations

**Where this should win, clearly:** generator invariance. HC3 is one model; RAID is
eleven plus adversarial attacks. That is the difference between "looks good in-domain"
and "holds up on text it has never seen".

**Where it may disappoint:** your measured failure is length-correlated. A longer window
helps by showing the model more text, but a transformer can still drift on very long
input. The by-length table in `eval_report.json` is the check.

**What I will not claim:** that this beats your current 0.896 balanced accuracy until the
numbers are in front of us. A bigger model on more data is a strong prior, not a result.

---

## Two traps that will cost you an hour each

1. **`VERITAS_ARTIFACT_DIR`** — tests point it at `ml/artifacts`, so a local run can
   silently read *training* artifacts instead of what you just built.
   `echo $VERITAS_ARTIFACT_DIR` when numbers look wrong.
2. **NLTK `punkt`** — if you install it, the backend's sentence splitter changes and
   `sentence_len_mean` drifts ~12 units versus the trained model. This only affects the
   classical/ngram path, not ModernBERT, but it will confuse a side-by-side comparison.

---

## When you want me to measure it

Push your branch and tell me the name. I can run the evaluation against `test_ood` here —
no deploy, just numbers — and tell you whether it actually beat 0.896.
