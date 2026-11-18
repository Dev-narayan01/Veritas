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
pip install -r ml/requirements.txt          # pulls torch from PyPI: the CPU wheel
pip install "transformers==4.49.0" accelerate sentencepiece

# then replace torch with a CUDA build; the tag comes from `nvidia-smi`
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

`transformers < 4.48` **cannot** load ModernBERT — it will fail with an
`AutoModelForSequenceClassification` error. 4.49.0 is the mature pin; 5.x changes APIs
for no benefit here.

Torch is installed **last on purpose.** `ml/requirements.txt` pulls
`sentence-transformers`, which depends on `torch`, so installing it first puts the CPU
wheel in place; a later `pip install torch --upgrade` then reports "already satisfied"
and changes nothing. On Windows the PyPI wheel is always CPU-only — only
`download.pytorch.org` ships CUDA builds for Windows.

Verify before going further. Check `torch.version.cuda` rather than calling
`get_device_name()`, which raises before printing anything useful:

```bash
python -c "import torch, transformers; \
print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'available', torch.cuda.is_available()); \
print('transformers', transformers.__version__)"
```

You need `cuda build 12.x` and `available True`. Then, separately:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

You want `NVIDIA GeForce RTX 3050`.

### Reading the first real result, and why it failed

A full 2-epoch run on 21,574 HC3 rows produced:

| split | balanced acc | ROC-AUC | human FPR |
|---|---|---|---|
| `val_in_domain` | **0.9885** | 0.9991 | 0.9% |
| `test_ood` | **0.4415** | 0.3052 | **99.0%** |

Both numbers are real and they disagree. `val_stack` is drawn from the same four HC3
domains and the same generator as training, so 0.9885 measures memorisation of a
format, not detection. On unseen text the model labels 99% of *human* writing as AI.

Compare against the trivial baselines before claiming anything:

```
test_ood is 90.1% human
always-human   accuracy 0.9011   balanced 0.5000   human FPR 0.000
this model     accuracy ~0.203   balanced 0.4415   human FPR 0.990
```

Balanced accuracy below 0.5 is worse than a coin flip, and ROC-AUC below 0.5 means AI
is ranked *below* human. The cause is distributional, not a threshold: every arXiv
abstract and every wiki_csai row was held out, so formal academic prose appeared only
in `test_ood` and none of it in training. The model learned "formal implies AI".

Note the failure is flat across lengths (94.6%-99.5%), unlike the old system's length
defect (2.4% at 60-100 words rising to 82.1% at 400+). This is a domain failure, not the
length failure the long context window was meant to fix.

### The fix: mix formal text into training

`prepare_data.py` now takes `--mix-formal`, the share of arXiv rows kept out of domain:

```bash
python ml/prepare_data.py --force --mix-formal 0.5
```

* `0.0` (default) is the original behaviour: nothing formal in train. Kept as the
  default so old numbers stay comparable.
* `0.5` puts half the abstracts into train/val/test and holds back the other half.

The two halves are disjoint by construction (each abstract carries its own group id,
and the split is seeded), so no document is both trained on and tested on. This is
pinned by `tests/test_prepare_data_formal_mix.py`.

### The mixed run, and why its aggregate OOD number is not comparable

At `--mix-formal 0.5` the same recipe (2 epochs, 1024 tokens, batch 4) produced:

| split | balanced acc | ROC-AUC | human FPR |
|---|---|---|---|
| `val_in_domain` | 0.9823 | 0.9981 | 1.0% |
| `test_ood` | **0.8834** | 0.9193 | **11.5%** |

That is a real change: 99.0% -> 11.5% human false positives. But the two runs were
**not scored on the same test set**, and the difference is not cosmetic:

```
mix 0.0:  test_ood n=8272,  AI=1537, human 81.4%
mix 0.5:  test_ood n=5282,  AI=1537, human 70.9%
```

The AI count is identical; only the human side shrank, by exactly the arXiv rows that
moved into training. The mixed model is therefore evaluated against **fewer of the hard
negatives it was trained to handle**. Some of the gain is a better model and some is an
easier exam, and an aggregate balanced accuracy cannot separate the two.

**Compare per domain, never in aggregate.** `eval_report.json` now carries `by_domain`
for every split. `wiki_csai` is the honest yardstick: it stays out of training at any
`--mix-formal` value, so its human FPR measures generalisation rather than recall of
the domain just added.

The length breakdown also changed character. With `0.0` the OOD error was flat
(94.6%-99.5%). With `0.5` it is length-dependent again:

| length | mix 0.0 | mix 0.5 |
|---|---|---|
| <100 | 98.6% | 5.4% |
| 100-200 | 99.5% | 11.0% |
| 200-400 | 97.9% | 35.4% |
| 400+ | 94.6% | **42.9%** |

Short text is now handled well and long text still is not. That is the original defect
re-emerging rather than a new one, and it is the thing a long context window was meant
to fix.

### Both models on one fixed test set (8,272 rows, per domain)

The aggregate comparison above was confounded. Scoring both saved models on the
identical split separates a better model from an easier exam:

| domain | n | mix 0.0 human FPR | mix 0.5 human FPR |
|---|---|---|---|
| `formal_academic` (human only) | 2,989 | **99.5%** | **3.3%** |
| `wiki_csai` (held out in both) | 1,496 | 94.8% | **44.0%** |
| `mage_heldout` (AI recall) | 797 | 75.5% | 79.2% |

`wiki_csai` is the honest yardstick: it was trained on in *neither* run, so its
improvement from 94.8% to 44.0% is generalisation, not recall of the domain just added.

So the domain-mix fix is real, but insufficient. **44% of held-out human encyclopaedic
text is still labelled AI.** That is not a shippable detector: on Wikipedia-style prose
it flags almost half of genuine human writing.

Reading the two rows together: `formal_academic` improved most because that is exactly
what was added to training. `wiki_csai` improved less because it is a *related* domain,
not the same one. The 41-point gap between them is the memorisation fraction.

**`--mix-formal` changes what `test_ood` means.** At 0.0, `formal_academic` is an
unseen *domain*. At 0.5 it is a seen domain with unseen *documents*. Any reported
number has to say which it is, which is why the manifest now records `ood_domains`,
`train_domains` and `mix_formal`.

### Two traps before the first run

**`train 1 rows` in the banner.** `ml/prepare_data.py` falls back to a 60-row synthetic
demo set when no dataset source is reachable, and the split then puts a single row in
`train`. It logs `synthetic_demo` at WARNING and records it in
`data/processed/split_manifest.json`. The training script now refuses to start below
`--min-rows` (default 100) rather than producing a meaningless number. Fix the data:

```bash
python ml/prepare_data.py --force
grep -o '"data_source": *"[^"]*"' data/processed/split_manifest.json
```

You want `hc3` or `local_csv`, not `synthetic_demo`.

**`InductorError: Failed to find C compiler`.** ModernBERT calls `torch.compile` on its
embeddings when `config.reference_compile` is truthy, and on CUDA that goes through
Triton, which JIT-builds a CUDA extension. A fresh WSL installation has no `gcc`, so the
first training step dies. The script sets `reference_compile = False` by default, which
skips the compiled path entirely:

```
17%|████ | 0/1 [00:05<?, ?it/s]
torch._inductor.exc.InductorError: RuntimeError: Failed to find C compiler.
```

`--compile` opts back in, and then you need `sudo apt install -y build-essential`.
Measured trade-off: compiling costs minutes of warmup for a marginal speedup on a
149M-parameter model, so off is the sane default.

### If `import torch` itself fails on Windows

Two different failures look alike. Separate them before doing anything:

| symptom | `torch.version.cuda` | cause | fix |
|---|---|---|---|
| works, `available False` | `None` | CPU wheel from PyPI | reinstall from the cu126 index |
| `OSError [WinError 4551] An Application Control policy has blocked this file` | n/a | **Smart App Control** refusing unsigned DLLs | see below |

The second is not an install problem and reinstalling will not clear it. PyTorch ships
unsigned DLLs in `torch/lib`; Smart App Control (policy
`{0283AC0F-FFF1-49AE-ADA1-8A933130CAD6}`, event 3077) requires them signed or
Microsoft-reputed. PyTorch tracks the signing gap as `pytorch/pytorch#199711`.

Measured on an RTX 3050 machine: the **PyPI CPU wheel imported fine**, and the **cu126
wheel died on `c10_cuda.dll`**. Reputation is per-file-hash, so which wheel survives is
not predictable from the version.

Confirm it is the policy, then pick an option:

```powershell
# is SAC enforcing? 1 = on, 0 = off, 2 = evaluation
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' |
    Select-Object VerifiedAndReputablePolicyState

# what got blocked, and under which policy
Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-CodeIntegrity/Operational'; Id=3077} -MaxEvents 5 |
    Format-List TimeCreated, Message
```

1. **WSL2** — no Smart App Control in Linux, and the Windows NVIDIA driver already
   exposes CUDA to WSL. Keeps the security feature on. Files live under
   `/mnt/c/...`, which is slow for I/O-heavy work, so clone into the Linux filesystem
   (`~/Veritas`) and copy data in.
2. **Turn Smart App Control off** — Windows Security ▸ App and browser control ▸
   Smart App Control. **It cannot be turned back on without reinstalling Windows.**
   A real trade-off, not a formality.
3. **CPU with `--allow-cpu`** — `python ml/train_modernbert.py --allow-cpu --limit 200`.
   Runs, and is a valid smoke test, but the real run is hours not minutes.

On a machine you do not administer, the policy belongs to whoever does; ask them for an
exclusion rather than fighting it.

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
