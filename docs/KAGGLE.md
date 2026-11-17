# Running the GPU pipeline on Kaggle

CPU training gives you `lite` and `fast` with the n-gram language model. This notebook adds
the fine-tuned transformer, Binoculars, Fast-DetectGPT, dense paraphrase retrieval and the
third-party baseline. It needs a GPU (T4 is enough) and about 1.5 h.

## 1. Notebook settings

* Accelerator: **GPU T4 ×2** or **P100**
* Internet: **on**
* Add your Hugging Face token as a Kaggle secret named `HF_TOKEN` (Add-ons ▸ Secrets).

## 2. Cell 1 — get the code

```python
!git clone https://github.com/Dev-narayan01/Veritas.git
%cd Veritas
!git checkout main
```

## 3. Cell 2 — dependencies

```python
!pip install -q -r ml/requirements.txt
!pip install -q torch --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers sentence-transformers faiss-cpu umap-learn datasets
```

Verify the GPU is visible:

```python
import torch; print(torch.__version__, torch.cuda.is_available())
```

## 4. Cell 3 — build the data

```python
!python scripts/make_bundled_corpus.py
!python ml/prepare_data.py --full
```

`--full` streams HC3 (all five domains, ~24 k rows) and arXiv abstracts from their GitHub
mirrors, or from Hugging Face when it is reachable. The split is group-disjoint by source
document and writes `data/processed/split_manifest.json` with a hash of the split.

Expected: `data/processed/{train,val_stack,val_cal,test,test_ood}.parquet`.

## 5. Cell 4 — train and evaluate

```python
!python ml/run_all.py --full
```

Steps run in this order (each writes its own artifact):

```
prepare_data → train_classical → train_perplexity_model → train_transformer
→ compute_zeroshot_features → train_stackers → tune_thresholds → evaluate
→ eval_cross_generator → eval_baseline_detector → robustness → bias_check
→ embedding_projection → build_corpus → make_plag_benchmark
→ tune_plag_thresholds → eval_plagiarism
```

Watch for these log lines:

* `platt calibration: 480 unique scores` — the calibration guard (needs ≥ 100)
* `modes fitted: [...]` — anything missing was not trained, and `/api/health` will say so
* `SMOKE RUN` must **not** appear: `--full` writes `run: full`

## 6. Cell 5 — inspect before you publish

```python
import json
m = json.load(open('ml/artifacts/metrics.json'))
print(m['run'], m['notes'])
print(json.dumps(m['splits']['test_ood']['modes'], indent=1))
print(json.load(open('ml/artifacts/manifest.json'))['source'])   # must be "kaggle"
```

If a component failed to train, `/api/health` on the Space will report that mode as
unavailable. That is the intended behaviour: never edit the manifest to look better.

## 7. Cell 6 — download the bundle

```python
from IPython.display import FileLink
FileLink('dist/veritas_artifacts.zip')
```

## 8. Local adoption

```bash
# locally, from the repository root
.venv/bin/python ml/adopt_artifacts.py dist/veritas_artifacts.zip
# --allow-smoke  accept a smoke bundle (deployment testing only)
# --keep         merge instead of replacing ml/artifacts
```

Then redeploy the Space (`bash scripts/deploy_space.sh --push`). The health badge switches
to "GPU-trained artifacts" because the manifest says `source: kaggle`.

## Cost and time

| step | T4 time |
|---|---|
| data build | 12 min |
| classical + perplexity | 10 min |
| transformer fine-tune | 45 min |
| zero-shot features | 25 min |
| stackers + thresholds + evaluation | 15 min |
| plagiarism benchmark + tuning | 10 min |

Kaggle's 30 h/week GPU quota covers several runs. Save a notebook version before changing
anything, so a failed run does not cost you the working one.

## Common failures

| symptom | cause | fix |
|---|---|---|
| `No module named 'torch'` in the log | CPU kernel | switch the accelerator and re-run |
| `datasets` cannot reach Hugging Face | internet off | enable internet, or rely on the GitHub mirrors |
| `CalibrationCollapsed` | stacker learned nothing (bad split) | delete `data/processed` and re-run `prepare_data --full` |
| bundle is 2 GB+ | `ml/artifacts/transformer` kept every checkpoint | keep only the final checkpoint; the loader reads `transformer/` directly |
