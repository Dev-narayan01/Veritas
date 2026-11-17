# Veritas

Transparent AI-text and plagiarism analysis — with the evidence, not just a score.

Veritas reads a document, decides whether it looks machine-written (with a calibrated
probability, an interval, the thresholds in force and the components that actually ran), and
reports exactly which spans match which sources after removing quotations and bibliographies.

The design rule is simple: **every number on the site comes from a file produced by an
evaluation script in this repository**, and anything that could not run says so.

```
backend/    FastAPI: parsers, integrity, detector modes, plagiarism, reports, history
ml/         17-step training + evaluation pipeline; writes every artifact the UI reads
frontend/   Vite + React + TypeScript; 3D scenes that degrade to 2D and Off
api/        Vercel Lite functions: pasted text, classical detector, stated limits
docs/       REPORT · LITERATURE · EXPLAINED · QUICKSTART · DEPLOY · HUMAN_STEPS · KAGGLE
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt
.venv/bin/python scripts/make_bundled_corpus.py
.venv/bin/python ml/run_all.py --smoke          # ~2 min on CPU, correctness only
(cd backend && ../.venv/bin/uvicorn app.main:app --port 8000)
(cd frontend && npm install && npm run dev)     # http://localhost:5173
```

Full instructions: [`docs/QUICKSTART.md`](docs/QUICKSTART.md). Deployment:
[`docs/DEPLOY.md`](docs/DEPLOY.md), with the owner-only steps in
[`docs/HUMAN_STEPS.md`](docs/HUMAN_STEPS.md).

## What it is honest about

* A mode is offered only when **every** component it names has loaded; `/api/health` gives the
  reason for each one that did not.
* Out-of-domain results are shown **first** on the Evaluation page, and the in-domain numbers
  are labelled optimistic.
* A bias flag appears **only** when the 95% interval of the difference excludes zero.
* Every perturbation reports `changed_fraction`, so an attack that did nothing cannot be
  claimed as robustness.
* The plagiarism corpus size is printed next to every percentage, and the benchmark is labelled
  synthetic.
* A `--smoke` run cannot reach the live site: the page renders a banner and
  `ml/adopt_artifacts.py` refuses the bundle.

## Tests

```bash
.venv/bin/python -m pytest tests -q     # 63
bash scripts/check_frontend.sh          # typecheck + 50 vitest + build (+ Playwright when installed)
bash scripts/smoke.sh                   # live end-to-end against the real artifacts
```
