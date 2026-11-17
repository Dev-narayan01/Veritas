# Quickstart

Veritas is three pieces that can run independently:

| piece | what it is | needs |
|---|---|---|
| `backend/` | FastAPI service: parsing, detection, plagiarism, reports, history | Python 3.11 |
| `ml/` | training + evaluation pipeline producing every number the UI shows | Python 3.11 (CPU is enough for `--smoke`) |
| `frontend/` | Vite + React + TypeScript SPA | Node 20+ |

## 1. Backend

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt

# the backend needs artifacts; produce a CPU bundle first (step 2) or it will
# report every detector mode as unavailable — which is the honest behaviour
cd backend
../.venv/bin/uvicorn app.main:app --reload --port 8000
```

Check it: `curl -s localhost:8000/api/health | python3 -m json.tool`.
`modes[]` tells you what can actually run and `reason` says why anything cannot.

## 2. ML pipeline

```bash
.venv/bin/python ml/run_all.py --smoke          # ~2–3 minutes on CPU, correctness only
.venv/bin/python ml/run_all.py                  # real data, CPU, still not the full GPU run
```

Outputs land in `ml/artifacts/` (`metrics.json`, `thresholds.json`, `robustness.json`,
`bias.json`, `plag_benchmark_results.json`, plots, the corpus index) and the bundle is
zipped to `dist/veritas_artifacts.zip`.

**A `--smoke` run writes a banner into `metrics.json`.** The Evaluation page reads it and
refuses to present those numbers as results.

## 3. Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to 127.0.0.1:8000
```

If the API is not running the site still loads: it says "server offline" and stays usable
in read-only mode. If only the Vercel Lite functions answer, the site switches to
pasted-text analysis and says which capabilities are missing.

## 4. Tests

```bash
.venv/bin/python -m pytest tests -q     # 63 backend + Lite function tests
.venv/bin/ruff check backend ml api tests
bash scripts/check_frontend.sh          # typecheck + 50 vitest tests + production build
bash scripts/smoke.sh                   # boots uvicorn and exercises the real endpoints
cd frontend && npx playwright test      # needs `npx playwright install chromium` once
```

`scripts/check_frontend.sh` prints `SKIPPED` for the browser suite when no Chromium is
installed. That is deliberate: a missing browser must not look like a passing E2E run.

## 5. Where things are

```
backend/app/       API, parsers, integrity, detector, plagiarism, reports
ml/                one script per pipeline step + run_all.py orchestrator
ml/bundled/        original sample documents and the two demo samples (no scraped text)
frontend/src/      pages, components, 3D scenes, API client
api/               Vercel Lite functions (pasted text, classical detector only)
docs/              this file, plus REPORT / LITERATURE / EXPLAINED / DEPLOY / KAGGLE
```
