# Backend (full server)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cd backend && ../.venv/bin/uvicorn app.main:app --reload
```

* `app/main.py` — application, CORS, startup warm-up
* `app/api/` — `analyses`, `health`, `metrics`, `corpus` routers
* `app/parsers/` — PDF, DOCX, PPTX, text, image OCR, URL, ZIP, code
* `app/integrity/` — invisible characters, homoglyphs, hidden PDF text
* `app/detector/` — features, classical TF-IDF, token statistics, zero-shot, stackers,
  Platt calibration, thresholds, segmentation, explanation
* `app/plagiarism/` — shingling, alignment, semantic retrieval, merge/exclusions,
  online check, batch comparison, code similarity
* `app/reports/` — matplotlib charts and the ReportLab PDF

Configuration is environment-only (`app/config.py`); secrets are never read from disk
committed to the repository. `docs/DEPLOY.md` covers the Space, `docs/QUICKSTART.md` the
local loop.
