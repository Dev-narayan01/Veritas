# Veritas — progress checklist

Status legend: `[ ]` not started, `[~]` in progress, `[x]` done.

Started 2026-11-17; semester deadline 2026-11-20. Workspace was empty apart from a one-line
`README.md`, so the project was built from the master specification and there was no earlier
build to audit.

## A — Foundation and honesty contract

- [x] **A1 Audit** — the workspace had no prior build, so §3 defects are addressed by
  construction; recorded in `docs/DECISIONS.md`.
- [x] **A2 Naming** — every user-visible string says Veritas. `tests/test_naming.py` fails the
  suite if a stale product name reaches the shipped tree.
- [x] **A3 Honesty fixes** — mode availability computed from loaded components; `components_run`
  / `components_unavailable` on every result; conditional band-collapse reporting; bias flag
  only when the CI excludes zero; `changed_fraction` on every perturbation; calibration guard
  (≥ 100 unique scores); OOD-first Evaluation page; smoke banner; adoption script refuses
  smoke bundles. Covered by `tests/test_api.py`, `tests/test_detector.py`,
  `tests/test_lite_deployment.py` and `frontend/tests/unit/*`.
- [x] **A4 Defect-4 regression coverage** — `tests/test_perturbations.py` asserts that each
  attack changes ≥ 80 % of documents *and* changes them the right way (a shuffle must keep the
  word multiset, homoglyphs must insert Cyrillic). A perturbation that silently stops working
  now fails the build instead of producing a flattering robustness table.
- [x] **A6 Honest unavailability reason** — the heavy modes name the missing component
  (`transformer: weights missing`) rather than their stacker; a stacker is useless without its
  inputs, so blaming it hides the real blocker. Regression-tested in `tests/test_api.py`.
- [x] **A7 Every input path verified end to end** — PDF, DOCX, PPTX, Markdown, code and ZIP were
  pushed through the live API and each returned `done` with the expected kind and word count.
  OCR answers a clear `ocr_unavailable` because tesseract is not installed here. The URL path
  originally escaped a network failure as an unhandled 500; every `requests` failure is now a
  422 naming the cause (`url_tls_failed`, `url_timeout`, `url_unreachable`, `url_http_error`),
  with an endpoint-level fallback so no input can ever produce a bare 500.
- [x] **A8 Batch comparison works and validates end to end** — batch mode was broken: `batch.py`
  read `match.query_span`, an attribute the aligner never had, so every batch analysis died with
  `'Match' object has no attribute 'query_span'`. Spans are now properties on `Match`, batch has
  5 tests, and a live run produced `[[100, 98.26], [48.8, 100]]` with one edge. Code plagiarism
  detects a fully renamed copy at similarity 1.0 and now reports `(original → copy)` function
  pairs, and those structural matches also appear on the similarity graph as `kind: "code"` edges
  so the graph cannot contradict the code card.
- [x] **A9 Exclusion filters verified** — quotes, bibliographies and the 8-word floor each
  leave the raw score intact and zero the filtered score, which is the behaviour the UI toggle
  implies (both default on). Three tests in `tests/test_plagiarism.py`.
- [x] **A10 The no-WebGL path is rendered, not just decided** — `tests/unit/scenes.test.tsx` now
  mounts the embedding scatter, the similarity graph and the verdict gauge at tier `off` and
  asserts there is no `<canvas>` and that the 2D/SVG/list alternative carries the same numbers.
  This runs in jsdom, so it covers the requirement that Playwright cannot verify here. Writing it
  immediately caught the test using a friendlier projection shape than the artifact ships.
- [x] **A5 Parser and security coverage** — `tests/test_parsers.py` (real PDF/DOCX/PPTX/TXT/code
  fixtures, ZIP traversal refusal, zip-bomb refusal, SSRF blocking, oversize page) and
  `tests/test_security.py` (upload cap, type refusal, error shape, limiter, path traversal on
  `/api/metrics/plots`).

## B — ML pipeline

- [x] **B1** Data + group-disjoint splits + four OOD holdouts (`ml/prepare_data.py`)
- [x] **B2** Training chain: classical, token statistics, zero-shot, stackers, Platt thresholds
- [x] **B3** Evaluation chain: metrics + ablation, cross-generator, baseline, robustness, bias,
  embedding projection — each writing its own JSON. `evaluate.py` also emits confusion matrices
  (abstentions counted separately), accuracy-by-length buckets and per-domain/generator/length
  AUROC, plus `confusion_*.png` and `accuracy_by_length_*.png`.
- [x] **B6** Adoption guard tests — `tests/test_artifacts_adoption.py` proves the Hub-adoption
  script refuses smoke bundles, foreign products, missing outputs and missing thresholds, and
  keeps a backup of replaced artifacts.
- [x] **B4** Kaggle bundle, `docs/KAGGLE.md`, `docs/HUMAN_STEPS.md`, `ml/adopt_artifacts.py`
- [x] **B5** CPU full-ish run for the live site: `ml/run_all.py --smoke --force` completes all
  17 steps in ~130 s and writes `dist/veritas_artifacts.zip`. A result-grade run is the user's
  step (see `docs/HUMAN_STEPS.md` §5); the report states which run produced its numbers.

## C — Backend and deployment

- [x] **C1** `artifacts.py` (Hub download, manifest-driven `artifact_source`), Dockerfile,
  `scripts/deploy_space.sh`, `scripts/deploy_hub.sh`
- [x] **C2** Frontend wake-up/Lite logic (90 s window, same-origin fallback, honest offline
  state), `scripts/deploy_vercel.sh`, Vercel Lite functions in `api/`

## D — Plagiarism

- [x] **D1** Corpus + LSH index + boilerplate filtering (`ml/build_corpus.py`)
- [x] **D2** Alignment, reorder pass, exclusions, per-source report
- [x] **D3** Paraphrase verification (reports itself unavailable without an embedding model)
- [x] **D4** Benchmark, tuning and evaluation (span F1 0.9412 on the synthetic test half)
- [x] **D5** Opt-in online check; function-level code similarity

## E — 3D

- [x] **E1** Tiers + `CanvasShell` + hero field, scanner, verdict gauge, similarity galaxy,
  embedding space, every one with a 2D/HTML fallback and an Effects → Off path.

## F — Docs and finalisation

- [x] **F1** REPORT, LITERATURE, EXPLAINED, QUICKSTART, DEPLOY, HUMAN_STEPS, KAGGLE
- [x] **F2** `scripts/smoke.sh`, `scripts/check_frontend.sh`, `scripts/finalize.sh`, test
  suites, manifest, this checklist
- [x] **F3** Gates as of the final run: `pytest -q` **112 passed**, `vitest run` **62 passed**,
  `tsc --noEmit` clean, `npm run build` clean, `ruff` clean, `scripts/smoke.sh` all passed,
  `scripts/finalize.sh` **ALL GATES PASSED** (Playwright reported `SKIPPED`).

## Known gaps (deliberate, documented)

- Playwright smoke cannot run in the build sandbox: the browser download host is unreachable,
  so `scripts/check_frontend.sh` prints `SKIPPED`. Commands are in `docs/HUMAN_STEPS.md` §6.
- `standard`/`deep`, cross-generator transfer and the third-party baseline need `torch` and
  pretrained weights, which the sandbox cannot fetch. All three report themselves unavailable
  with reasons instead of showing invented numbers.
- Pushing the Space, the Hub repo and the Vercel project needs credentials that only the owner
  has: `docs/HUMAN_STEPS.md` §1–4.
- The 500+ word bucket in the length table holds 5 held-out documents. It is published with its
  `n` because hiding it would look like cherry-picking, but no conclusion should be drawn from
  it; the < 100-word drop (99 documents, AUROC 0.6661) is the one with enough data to matter.
- Playwright is the only gate that cannot be executed here, so the e2e specs are unverified in
  this environment. They are kept because they are the check that the 3D-off path keeps working
  in a real browser.
