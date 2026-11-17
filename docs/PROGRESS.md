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

## B — ML pipeline

- [x] **B1** Data + group-disjoint splits + four OOD holdouts (`ml/prepare_data.py`)
- [x] **B2** Training chain: classical, token statistics, zero-shot, stackers, Platt thresholds
- [x] **B3** Evaluation chain: metrics + ablation, cross-generator, baseline, robustness, bias,
  embedding projection — each writing its own JSON
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

## Known gaps (deliberate, documented)

- Playwright smoke cannot run in the build sandbox: the browser download host is unreachable,
  so `scripts/check_frontend.sh` prints `SKIPPED`. Commands are in `docs/HUMAN_STEPS.md` §6.
- `standard`/`deep`, cross-generator transfer and the third-party baseline need `torch` and
  pretrained weights, which the sandbox cannot fetch. All three report themselves unavailable
  with reasons instead of showing invented numbers.
- Pushing the Space, the Hub repo and the Vercel project needs credentials that only the owner
  has: `docs/HUMAN_STEPS.md` §1–4.
