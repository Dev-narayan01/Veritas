# Veritas — progress checklist

Status legend: `[ ]` not started, `[~]` in progress, `[x]` done.

Started 2026-11-17. Workspace was empty (only a one-line `README.md`), so the project is
built from the master specification; there was no earlier `VeriText` code to audit.

## A — Foundation, rename state, honesty fixes

- [x] **A1 Audit** — *Done when* every defect is listed with its planned fix.
  Workspace audit: no prior build existed, therefore the section-3 defects are addressed
  by construction. `docs/DECISIONS.md` (2026-11-17) records the audit.
- [ ] **A2 Naming** — every string says Veritas (nothing to rename; verified by grep + test).
- [ ] **A3 Honesty fixes** — mode availability from loaded components, `components_run` /
  `components_unavailable`, bias flag from CI, perturbation `changed_fraction`,
  calibration guard, OOD-first evaluation page. *Done when* related unit tests pass.

## B — ML pipeline

- [ ] **B1** Data, group-disjoint splits, four OOD holdouts `--smoke`
- [ ] **B2** Training chain (classical, ppl_gpt2, zero-shot, stackers, thresholds) `--smoke`
- [ ] **B3** Evaluation chain (metrics+ablation, cross-generator, baseline, robustness, bias,
      embedding projection) `--smoke`
- [ ] **B4** Kaggle bundle, `docs/KAGGLE.md`, `docs/HUMAN_STEPS.md`, `adopt_artifacts.py`
- [ ] **B5** CPU full-ish run for the live site (real, non-smoke artifacts)

## C — Backend + frontend deployment

- [ ] **C1** `artifacts.py`, Dockerfile, Space deploy script, Hub push
- [ ] **C2** Frontend wake-up/Lite logic, Vercel deploy

## D — Plagiarism

- [ ] **D1** Corpus + dense retrieval, boilerplate filter
- [ ] **D2** Alignment, reorder pass, exclusions, per-source report
- [ ] **D3** Paraphrase verification
- [ ] **D4** Benchmark, tuning, evaluation
- [ ] **D5** Online check, function-level code plagiarism

## E — 3D

- [ ] **E1** Tiers, CanvasShell, HeroField, Scanner, VerdictGauge, SimilarityGalaxy,
      EmbeddingSpace + 2D fallbacks

## F — Docs and finalisation

- [ ] **F1** REPORT, LITERATURE, EXPLAINED, QUICKSTART, DEPLOY, HUMAN_STEPS, KAGGLE
- [ ] **F2** finalize.sh, tests, deploy, smoke, manifest, final summary
