# How Veritas works, explained

This is the "explain it to a smart friend" version. `docs/REPORT.md` has the numbers and
`docs/LITERATURE.md` the citations.

## The problem

Given a document, answer two questions:

1. **Was this written by a machine?** — a probability, with an honest interval, plus *where*
   in the text the machine-like writing is.
2. **Was it copied?** — which spans match which sources, after removing what should not
   count (quoted material, your bibliography).

Both answers are politically loaded. A false accusation costs a student their degree, so the
design rule throughout is: **report what you measured, never what would look good.**

## 1. Reading the document

Files arrive as PDF, DOCX, PPTX, plain text, images (OCR), URLs, ZIP archives or source code.
Each parser returns text plus warnings. Then:

* **Integrity scan** — zero-width spaces, soft hyphens, Cyrillic homoglyphs and invisible PDF
  text are found, counted and (for the invisible ones) removed. Each one is a hint that
  somebody tried to game a detector; the count is shown, it is not a verdict.
* **Normalisation with an offset map** — the text is NFKC-normalised and whitespace-collapsed,
  but for every cleaned character the pipeline remembers its position in the original. That
  is why a highlight never drifts by a character, even after quotes are converted or
  ligatures are expanded.

## 2. The detector

### Components

| component | what it actually measures | cost |
|---|---|---|
| `classical` | TF-IDF (char 3–5-grams *or* word 1–2-grams, whichever validates better) → logistic regression | milliseconds |
| `ppl_gpt2` | token log-probability statistics from a language model: mean, spread, burstiness, rank percentiles, plus stylometry (sentence-length variation, punctuation rhythm, function-word rates) | milliseconds (n-gram) / GPU (GPT-2) |
| `binoculars` | ratio of two models' perplexities — the "is this text too easy to predict?" trick | GPU |
| `fast_detectgpt` | curvature: does a slight re-sampling of each token make the text *more* likely? | GPU |
| `transformer` | a fine-tuned classifier on the same data | GPU |

Only the first two are needed for `lite` and `fast`. `standard` adds Binoculars, `deep`
swaps in Fast-DetectGPT. A mode is offered **only when every component it names has
loaded**; `/api/health` returns the reason for each one that did not, and the result carries
`components_run` and `components_unavailable`.

### From scores to a probability

Component scores are not probabilities, so:

1. A **stacker** (logistic regression) combines the component scores with
   `log(word_count)` — because every signal behaves differently on a 60-word paragraph than
   on a 3,000-word essay.
2. **Platt scaling** fits `p = σ(a·decision + b)` on a dedicated calibration split
   (`val_cal`), never on the test split. The calibration guard refuses to publish thresholds
   if fewer than 100 unique calibrated scores appear, because a collapsed distribution means
   the stacker learned nothing.
3. **Two thresholds** are derived from the *human* score distribution at a 1% false-positive
   target: `t_human` (below it: likely human) and `t_ai` (above it: likely AI). The band
   between them is reported as **uncertain** — that is a feature, not a failure. If the band
   collapses (t_human ≥ t_ai) the result says so instead of pretending.

Why per-component thresholds instead of a fresh Platt fit at request time? Because calibrating
per request would make the same text get different verdicts on different days. The scaler is
fitted once, in `ml/tune_thresholds.py`, and shipped in the artifact.

### Where the machine-like writing is

Each sentence is scored **in a ±1-sentence context window** (a sentence alone is too short to
judge) and the scores are smoothed across three neighbouring sentences. A two-state HMM with
a strong stay probability (0.9) then decodes a segment sequence, so the UI draws contiguous
regions instead of a confetti of single sentences. The heatmap colour comes from the
calibrated sentence score; the mixed-authorship stripe comes from the HMM.

Token-level evidence lists the most *predictable* tokens (lowest surprisal), because
predictability is what machine text has too much of.

## 3. Plagiarism

Two stages, both span-based:

**Retrieval.** Every corpus document is fingerprinted with winnowed k-gram hashes
(MinHash-LSH in the index) so a 100k-document corpus can be searched in milliseconds. Short
boilerplate (headers, licence text) is filtered before indexing so it does not dominate.

**Alignment.** Matching fingerprints between query and source are chained into runs that are
near-diagonal in both texts; a run becomes a match only if it survives three checks:
containment ≥ `lexical_min_containment`, at least `min_match_words` words, and a re-ranker
score above `rerank_min_score`. Those three numbers are *tuned on a validation half* of a
benchmark and then frozen (`ml/plag_thresholds.json`).

**Paraphrase.** Sentence embeddings (when a model is available) retrieve near-duplicates that
share no k-grams at all; each candidate is then verified lexically so a fluent paraphrase of
a *different* sentence cannot be reported as a match. When the embedding model is missing,
the engine says so rather than silently dropping recall.

**Exclusions.** Quoted material (≥15-word inline quotations, `>` blocks) and the bibliography
are tagged once per document. Matches keep raw flags, so the viewer can re-filter instantly —
the "after exclusions" percentage is computed live, and the UI shows both numbers.

**Online check (opt-in).** Only then are 6–8 of the most distinctive sentences (≤200
characters each) sent as queries to Wikipedia, arXiv and OpenAlex. Private addresses are
refused outright; the results page lists exactly which sentences were sent.

**Code.** Source files are tokenised structurally (identifiers normalised, comments and
whitespace dropped), winnowed, and compared with layout fingerprints, so renaming variables
or reformatting does not hide a copy.

## 4. The measurement

Everything the Evaluation page shows comes from `ml/` scripts writing JSON:

| file | what it establishes |
|---|---|
| `metrics.json` | ROC-AUC, PR-AUC, TPR at 1% and 0.1% FPR, F1, ECE, with bootstrap CIs; components and modes; ablation; **OOD split first** |
| `robustness.json` | recall before/after each perturbation **and `changed_fraction`** — how much of the text actually changed. A perturbation that did nothing cannot be claimed as robustness |
| `bias.json` | human false-positive rate on normal vs. simplified (ESL-proxy) text, difference with a 95% interval; flagged **only** when the interval excludes zero |
| `plag_benchmark_results.json` | span F1 on a synthetic benchmark with known inserted spans, per obfuscation type, false alarms on negatives |
| `cross_generator.json` | leave-one-generator-out transfer (needs ≥2 generator families) |
| `baseline.json` | one third-party detector on the OOD split, clearly labelled |
| `embedding_projection.json` | 2D/3D projection of the document feature space, for intuition only |

Two honesty rules are enforced in code, not by good intentions:

* `test_ood` is never used for tuning. The in-domain numbers are labelled "optimistic" on the
  page because in-domain test data shares topics with training.
* A `--smoke` run stamps a banner into `metrics.json`; the page renders it and refuses to
  present those numbers as results. The adoption script refuses a smoke bundle outright
  unless explicitly told otherwise.

## 5. The interface

The chart, the number and the text are the interface. 3D is decoration that must never be the
only carrier of information:

* Graphics **tiers**: `high` (instanced scenes, bloom), `low` (fewer instances, no
  post-processing), `off` (no WebGL at all). The tier is decided from WebGL2 availability,
  `prefers-reduced-motion`, core count, memory, touch input and a frame-time probe, and the
  user can override it. **Effects → Off** is one click in the header.
* Every scene is wrapped in a shell that pauses rendering when the tab is hidden or the
  canvas is off-screen, drops `aria-hidden` text alternatives, and falls back to a 2D
  equivalent on context loss. Document text is never rendered inside WebGL.
* At most one WebGL canvas is mounted per page.

## 6. What this system cannot do

* Tell you a document *was* written by a human. It reports a calibrated probability, an
  interval and the thresholds in force.
* Guarantee an interval covers the truth: the interval is a heuristic widening from binomial
  noise around the calibrated score, not a formal posterior.
* Detect plagiarism outside its corpus. The corpus size is printed next to every percentage.
* Survive determined evasion. `robustness.json` reports what was measured, including the
  perturbations that changed nothing.
* Match commercial services. It is a transparent academic system, compared against one
  third-party detector on one split.
