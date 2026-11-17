---
title: Veritas API
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI-text and plagiarism analysis with honest reporting
---

# Veritas API

FastAPI service behind the Veritas frontend. It answers `/api/health` first (so the site can
tell you exactly which detector modes can run here), then `/api/analyses`,
`/api/metrics` and `/api/corpus/stats`.

* Artifacts are read from `VERITAS_ARTIFACT_DIR`, or downloaded from the Hub repo in
  `VERITAS_HUB_REPO` when that is set.
* Documents are deleted as soon as an analysis finishes unless `retain` is requested.
* No document text leaves this Space unless the caller enables the opt-in online source
  check, which sends a handful of short sentences to Wikipedia, arXiv and OpenAlex.

The API is not a verdict machine. Every response carries the components that actually ran,
the thresholds in force and a disclaimer.
