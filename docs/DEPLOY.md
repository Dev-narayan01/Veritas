# Deploying Veritas

Three targets, in this order. Nothing here deletes the existing Vercel project or the
existing Hugging Face Space; they are left untouched for rollback.

## 0. Secrets

All credentials live in `.env.deploy`, which is gitignored. Copy `.env.deploy.example`:

```bash
cp .env.deploy.example .env.deploy
```

Required keys: `HF_USERNAME`, `HF_TOKEN`, `SPACE_NAME` (default `veritas-api`),
`VERITAS_HUB_REPO` (`<HF_USERNAME>/veritas-detector`), `VITE_API_BASE_URL` (the Space URL).
The scripts never echo a token.

## 1. Artifacts → Hugging Face Hub

```bash
.venv/bin/python ml/run_all.py           # writes dist/veritas_artifacts.zip
bash scripts/deploy_hub.sh               # uploads to $VERITAS_HUB_REPO
```

The bundle contains `manifest.json` (commit, data source, split hash, which components
trained, `source: kaggle|cpu`). `/api/health` reads it, so the "GPU-trained" / "CPU-trained
baseline" badge in the UI cannot drift from reality.

The repo must be **public** if the Space should download it without a token; alternatively
set `HF_TOKEN` as a Space secret.

## 2. Backend → Hugging Face Space (Docker)

```bash
bash scripts/deploy_space.sh             # stages dist/space, prints its size
bash scripts/deploy_space.sh --push      # creates/updates <HF_USERNAME>/veritas-api
```

Then, in the Space settings:

| type | name | value |
|---|---|---|
| variable | `VERITAS_ARTIFACT_DIR` | `/app/artifacts` |
| variable | `VERITAS_DATA_DIR` | `/data/state` |
| variable | `VERITAS_HUB_REPO` | `<HF_USERNAME>/veritas-detector` |
| variable | `CORS_ORIGINS` | your Vercel URL and `http://localhost:5173` |
| secret | `HF_TOKEN` | only if the Hub repo is private |

The image installs no torch by default, so the Space runs `lite` and `fast`. To enable
`standard` and `deep`:

```bash
# in the Space, temporarily: Settings ▸ Factory rebuild with build args is not exposed,
# so build locally instead:
docker build --build-arg WITH_TORCH=1 -t veritas-api backend/
```

Check the Space's disk quota before doing that; CPU torch plus transformers is ~1 GB. The
free tier may reject the image. Either way `/api/health` reports the truth.

Verify:

```bash
curl -s https://<user>-veritas-api.hf.space/api/health | python3 -m json.tool
curl -s https://<user>-veritas-api.hf.space/api/metrics | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["artifact_source"], list((d.get("metrics") or {}).get("splits", {})))'
```

The first request after idling wakes the Space (~30–60 s). The frontend shows a wake-up
message for up to 90 s, then falls back to Lite.

## 3. Frontend → Vercel

The Vercel project is `veritas` (`prj_BaQtllto52VH7KYNU8zosZdEwFsm`). **Deploy from the
repository root, not from `frontend/`.** Vercel only auto-detects `api/*.py` relative to the
directory it builds from, so a `frontend/`-rooted build silently ships no API at all. The
root `vercel.json` therefore owns the build:

| key | value |
|---|---|
| `installCommand` | `npm --prefix frontend ci` |
| `buildCommand` | `npm --prefix frontend run build` |
| `outputDirectory` | `frontend/dist` |
| `functions` | `api/{health,samples,metrics}.py` 10 s, `api/analyze.py` 30 s |
| `rewrites` | every non-`/api` path → `index.html` (SPA routing) |
| `headers` | immutable caching for `/assets/*`, `nosniff`, referrer policy |

`api/vercel.json` is now documentation only — nothing in it is honoured. `frontend/vercel.json`
is *also* not read, because the build root is the repository root, so its SPA rewrite and
asset headers were silently missing from the deployment; both now live in the root
`vercel.json`.

`memory` is deliberately not set: Vercel ignores it on Active CPU billing and warns about it.

The Lite model bundle is committed at **`api/artifacts/`** (`classical.pkl` +
`thresholds.json` + `metrics.json` + `manifest.json`, 512 KB). This is not optional: Vercel
packages the functions *before* any build step runs, and `frontend/public/artifacts/` is
gitignored, so a git-based deploy ships no model at all and `/api/health` answers
`"no lite artifact bundle is deployed with this build"`. `api/_lib/core.py` searches
`api/artifacts`, then `artifacts/` beside the function, then the staging and training
directories. `VERITAS_ARTIFACT_DIR` overrides the search entirely.

```bash
vercel link --project prj_BaQtllto52VH7KYNU8zosZdEwFsm   # once, from the repo root
bash scripts/deploy_vercel.sh            # preview
bash scripts/deploy_vercel.sh --prod     # production
```

`scripts/deploy_vercel.sh` needs an authenticated `vercel` CLI (`vercel login`, or
`VERCEL_TOKEN` in the environment). The script fails fast with a clear message if neither is
present, so a CI run needs `VERCEL_TOKEN`, `VERCEL_ORG_ID` and `VERCEL_PROJECT_ID`.

### Deploying from the CLI vs. from Git

Git-based deploys need the Vercel project connected to `Dev-narayan01/Veritas` **and** the
branch pushed to GitHub — Vercel only builds commits that exist on the remote. If
`vercel deploy` fails with `missing_lock_file`, the uploaded file set was incomplete; if an
API deploy fails with *"repository does not contain the requested branch"*, the branch has
not been pushed yet. Deploy from a working directory that is at least the repo root; both
errors disappear once `frontend/package-lock.json` is present and the branch is on the remote.

Environment variable in the Vercel project:

* `VITE_API_BASE_URL` = `https://<user>-veritas-api.hf.space`

`api/requirements.txt` pins only `fastapi`, `numpy`, `scikit-learn` and `joblib`. The Lite
functions must not import the `app` package; `api/_lib/core.py` installs a three-module
pickle shim so `classical.pkl` loads without it.

### How the Python functions are packaged

Vercel's Python builder ships **one file per function** — the entrypoint and nothing else
from this repository. Two deploys proved it the hard way:

| symptom in the deployed function | cause |
|---|---|
| `ModuleNotFoundError: No module named '_lib'` | `from _lib.core import ...`, a sibling package |
| `ModuleNotFoundError: No module named '_core'` | `from _core import ...`, a sibling module |

Both builds reported READY; the failures only appeared on the first request. So the
functions are generated:

* `api/_lib/core.py` — the Lite core, the single source of truth, imported directly by tests
* `api/handlers/<name>.py` — request handling only, no imports from this repository
* `api/<name>.py` — **generated**: `python scripts/build_api.py` splices the two together

Edit the sources, then run `scripts/build_api.py` and commit the result (a git deploy
packages the repository before any build step, so the generated file must be in git).
`tests/test_lite_deployment.py` rebuilds the functions and fails if they are stale, and
asserts no function imports anything from this repository.

Two more traps, both of which produce a green build and a broken runtime:

* **Do not add `api/pyproject.toml`.** Vercel prefers it over `api/requirements.txt`, then
  resolves from a generated `uv.lock` and installs nothing the functions need — the symptom
  is `No module named 'joblib'` from a healthy-looking build. Lint configuration lives in
  `ruff.toml`, which Vercel ignores. A test enforces this.
* **Every function must be importable on its own.** `api/health.py` builds its payload
  defensively so a broken bundle answers 200 with a `degraded` body naming the failure,
  instead of an opaque `500 FUNCTION_INVOCATION_FAILED`. Read that body first when
  debugging: `curl -s https://<project>.vercel.app/api/health | python3 -m json.tool`.

### What Lite can and cannot do

| capability | full server | Lite |
|---|---|---|
| pasted text, classical detector, calibrated verdict | yes | yes |
| PDF / DOCX / PPTX / OCR / URL / ZIP | yes | no |
| plagiarism, batch comparison, code similarity | yes | no |
| sentence heatmap and token view | yes | no (no per-sentence calibration) |
| history, PDF/CSV reports | yes | no (session storage only) |
| evaluation page numbers | yes | no — the page says so |

The UI disables what it cannot do and shows the server's own reason, so a Lite visitor is
never shown a disabled control without an explanation.

## 4. Rollback

* Space: the previous revision is still in the Space's history ("Settings ▸ Roll back").
* Vercel: instant rollback to the previous deployment from the dashboard.
* The old Vercel project and old Space referenced by the specification are **not** deleted.

## 5. Post-deploy checklist

```bash
BASE_URL=https://<user>-veritas-api.hf.space bash scripts/smoke.sh
curl -s -o /dev/null -w '%{http_code}\n' https://<project>.vercel.app/results/nope   # 200 (SPA)
# Lite on Vercel must report status=ok, artifact_source=cpu, classical=true:
curl -s https://<project>.vercel.app/api/health | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["status"], d["artifact_source"], d["models_loaded"]["classical"])'
for p in health samples metrics; do curl -s -o /dev/null -w "$p %{http_code}\n" https://<project>.vercel.app/api/$p; done
curl -s -X POST -H 'content-type: application/json' \
  -d '{"text":"'"$(head -c 1500 ml/bundled/samples/sample_human.txt | tr -d '"\\')"'"}' \
  https://<project>.vercel.app/api/analyze | head -c 200; echo
```

A cold start can answer 500 on the very first request; retry once before treating it as a
regression (the first `/api/health` after a deploy did exactly that here).

Then, in the browser: toggle **Effects → Off** and confirm every page still works; disable
WebGL (`chrome://flags` or the 2D fallback path) and confirm the heatmap, verdict number and
similarity report are unchanged.
