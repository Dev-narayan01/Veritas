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

The Vercel project is `veritas`; the site is built from `frontend/` with the repository
root `api/` folder exposed as Python functions. `frontend/vercel.json` rewrites every
non-`/api` path to `index.html` (SPA routing) and caches hashed assets immutably.

```bash
bash scripts/deploy_vercel.sh            # preview
bash scripts/deploy_vercel.sh --prod     # production
```

Environment variable in the Vercel project:

* `VITE_API_BASE_URL` = `https://<user>-veritas-api.hf.space`

`api/requirements.txt` pins only `fastapi`, `numpy`, `scikit-learn` and `joblib`. The Lite
functions must not import the `app` package; `api/_lib/core.py` installs a three-module
pickle shim so `classical.pkl` loads without it.

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
```

Then, in the browser: toggle **Effects → Off** and confirm every page still works; disable
WebGL (`chrome://flags` or the 2D fallback path) and confirm the heatmap, verdict number and
similarity report are unchanged.
