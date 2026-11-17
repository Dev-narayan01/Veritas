# Steps only a person can do

Everything else is scripted. These need accounts, billing, or a GPU, so they are listed in
order with the exact commands and the expected output.

## 1. Create `.env.deploy`

```bash
cp .env.deploy.example .env.deploy
# fill in HF_USERNAME, HF_TOKEN, VITE_API_BASE_URL
```

`HF_TOKEN` must be a **write** token (Settings ▸ Access Tokens ▸ New token ▸ "Write").
Never paste it into a chat, a commit, or a document.

## 2. Publish the artifacts and the Space

```bash
.venv/bin/python ml/run_all.py
bash scripts/deploy_hub.sh
bash scripts/deploy_space.sh --push
```

Expected: a model repo URL, then `pushed to https://huggingface.co/spaces/<user>/veritas-api`.
The first Space build takes 5–10 minutes (it installs the Python requirements).

## 3. Set the Space variables

In the Space: **Settings ▸ Variables and secrets**. Add the four variables from
`docs/DEPLOY.md` §2. Without `CORS_ORIGINS` the browser will block the API.

## 4. Deploy the frontend

```bash
bash scripts/deploy_vercel.sh --prod
```

Set `VITE_API_BASE_URL` in the Vercel project (Settings ▸ Environment Variables) **before**
deploying, or the build will fall back to Lite mode.

## 5. Run the optional GPU pipeline on Kaggle

Needed only if you want Transformer / Binoculars / Fast-DetectGPT numbers and
cross-generator transfer. See `docs/KAGGLE.md`. It produces a bundle whose manifest says
`source: kaggle`; drop it into `ml/artifacts/` and redeploy the Space.

## 6. Verify in a real browser

These checks cannot be automated from this sandbox because the Playwright browser
download (`cdn.playwright.dev`) is unreachable here.

```bash
cd frontend
npx playwright install chromium
npx playwright test
```

Then manually:

1. **Effects → Off** — every page must render identically in content.
2. **WebGL disabled** — in Chrome, `chrome://flags/#disable-webgl`; the verdict number,
   heatmap and similarity report must be unchanged, only the 3D decoration is replaced by
   the 2D fallback.
3. **Wake-up path** — open the deployed site after the Space has slept; the banner should
   say it is starting the server, then switch to full mode.
4. **Lite path** — temporarily point `VITE_API_BASE_URL` at an invalid host; the site must
   show "server offline" and, if the Vercel functions answer, "Lite deployment · pasted
   text only".
5. **Mobile** — the results page must keep the verdict number and percentage visible above
   the fold.

## 7. Review the honesty clauses

Before submitting, read `docs/REPORT.md` §Limitations and confirm every claim matches what
`/api/metrics` serves from your own artifacts. If you re-ran the pipeline, the numbers in
the report may need updating: the report quotes a specific artifact snapshot (see its
header).
