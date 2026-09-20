# Deployment checklist

**Status: the ensemble build has been verified locally but not redeployed.** Verified locally: the Docker image builds (≈1.31 GB on disk), the container answers `/health` 1.8 s after start with the 3-model ensemble loaded (≈254 MiB idle), every endpoint responds correctly, and the frontend passes its browser checks against that container. Render deploys from the GitHub repository, so a redeploy needs the new commit pushed, and the Vercel frontend needs redeploying to pick up the ensemble wording. **Compatibility with your Render instance's memory/CPU limits has not been verified** — check the instance limits in the dashboard against the measurements below before switching.

## Architecture (two pieces, nothing else)

```
Browser ── static React app (Vite build, VITE_API_URL baked in) ──► FastAPI container ──► model bundle (loaded once)
```

No database, cache, queue, auth or Kubernetes. The API is read-only and stateless.

## Recommended hosts and why

| Piece | Suggestion | Why it fits |
|---|---|---|
| API | **Render** (Docker web service) — Railway / Fly.io / Cloud Run work the same way | Deploys straight from the repo's `Dockerfile`, honours `$PORT`, health-check path, free tier available. Trade-off: free instances sleep when idle, so the first request after a pause is slow (start-up + model load). |
| Frontend | **Vercel** (or Netlify / Cloudflare Pages) | Static hosting for a Vite app; `frontend/vercel.json` already rewrites deep links to `index.html`. |

## 0. Decide what the public deployment serves — do this first

The Docker image contains **only synthetic bundles**: `artifacts/demo/` (the final 3-model ensemble, ≈22 MB) and `artifacts/demo_single/` (the previous single-LightGBM bundle, kept as a rollback target). That is the safe default for a public demo: the app shows the banner "Demo mode — using synthetic data" and every number is illustrative.

Do **not** put a bundle trained on the original dataset into a *public* deployment unless you have verified that its terms allow it: the API's `/historical-data` endpoint returns the underlying daily records, and `/backtest` returns real actuals. If you want a private deployment on your own data, build a private image or mount the bundle at runtime and set `ARTIFACT_DIR`; keep it out of git and out of any public registry.

## 1. Backend

Local Docker test first:

```bash
make docker                                   # docker build -t demand-forecasting-api .
docker run --rm -p 8000:8000 -e CORS_ORIGINS=http://localhost:5173 demand-forecasting-api
curl http://localhost:8000/health             # expect "model_loaded": true, "data_label": "synthetic-demo"
```

Render (web service): New → Web Service → connect the GitHub repo → Environment **Docker** → Dockerfile path `./Dockerfile`; Health check path `/health`; set the environment variables below; deploy. Image ≈ 1.31 GB on disk (Python 3.11-slim + pandas/numpy/LightGBM/CatBoost/XGBoost; CatBoost and its dependencies add ≈ 450 MB of packages). Measured locally in Docker with the demo bundle: ready in 1.8 s, ≈ 254 MiB after start-up, ≈ 261 MiB after 60 forecasts, ≈ 19 ms for an uncached 42-day forecast. Models trained on the real dataset are larger (whole-API process memory measured locally: 548 MB single LightGBM vs 710 MB ensemble, both including the 970k-row history), but those are not deployed publicly.

## 2. Environment variables (API)

| Variable | Value | Notes |
|---|---|---|
| `CORS_ORIGINS` | your frontend URL(s), comma-separated, e.g. `https://your-app.vercel.app` | Required in production. Wildcards are not used. Add the URL *after* the frontend is deployed, then redeploy the API. |
| `ARTIFACT_DIR` | `/app/artifacts/demo` (already the image default) | **Rollback:** set to `/app/artifacts/demo_single` to serve the previous single-LightGBM bundle without rebuilding the image. Also used to serve a different bundle. |
| `LOG_LEVEL` | `INFO` | Optional. |
| `PORT` | injected by the host | The container command honours it. |

Frontend (build-time only): `VITE_API_URL=https://<your-api-host>` — no trailing slash.

## 3. Model / data artifacts

| Bundle | In git? | How it is produced |
|---|---|---|
| `artifacts/demo/` (synthetic ensemble) | yes, ≈22 MB | `make demo` (≈15 min; the committed copy was made with `configs/demo.yaml`; the three model files are hash-pinned in `model_card.json`) |
| `artifacts/demo_single/` (synthetic single LightGBM) | yes, ≈1 MB | the previous demo bundle, kept for rollback |
| `artifacts/` (your data) | **no** (git-ignored) | `make train` after placing your data in `data/raw/` |

To refresh the committed demo bundle: `make demo`, then review `git status` and commit yourself.

## 4. Frontend

```bash
cd frontend
npm ci
VITE_API_URL=https://<your-api-host> npm run build     # → frontend/dist  (runs type-check first)
```

Vercel: import the repo → **Root Directory** `frontend` → framework preset **Vite** → env var `VITE_API_URL` → deploy. On Netlify add a `_redirects` file containing `/* /index.html 200` so `/forecast` etc. survive a refresh.

## 5. Health check and end-to-end test

- [ ] `GET https://<api>/health` → `"status":"ok"`, `"model_loaded":true`, expected `model_version`
- [ ] `GET https://<api>/docs` loads
- [ ] `curl -X POST https://<api>/forecast -H 'content-type: application/json' -d '{"store_id":1,"horizon":14}'` → 14 rows
- [ ] Invalid input returns 422 (`"horizon":99`), unknown store returns 404
- [ ] Open the frontend URL: Overview loads with charts, **no CORS errors** in the browser console
- [ ] Forecast page: change store and horizon, open a row's driver panel, export CSV
- [ ] Toggle light/dark, reload — the theme persists
- [ ] Wait until the API instance sleeps (free tier), reload — you get a working page (slow first response), not a broken one

## 6. Add the live URLs to the README

Only after section 5 passes: add a "Live demo" line near the top of `README.md` with the frontend URL and the API `/docs` URL, change the README status line from "Not deployed" to what is actually true (say it runs on synthetic demo data), and add the URL to the resume entry.

## 7. What this deployment does not include

Authentication, rate limiting, monitoring/alerting, scheduled retraining, persistent storage. For a portfolio demo that is intentional; see `docs/ARCHITECTURE.md` §10 for what production would need.
