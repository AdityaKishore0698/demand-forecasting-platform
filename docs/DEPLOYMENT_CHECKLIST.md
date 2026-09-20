# Deployment checklist

**Status: not deployed.** Nothing here has been run against a cloud provider. What *has* been verified locally: the Docker image builds, starts, loads its model and returns the same forecast as the local API; the frontend production build passes type-check and unit tests. Only add live URLs to the README after you have completed the steps below and checked them.

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

The Docker image contains **only the synthetic demo bundle** (`artifacts/demo/`). That is the safe default for a public demo: the app shows the banner "Demo mode — using synthetic data" and every number is illustrative.

Do **not** put a bundle trained on the original dataset into a *public* deployment unless you have verified that its terms allow it: the API's `/historical-data` endpoint returns the underlying daily records, and `/backtest` returns real actuals. If you want a private deployment on your own data, build a private image or mount the bundle at runtime and set `ARTIFACT_DIR`; keep it out of git and out of any public registry.

## 1. Backend

Local Docker test first:

```bash
make docker                                   # docker build -t demand-forecasting-api .
docker run --rm -p 8000:8000 -e CORS_ORIGINS=http://localhost:5173 demand-forecasting-api
curl http://localhost:8000/health             # expect "model_loaded": true, "data_label": "synthetic-demo"
```

Render (web service): New → Web Service → connect the GitHub repo → Environment **Docker** → Dockerfile path `./Dockerfile`; Health check path `/health`; set the environment variables below; deploy. Image ≈ 670 MB (Python 3.11-slim + pandas/numpy/LightGBM), which is normal for this stack.

## 2. Environment variables (API)

| Variable | Value | Notes |
|---|---|---|
| `CORS_ORIGINS` | your frontend URL(s), comma-separated, e.g. `https://your-app.vercel.app` | Required in production. Wildcards are not used. Add the URL *after* the frontend is deployed, then redeploy the API. |
| `ARTIFACT_DIR` | `/app/artifacts/demo` (already the image default) | Change only to serve a different bundle. |
| `LOG_LEVEL` | `INFO` | Optional. |
| `PORT` | injected by the host | The container command honours it. |

Frontend (build-time only): `VITE_API_URL=https://<your-api-host>` — no trailing slash.

## 3. Model / data artifacts

| Bundle | In git? | How it is produced |
|---|---|---|
| `artifacts/demo/` (synthetic) | yes, ≈1 MB | `make demo` (deterministic; the committed copy was made with `configs/demo.yaml`) |
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
