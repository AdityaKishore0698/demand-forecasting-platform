# API image: serves forecasts from a pre-built model bundle (no training at runtime).
# Default bundle: artifacts/demo = the final 3-model ensemble (LightGBM + CatBoost + XGBoost) trained on SYNTHETIC
# data. artifacts/demo_single is the previous single-LightGBM synthetic bundle, kept as a rollback target:
#   set ARTIFACT_DIR=/app/artifacts/demo_single on the host to roll back without rebuilding.
# To serve a bundle trained on your own data, mount/copy it and set ARTIFACT_DIR (docs/DEPLOYMENT_CHECKLIST.md).
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY configs ./configs
COPY artifacts/demo ./artifacts/demo
COPY artifacts/demo_single ./artifacts/demo_single

ENV ARTIFACT_DIR=/app/artifacts/demo
EXPOSE 8000

# $PORT is honoured for platforms that inject it (Render, Railway, Cloud Run).
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
