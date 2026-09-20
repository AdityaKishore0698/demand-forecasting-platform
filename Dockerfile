# API image: serves forecasts from a pre-built model bundle (no training at runtime).
# By default it contains ONLY the synthetic demo bundle (artifacts/demo). To serve a bundle trained on
# your own data, mount/copy it and set ARTIFACT_DIR (see docs/DEPLOYMENT_CHECKLIST.md).
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

ENV ARTIFACT_DIR=/app/artifacts/demo
EXPOSE 8000

# $PORT is honoured for platforms that inject it (Render, Railway, Cloud Run).
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
