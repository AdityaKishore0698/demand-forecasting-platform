"""FastAPI application.

Run locally:  ``uvicorn api.main:app --reload``   (docs at /docs)

Environment variables (see ``.env.example``):
  ARTIFACT_DIR   directory holding the trained bundle   (default: ./artifacts if trained locally,
                 otherwise the synthetic demo bundle ./artifacts/demo)
  CORS_ORIGINS   comma-separated allowed browser origins (default: local Vite dev server)
  LOG_LEVEL      python logging level                    (default: INFO)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import __version__
from api.routers import forecast, insights, stores, system
from src.forecasting.artifacts import default_artifact_dir
from src.forecasting.service import (
    ArtifactsNotFound, ForecastRequestError, ForecastService, PredictionIntegrityError, StoreNotFound,
)

logger = logging.getLogger("api")


def _origins() -> list:
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app(artifact_dir: Optional[Path] = None) -> FastAPI:
    artifact_path = Path(artifact_dir or os.environ.get("ARTIFACT_DIR") or default_artifact_dir())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
        app.state.service = None
        app.state.load_error = None
        try:
            app.state.service = ForecastService.load(artifact_path)   # loaded ONCE, reused per request
            logger.info("Loaded model %s from %s", app.state.service.model_version, artifact_path)
        except ArtifactsNotFound as exc:
            app.state.load_error = str(exc)
            logger.error("%s", exc)
        yield

    app = FastAPI(
        title="Demand Forecasting & Analytics API",
        version=__version__,
        description="Store-level daily demand forecasts, hold-out evaluation and model interpretation.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=_origins(), allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
    )

    @app.exception_handler(StoreNotFound)
    async def _store_not_found(_: Request, exc: StoreNotFound):
        return JSONResponse(status_code=404, content={"detail": exc.args[0]})

    @app.exception_handler(ForecastRequestError)
    async def _bad_request(_: Request, exc: ForecastRequestError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(PredictionIntegrityError)
    async def _integrity(_: Request, exc: PredictionIntegrityError):
        logger.error("Prediction integrity check failed: %s", exc)
        return JSONResponse(status_code=500, content={"detail": f"Forecast rejected by an internal consistency check: {exc}"})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    for r in (system.router, stores.router, forecast.router, insights.router):
        app.include_router(r)

    @app.get("/", include_in_schema=False)
    def root():
        return {"name": app.title, "version": __version__, "docs": "/docs", "health": "/health"}

    return app


app = create_app()
