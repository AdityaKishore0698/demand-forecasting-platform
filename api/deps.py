from fastapi import HTTPException, Request

from src.forecasting.service import ForecastService


def get_service(request: Request) -> ForecastService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(request.app.state, "load_error", None) or "Model artifacts are not loaded.",
        )
    return service
