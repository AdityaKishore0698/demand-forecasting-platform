from fastapi import APIRouter, Depends, Request

from api.deps import get_service
from api.schemas import HealthResponse
from src.forecasting.service import ForecastService

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness / readiness")
def health(request: Request) -> HealthResponse:
    service = getattr(request.app.state, "service", None)
    if service is None:
        return HealthResponse(status="degraded", model_loaded=False,
                              detail=getattr(request.app.state, "load_error", None))
    return HealthResponse(
        status="ok", model_loaded=True, model_version=service.model_version,
        data_through=service.origin.date(), forecast_horizon_days=service.horizon_days, data_label=service.data_label,
        model_type=service.model_type_label,
        model_components=[c["family"] for c in service.card["components"]] if "components" in service.card else ["LightGBM"],
    )


@router.get("/model-info", summary="Model card: algorithm, parameters, features, training window")
def model_info(service: ForecastService = Depends(get_service)):
    return service.model_info()


@router.get("/dataset-info", summary="Descriptive facts about the dataset the model was trained on")
def dataset_info(service: ForecastService = Depends(get_service)):
    return service.dataset_summary
