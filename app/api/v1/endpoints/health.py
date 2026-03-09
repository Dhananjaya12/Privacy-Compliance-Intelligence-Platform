"""
app/api/v1/endpoints/health.py
-------------------------------
Liveness and readiness probes for Kubernetes.
"""

from fastapi import APIRouter, Request
from app.models.schemas import HealthResponse, ReadinessResponse
from app.core.config import get_settings

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def liveness():
    """Always returns 200 if the process is alive."""
    return HealthResponse(status="ok", version=settings.APP_VERSION)


@router.get("/health/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def readiness(request: Request):
    """Returns 200 only when the RAG pipeline has finished loading."""
    ready = getattr(request.app.state, "ready", False)
    return ReadinessResponse(
        status="ready" if ready else "loading",
        pipeline_ready=ready,
        version=settings.APP_VERSION,
    )
