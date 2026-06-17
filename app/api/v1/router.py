from fastapi import APIRouter

from app.api.v1.endpoints import (
    conflicts,
    graph,
    health,
    history,
    ingest,
    query,
    regulations,
    report,
)

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(query.router)
api_router.include_router(ingest.router)
api_router.include_router(report.router)
api_router.include_router(history.router)
api_router.include_router(graph.router)
api_router.include_router(conflicts.router)
api_router.include_router(regulations.router)
