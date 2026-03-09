"""
app/api/v1/router.py
--------------------
Aggregates all v1 endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import health, ingest, query

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(query.router)
api_router.include_router(ingest.router)
