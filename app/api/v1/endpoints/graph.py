"""
app/api/v1/endpoints/graph.py
------------------------------
GET /api/v1/graph — knowledge-graph nodes/edges for the D3.js Graph Explorer.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request, status

logger = logging.getLogger(__name__)
router = APIRouter(tags=["graph"])


def _retriever(request: Request):
    r = getattr(request.app.state, "compliance_retriever", None)
    if r is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Retriever not ready.")
    return r


@router.get("/graph", summary="Knowledge-graph nodes and edges")
async def graph(
    request: Request,
    limit: int = Query(default=200, ge=1, le=1000),
    regulation: str | None = Query(default=None),
):
    try:
        return _retriever(request).get_graph(limit=limit, regulation=regulation)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Graph export failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc
