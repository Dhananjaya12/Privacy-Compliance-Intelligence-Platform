"""
app/api/v1/endpoints/regulations.py
------------------------------------
GET /api/v1/regulations          — list indexed regulations.
GET /api/v1/regulations/chunks   — browse regulation text (optionally filtered).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request, status

logger = logging.getLogger(__name__)
router = APIRouter(tags=["regulations"])


def _retriever(request: Request):
    r = getattr(request.app.state, "compliance_retriever", None)
    if r is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Retriever not ready.")
    return r


@router.get("/regulations", summary="List indexed regulations")
async def list_regulations(request: Request):
    try:
        return {"regulations": _retriever(request).list_regulation_documents()}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Regulation listing failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc


@router.get("/regulations/chunks", summary="Browse regulation text")
async def regulation_chunks(
    request: Request,
    regulation: str | None = Query(default=None),
    top: int = Query(default=50, ge=1, le=200),
):
    try:
        return {"chunks": _retriever(request).get_regulation_chunks(regulation=regulation, top=top)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Regulation browse failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc
