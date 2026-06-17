"""
app/api/v1/endpoints/conflicts.py
----------------------------------
GET /api/v1/conflicts — cross-regulation conflicts (CONFLICTS_WITH /
STRICTER_THAN edges from the KG), independent of any single audit's
jurisdiction scope. Used by the Dashboard's global conflicts card.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request, status

logger = logging.getLogger(__name__)
router = APIRouter(tags=["conflicts"])


def _retriever(request: Request):
    r = getattr(request.app.state, "compliance_retriever", None)
    if r is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Retriever not ready.")
    return r


@router.get("/conflicts", summary="Cross-regulation conflicts")
async def conflicts(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
):
    try:
        return {"conflicts": _retriever(request).get_conflicts(limit=limit)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Conflicts fetch failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc
