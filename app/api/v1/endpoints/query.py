"""
app/api/v1/endpoints/query.py
------------------------------
POST /api/v1/query  — ask a question against the ingested PDFs.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


def _get_rag_service(request: Request) -> RAGService:
    """Dependency: pull the RAGService off app.state."""
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline is still loading. Please retry in a moment.",
        )
    return RAGService(
        agent=request.app.state.agent,
        pipeline=request.app.state.pipeline,
        config=request.app.state.config,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the RAG agent",
    description="Submit a natural-language question. The agent retrieves relevant PDF chunks, optionally performs a web search, and returns a synthesised answer.",
)
async def query_endpoint(
    body: QueryRequest,
    service: RAGService = Depends(_get_rag_service),
):
    try:
        return service.query(body.query)
    except Exception as exc:
        logger.exception("Error during query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
