"""
app/api/v1/endpoints/ingest.py
-------------------------------
POST /api/v1/ingest  — upload one or more PDFs and trigger ingestion.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.models.schemas import IngestResponse
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])

MAX_FILE_SIZE_MB = 50
ALLOWED_CONTENT_TYPES = {"application/pdf"}


def _get_rag_service(request: Request) -> RAGService:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline is still loading.",
        )
    return RAGService(
        agent=request.app.state.agent,
        pipeline=request.app.state.pipeline,
        config=request.app.state.config,
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest PDF documents",
    description="Upload one or more PDF files. They will be extracted, chunked, and indexed into the vector store.",
)
async def ingest_endpoint(
    files: List[UploadFile] = File(..., description="PDF files to ingest"),
    service: RAGService = Depends(_get_rag_service),
):
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    saved_paths: List[Path] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for upload in files:
            # Validate content type
            if upload.content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"File '{upload.filename}' is not a PDF.",
                )

            content = await upload.read()

            # Validate file size
            if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File '{upload.filename}' exceeds {MAX_FILE_SIZE_MB} MB limit.",
                )

            dest = Path(tmpdir) / (upload.filename or "upload.pdf")
            dest.write_bytes(content)
            saved_paths.append(dest)
            logger.info(f"Saved upload: {dest}")

        try:
            return service.ingest_pdfs(saved_paths)
        except Exception as exc:
            logger.exception("Ingest failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
