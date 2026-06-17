from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Build the pipeline on startup; clean up on shutdown."""
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    logger.info("Starting up PDF-RAG API…")

    # Lazy imports — keep startup fast when running tests without GPU deps
    from agent.graph import build_agent
    from pipeline.compliance_retriever import ComplianceRetriever
    from pipeline.rag_pipeline import RAGPipeline

    config = settings.as_pipeline_config()

    # RAGPipeline is now policy-ingestion only (uploads → policies index).
    pipeline = RAGPipeline(config)

    compliance_retriever = ComplianceRetriever()
    agent_app = build_agent(config, compliance_retriever)

    # Attach to app.state for dependency injection
    app.state.agent = agent_app
    app.state.config = config
    app.state.pipeline = pipeline
    app.state.compliance_retriever = compliance_retriever
    app.state.ready = True

    logger.info("Pipeline ready — serving requests.")

    yield  # application runs here

    logger.info("Shutting down PDF-RAG API.")
    app.state.ready = False
