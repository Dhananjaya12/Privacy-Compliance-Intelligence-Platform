from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Question to ask the RAG agent")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Override default retrieval top-k")

    model_config = {"json_schema_extra": {"example": {"query": "What is the transformer architecture?"}}}

class SourceChunk(BaseModel):
    paper_id: str
    page: int
    text: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    source_chunks: List[SourceChunk] = []
    retrieval_score: Optional[float] = None
    used_web_search: bool = False

class IngestResponse(BaseModel):
    message: str
    files_processed: int
    chunks_created: int

class HealthResponse(BaseModel):
    status: str
    version: str

class ReadinessResponse(BaseModel):
    status: str
    pipeline_ready: bool
    version: str

class ErrorResponse(BaseModel):
    detail: str
