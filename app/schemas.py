from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieval_score: Optional[float] = None
    used_web_search: bool = False


class IngestResponse(BaseModel):
    message: str
    files_processed: int
    chunks_created: int
