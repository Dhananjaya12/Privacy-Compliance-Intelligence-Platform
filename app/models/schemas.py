from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Compliance question to audit")
    policy_document: Optional[str] = Field(
        default=None,
        description="Target policy filename to audit. If omitted, it is inferred "
                    "from the query; ambiguous queries return a clarification.",
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Override default retrieval top-k")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Does the Google policy comply with GDPR Article 17?",
                "policy_document": "google_privacy_policy_latest.pdf",
            }
        }
    }


class SourceChunk(BaseModel):
    paper_id: str
    page: int
    text: str


class ComplianceDetail(BaseModel):
    jurisdictions: List[str] = []
    documents: List[str] = []                        # documents audited
    compliance_score: Optional[float] = None         # 0-100, higher = better
    per_reg_compliance: dict = {}                     # {regulation: 0-100}
    overall_risk: Optional[float] = None             # internal 0-10
    gaps: List[dict] = []
    conflicts: List[dict] = []                        # value-level cross-reg conflicts
    remediations: List[dict] = []
    financial_exposure: str = ""
    gap_groups: List[dict] = []                       # themed findings + remediation, for checklist UI
    obligation_counts: Dict[str, int] = {}            # {regulation: obligation count}, for projected scoring


class QueryResponse(BaseModel):
    query: str
    answer: str                                      # markdown report
    source_chunks: List[SourceChunk] = []
    clarification: Optional[str] = None              # set when the query is ambiguous
    query_intent: str = "audit"                      # "audit" | "coverage"
    compliance: Optional[ComplianceDetail] = None


class IngestResponse(BaseModel):
    message: str
    files_processed: int
    chunks_created: int
    paper_ids: List[str] = []                         # resolved policy filenames


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    pipeline_ready: bool
    version: str


class ErrorResponse(BaseModel):
    detail: str
