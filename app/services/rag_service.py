from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Iterator, List, Optional

from app.models.schemas import (
    ComplianceDetail,
    IngestResponse,
    QueryResponse,
    SourceChunk,
)

logger = logging.getLogger(__name__)

# Human-readable labels for the agent's LangGraph nodes, surfaced to the UI
# while a query streams so the user can see what's happening behind the scenes.
NODE_LABELS = {
    "doc_resolver":          "Identifying target document(s)",
    "jurisdiction_detector": "Detecting applicable regulations",
    "kg_retriever":          "Retrieving obligations from knowledge graph",
    "gap_analyzer":          "Analyzing compliance gaps",
    "conflict_detector":     "Checking cross-regulation conflicts",
    "risk_scorer":           "Calculating compliance score",
    "remediation":           "Generating remediation recommendations",
    "report_generator":      "Generating report",
}


def _sse(payload: dict) -> str:
    """Format a dict as one SSE `data:` frame."""
    return f"data: {json.dumps(payload)}\n\n"


class RAGService:
    """Wraps the compiled compliance LangGraph agent for API endpoint handlers."""

    def __init__(self, agent, pipeline, config: dict) -> None:
        self.agent    = agent
        self.pipeline = pipeline
        self.config   = config

    def query(self, question: str, policy_document: Optional[str] = None) -> QueryResponse:
        """Run the compliance agent and return a structured response. MLflow-tracked."""
        logger.info("Processing compliance query", extra={"query": question})

        result = self.agent.invoke({"query": question, "policy_document": policy_document})

        self._track_mlflow(question, result, policy_document)
        return self._build_response(question, result)

    def query_stream(self, question: str, policy_document: Optional[str] = None) -> Iterator[str]:
        """Run the compliance agent step-by-step, yielding SSE progress frames.

        Each frame is a complete `data: {...}\\n\\n` string. Intermediate frames
        report which pipeline stage just finished (`type: "progress"`); the
        final frame (`type: "done"`) carries the full query result, matching
        the shape of `query()`'s response.
        """
        logger.info("Streaming compliance query", extra={"query": question})

        initial_state = {"query": question, "policy_document": policy_document}
        result: Optional[dict] = None

        try:
            for update in self.agent.stream(initial_state, stream_mode="updates"):
                for node_name, node_state in update.items():
                    result = node_state
                    yield _sse({
                        "type":  "progress",
                        "node":  node_name,
                        "label": NODE_LABELS.get(node_name, node_name),
                    })
        except Exception as exc:
            logger.error("Streaming query failed: %s", exc, exc_info=True)
            yield _sse({"type": "error", "message": str(exc)})
            return

        if result is None:
            yield _sse({"type": "error", "message": "Agent produced no output"})
            return

        self._track_mlflow(question, result, policy_document)
        response = self._build_response(question, result)
        yield _sse({"type": "done", "result": response.model_dump()})

    def _build_response(self, question: str, result: dict) -> QueryResponse:
        """Translate the agent's final state into a QueryResponse."""
        final_answer  = result.get("final_answer") or result.get("compliance_report", "")
        clarification = result.get("clarification_needed")
        targets       = result.get("target_documents", [])

        # Source chunks from the primary document's retrieved policy chunks.
        source_chunks: List[SourceChunk] = []
        for c in result.get("kg_chunks", []):
            meta = c.get("metadata", {}) if isinstance(c, dict) else {}
            source_chunks.append(
                SourceChunk(
                    paper_id=meta.get("paper_id", "unknown"),
                    page=int(meta.get("page", 0) or 0),
                    text=(c.get("page_content", "") if isinstance(c, dict) else "")[:500],
                )
            )

        compliance = None
        if result.get("jurisdictions") is not None and not clarification:
            # Themed gap groups + per-regulation obligation counts come from the
            # primary audited document, for the checklist + projected-score UI.
            primary_data = result.get("per_doc_results", {}).get(targets[0], {}) if targets else {}
            gap_groups = primary_data.get("gap_groups", [])
            obligation_counts: dict = {}
            for ob in primary_data.get("obligations", []):
                reg = ob.get("regulation")
                if reg:
                    obligation_counts[reg] = obligation_counts.get(reg, 0) + 1

            compliance = ComplianceDetail(
                jurisdictions      = result.get("jurisdictions", []),
                documents          = targets,
                compliance_score   = result.get("compliance_score"),
                per_reg_compliance = result.get("per_reg_compliance", {}),
                overall_risk       = result.get("overall_score"),
                gaps               = result.get("gaps", []),
                conflicts          = result.get("conflicts", []),
                remediations       = result.get("remediations", []),
                financial_exposure = result.get("financial_exposure", ""),
                gap_groups         = gap_groups,
                obligation_counts  = obligation_counts,
            )

        return QueryResponse(
            query         = question,
            answer        = final_answer,
            source_chunks = source_chunks,
            clarification = clarification,
            query_intent  = result.get("query_intent", "audit"),
            compliance    = compliance,
        )

    def _track_mlflow(self, question: str, result: dict, policy_document: Optional[str]) -> None:
        """Log a successful audit run to MLflow (non-fatal on failure)."""
        if result.get("jurisdictions") is None or result.get("clarification_needed"):
            return

        targets = result.get("target_documents", [])
        try:
            from mlops.compliance_tracker import get_tracker
            policy_name = targets[0] if targets else (policy_document or "unknown")
            tracker = get_tracker()
            with tracker.start_run(query=question, policy_name=policy_name):
                tracker.log_result(result)
        except Exception as e:
            logger.warning("MLflow tracking failed (non-fatal): %s", e)

    def ingest_pdfs(self, pdf_paths: List[Path]) -> IngestResponse:
        """Ingest uploaded policy PDFs into the policies index."""
        pdf_dir: Path = self.pipeline.pdf_dir
        pdf_dir.mkdir(parents=True, exist_ok=True)

        paper_ids: List[str] = []
        for path in pdf_paths:
            dest = pdf_dir / path.name
            shutil.copy2(str(path), str(dest))
            paper_ids.append(path.name)
            logger.info(f"Copied {path.name} → {dest}")

        documents = self.pipeline.extract_text()
        chunks    = self.pipeline.chunk_documents(documents)

        # Stamp policy metadata so the retriever can scope by document.
        for i, ch in enumerate(chunks):
            ch.metadata.setdefault("doc_type", "policy")
            ch.metadata.setdefault("regulation", "UNKNOWN")
            ch.metadata.setdefault(
                "chunk_id", f"{ch.metadata.get('paper_id','doc')}:{i}"
            )

        self.pipeline.build_vectorstore(chunks)

        # ── Log ingestion run to MLflow ───────────────────────────────────────
        try:
            from mlops.compliance_tracker import get_tracker
            get_tracker().log_ingestion_run(
                input_dir    = str(pdf_dir),
                strategy     = self.config.get("chunking", {}).get("strategy", "token"),
                total_files  = len(pdf_paths),
                total_chunks = len(chunks),
                uploaded     = len(chunks),
                errors       = 0,
                latency_ms   = 0,
            )
        except Exception as e:
            logger.warning("MLflow ingestion tracking failed (non-fatal): %s", e)

        return IngestResponse(
            message         = f"Successfully ingested {len(pdf_paths)} policy PDF(s).",
            files_processed = len(pdf_paths),
            chunks_created  = len(chunks),
            paper_ids       = paper_ids,
        )
