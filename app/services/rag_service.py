from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from app.models.schemas import QueryResponse, SourceChunk, IngestResponse

logger = logging.getLogger(__name__)

class RAGService:
    """Wraps the compiled LangGraph agent for use in API endpoint handlers."""

    def __init__(self, agent, pipeline, config: dict) -> None:
        self.agent = agent
        self.pipeline = pipeline
        self.config = config

    def query(self, question: str) -> QueryResponse:
        """Run the agent against *question* and return a structured response."""
        logger.info("Processing query", extra={"query": question})

        result = self.agent.invoke({"query": question})

        final_answer = result.get("final_answer") or result.get("rag_answer", "")
        retrieval_score = result.get("retrieval_score")
        used_web = bool(result.get("web_results"))

        # Build source chunks from retrieved docs metadata
        source_chunks: List[SourceChunk] = []
        for doc in result.get("retrieved_docs", []):
            if hasattr(doc, "metadata"):
                source_chunks.append(
                    SourceChunk(
                        paper_id=doc.metadata.get("paper_id", "unknown"),
                        page=doc.metadata.get("page", 0),
                        text=doc.page_content[:500],
                    )
                )
            elif isinstance(doc, str):
                source_chunks.append(
                    SourceChunk(paper_id="unknown", page=0, text=doc[:500])
                )

        return QueryResponse(
            query=question,
            answer=final_answer,
            source_chunks=source_chunks,
            retrieval_score=retrieval_score,
            used_web_search=used_web,
        )

    def ingest_pdfs(self, pdf_paths: List[Path]) -> IngestResponse:
        import shutil

        pdf_dir: Path = self.pipeline.pdf_dir
        pdf_dir.mkdir(parents=True, exist_ok=True)

        for path in pdf_paths:
            dest = pdf_dir / path.name
            shutil.copy2(str(path), str(dest))
            logger.info(f"Copied {path.name} → {dest}")

        # Force re-extraction and re-chunking
        documents = self.pipeline.extract_text()
        chunks = self.pipeline.chunk_documents(documents)
        self.pipeline.build_vectorstore(chunks)

        # Rebuild agent with updated retriever
        from agent.graph import build_agent

        retriever = self.pipeline.build_vectorstore(chunks)
        self.agent = build_agent(retriever, self.pipeline._generator, self.config)

        return IngestResponse(
            message=f"Successfully ingested {len(pdf_paths)} PDF(s).",
            files_processed=len(pdf_paths),
            chunks_created=len(chunks),
        )
