from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from tqdm import tqdm

from pipeline.chunker import ChunkingManager
from pipeline.extractor import PDFTextExtractor
from pipeline.generator import LLMGenerator
from pipeline.vectorstore import VectorStoreFactory


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docs.append(Document(page_content=r["text"], metadata=r["metadata"]))
    return docs


def _save_jsonl(documents: List[Document], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents):
            f.write(
                json.dumps({"id": i, "text": doc.page_content, "metadata": doc.metadata})
                + "\n"
            )


# ── RAGPipeline ───────────────────────────────────────────────────────────────

class RAGPipeline:

    def __init__(self, config: dict) -> None:
        self.config = config
        save_dir    = Path(config["paths"]["save_dir"])

        self.pdf_dir              = save_dir / "pdfs"
        self.extracted_text_path  = save_dir / "extracted_text" / "pages.jsonl"
        self.chunks_dir           = save_dir / "chunks"

        for d in (
            self.pdf_dir,
            self.extracted_text_path.parent,
            self.chunks_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        self._generator: LLMGenerator | None = None

    # ── Stages ────────────────────────────────────────────────────────────────

    def extract_text(self) -> List[Document]:
        if self.extracted_text_path.exists():
            print("✅ Loaded existing extracted text.")
            return _load_jsonl(self.extracted_text_path)

        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(
                "No PDFs found. Upload PDFs via POST /api/v1/ingest "
                "or place them in data/pdfs/."
            )

        extractor = PDFTextExtractor()
        all_docs: List[Document] = []
        for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
            all_docs.extend(
                extractor.extract(str(pdf_path), self.extracted_text_path, all_docs)
            )

        # Deduplicate by (paper_id, page)
        seen: set = set()
        unique: List[Document] = []
        for doc in sorted(
            all_docs, key=lambda d: (d.metadata["paper_id"], d.metadata["page"])
        ):
            key = (doc.metadata["paper_id"], doc.metadata["page"])
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        _save_jsonl(unique, self.extracted_text_path)
        return unique

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        strategy   = self.config["chunking"]["strategy"]
        chunk_file = self.chunks_dir / f"{strategy}.jsonl"

        existing = _load_jsonl(chunk_file) if chunk_file.exists() else []
        done = {
            (d.metadata.get("paper_id"), d.metadata.get("page")) for d in existing
        }
        remaining = [
            d for d in documents
            if (d.metadata.get("paper_id"), d.metadata.get("page")) not in done
        ]

        if not remaining:
            print(f"✅ Using existing '{strategy}' chunks.")
            return existing

        chunker = ChunkingManager(
            embedding_model=self.config["chunking"]["embedding_model"],
            llm_model=self.config["chunking"]["llm_model"],
            config=self.config,
        )

        if strategy == "token":
            new = chunker.token_chunking(
                remaining,
                chunk_file,
                self.config["chunking"]["token_chunk_size"],
                self.config["chunking"]["token_chunk_overlap"],
                append=True,
            )
        elif strategy == "semantic":
            new = chunker.semantic_chunking(remaining, chunk_file, append=True)
        elif strategy == "agentic":
            new = chunker.agentic_chunking(remaining, chunk_file, append=True)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        return existing + new

    def build_vectorstore(self, chunks: List[Document]):
        """
        Upload uploaded-policy chunks to the POLICIES Azure AI Search index.
        Returns the retriever (kept for callers, though the compliance flow
        uses ComplianceRetriever directly).
        """
        vs_cfg = self.config["vectorstore"]

        factory = VectorStoreFactory(
            embedding_model=vs_cfg["embedding_model"],
            index_name=vs_cfg.get("policies_index", vs_cfg["index_name"]),
            top_k=vs_cfg["top_k"],
        )
        factory.build_or_load(chunks, force=True)
        return factory.as_retriever()

    def build(self) -> Tuple:
        print("🔹 Building RAG pipeline...")

        # Connect to the existing Azure AI Search index.
        # Extraction and chunking only happen during ingest (POST /api/v1/ingest
        # or scripts/ingest_policy.py) — not on every server startup.
        vs_cfg = self.config["vectorstore"]
        factory = VectorStoreFactory(
            embedding_model=vs_cfg["embedding_model"],
            index_name=vs_cfg["index_name"],
            top_k=vs_cfg["top_k"],
        )
        factory.build_or_load([])  # empty list = connect only, no upload
        retriever = factory.as_retriever()

        self._generator = LLMGenerator(
            llm_model_name=self.config["llm"]["llm_model_name"],
            config=self.config,
        )
        print("🎉 Pipeline ready.")
        return retriever, self._generator