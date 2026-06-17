"""
tests/test_4_pipeline.py

Layer 4 — Pipeline unit tests.
Mocks Azure, HuggingFace, Neo4j — tests pipeline logic only.

Run:
    python -m pytest tests/test_4_pipeline.py -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from langchain_core.documents import Document


# ── RAGPipeline ───────────────────────────────────────────────────────────────

class TestRAGPipeline:

    def _make_config(self, tmp_path: Path) -> dict:
        return {
            "paths": {"save_dir": str(tmp_path)},
            "chunking": {
                "strategy":           "token",
                "token_chunk_size":   512,
                "token_chunk_overlap": 100,
                "embedding_model":    "sentence-transformers/all-MiniLM-L6-v2",
                "llm_model":          "meta-llama/Llama-3.1-8B-Instruct",
            },
            "vectorstore": {
                "endpoint":        "https://fake.search.windows.net",
                "key":             "fakekey",
                "index_name":      "pdf-rag-index",
                "top_k":           5,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "llm": {
                "llm_model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "max_new_tokens": 300,
                "temperature":    1e-9,
            },
        }

    def test_init_creates_directories(self, tmp_path):
        from pipeline.rag_pipeline import RAGPipeline
        config = self._make_config(tmp_path)
        pipeline = RAGPipeline(config)
        assert pipeline.pdf_dir.exists()
        assert pipeline.extracted_text_path.parent.exists()
        assert pipeline.chunks_dir.exists()

    def test_extract_text_returns_cached_if_exists(self, tmp_path):
        from pipeline.rag_pipeline import RAGPipeline
        config = self._make_config(tmp_path)
        pipeline = RAGPipeline(config)

        # Write a fake extracted_text JSONL
        pipeline.extracted_text_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.extracted_text_path.write_text(
            json.dumps({"text": "cached content", "metadata": {"paper_id": "a.pdf", "page": 1}}) + "\n",
            encoding="utf-8",
        )

        docs = pipeline.extract_text()
        assert len(docs) == 1
        assert docs[0].page_content == "cached content"

    def test_extract_text_raises_when_no_pdfs(self, tmp_path):
        from pipeline.rag_pipeline import RAGPipeline
        config = self._make_config(tmp_path)
        pipeline = RAGPipeline(config)
        with pytest.raises(ValueError, match="No PDFs found"):
            pipeline.extract_text()

    def test_chunk_documents_returns_cached(self, tmp_path):
        from pipeline.rag_pipeline import RAGPipeline
        config = self._make_config(tmp_path)
        pipeline = RAGPipeline(config)
        pipeline.chunks_dir.mkdir(parents=True, exist_ok=True)

        chunk_file = pipeline.chunks_dir / "token.jsonl"
        chunk_file.write_text(
            json.dumps({"text": "chunk 1", "metadata": {"paper_id": "a.pdf", "page": 1,
                        "chunk_strategy": "token", "section_id": 0, "doc_index": 0}}) + "\n",
            encoding="utf-8",
        )

        docs = [Document(page_content="original", metadata={"paper_id": "a.pdf", "page": 1})]
        result = pipeline.chunk_documents(docs)
        assert len(result) == 1
        assert result[0].page_content == "chunk 1"

    def test_build_vectorstore_calls_factory(self, tmp_path):
        from pipeline.rag_pipeline import RAGPipeline
        config = self._make_config(tmp_path)
        pipeline = RAGPipeline(config)

        mock_factory = MagicMock()
        mock_factory.as_retriever.return_value = MagicMock()

        with patch("pipeline.rag_pipeline.VectorStoreFactory", return_value=mock_factory) as MockFactory:
            chunks = [Document(page_content="c", metadata={})]
            pipeline.build_vectorstore(chunks)

        MockFactory.assert_called_once_with(
            embedding_model=config["vectorstore"]["embedding_model"],
            index_name=config["vectorstore"]["index_name"],
            top_k=config["vectorstore"]["top_k"],
        )
        mock_factory.build_or_load.assert_called_once_with(chunks)
        mock_factory.as_retriever.assert_called_once()

    def test_build_vectorstore_no_faiss_chroma_args(self, tmp_path):
        """build_vectorstore must not pass vs_type, persist_dir, or pgvector_connection."""
        from pipeline.rag_pipeline import RAGPipeline
        config = self._make_config(tmp_path)
        pipeline = RAGPipeline(config)

        with patch("pipeline.rag_pipeline.VectorStoreFactory") as MockFactory:
            MockFactory.return_value.as_retriever.return_value = MagicMock()
            pipeline.build_vectorstore([])

        call_kwargs = MockFactory.call_args.kwargs
        assert "vs_type"            not in call_kwargs
        assert "persist_dir"        not in call_kwargs
        assert "pgvector_connection" not in call_kwargs


# ── PDFTextExtractor ──────────────────────────────────────────────────────────

class TestPDFTextExtractor:

    def test_init_raises_without_credentials(self):
        from pipeline.extractor import PDFTextExtractor
        with patch.dict("os.environ", {}, clear=True):
            # Remove the Azure keys so __init__ raises
            import os
            os.environ.pop("AZURE_DOC_INTEL_ENDPOINT", None)
            os.environ.pop("AZURE_DOC_INTEL_KEY", None)
            with pytest.raises(ValueError, match="AZURE_DOC_INTEL"):
                PDFTextExtractor()

    def test_is_scanned_detection(self, tmp_path):
        """_is_scanned returns True for PDFs with very little text."""
        from pipeline.extractor import _is_scanned

        # Create a mock fitz document with very little text
        mock_page = MagicMock()
        mock_page.get_text.return_value = "   "  # effectively blank

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__iter__.return_value = iter([mock_page])
        mock_doc.__getitem__ = lambda self, i: mock_page

        with patch("pipeline.extractor.fitz.open", return_value=mock_doc):
            result = _is_scanned("fake.pdf")

        assert result is True

    def test_extract_returns_cached_if_exists(self, tmp_path):
        from pipeline.extractor import PDFTextExtractor
        with patch.dict("os.environ", {
            "AZURE_DOC_INTEL_ENDPOINT": "https://fake.cognitiveservices.azure.com",
            "AZURE_DOC_INTEL_KEY": "fakekey",
        }):
            with patch("pipeline.extractor.DocumentIntelligenceClient"):
                extractor = PDFTextExtractor()

        existing_doc = Document(
            page_content="already extracted",
            metadata={"paper_id": "policy.pdf", "page": 1},
        )
        result = extractor.extract("policy.pdf", tmp_path / "out.jsonl", [existing_doc])
        assert len(result) == 1
        assert result[0].page_content == "already extracted"


# ── ChunkingManager — token chunking ─────────────────────────────────────────

class TestChunkingManagerToken:

    def test_token_chunking_produces_chunks(self, tmp_path):
        from pipeline.chunker import ChunkingManager

        docs = [Document(
            page_content="word " * 200,  # enough tokens to split
            metadata={"paper_id": "a.pdf", "page": 1},
        )]
        output_path = tmp_path / "token.jsonl"

        chunker = ChunkingManager(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            config={"llm": {"temperature": 0.0, "max_new_tokens": 100}},
        )

        # Mock the TokenTextSplitter to avoid needing sentencetransformers
        mock_chunks = [
            Document(page_content="part 1", metadata={}),
            Document(page_content="part 2", metadata={}),
        ]
        with patch("pipeline.chunker.TokenTextSplitter") as MockSplitter:
            MockSplitter.return_value.split_documents.return_value = mock_chunks
            result = chunker.token_chunking(docs, output_path, chunk_size=50, chunk_overlap=10)

        assert len(result) == 2
        assert result[0].metadata["chunk_strategy"] == "token"
        assert result[0].metadata["section_id"] == 0
        assert result[1].metadata["section_id"] == 1

    def test_token_chunking_writes_jsonl(self, tmp_path):
        from pipeline.chunker import ChunkingManager
        docs = [Document(page_content="hello world", metadata={"paper_id": "x.pdf", "page": 1})]
        output_path = tmp_path / "token.jsonl"

        chunker = ChunkingManager(config={"llm": {"temperature": 0.0, "max_new_tokens": 100}})
        mock_chunk = [Document(page_content="hello world", metadata={})]

        with patch("pipeline.chunker.TokenTextSplitter") as MockSplitter:
            MockSplitter.return_value.split_documents.return_value = mock_chunk
            chunker.token_chunking(docs, output_path)

        assert output_path.exists()
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "text" in record
        assert "metadata" in record

    def test_bnb_config_inside_class(self):
        """BitsAndBytesConfig must not be instantiated at module import time."""
        import pipeline.chunker as ch
        import inspect
        src = inspect.getsource(ch)
        # bnb_config must appear after the class definition line
        bnb_pos   = src.find("bnb_config = BitsAndBytesConfig")
        class_pos = src.find("class ChunkingManager")
        assert bnb_pos > class_pos


# ── VectorStoreFactory ────────────────────────────────────────────────────────

class TestVectorStoreFactory:

    def test_raises_without_azure_credentials(self):
        from pipeline.vectorstore import VectorStoreFactory
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("AZURE_SEARCH_ENDPOINT", None)
            os.environ.pop("AZURE_SEARCH_KEY", None)
            with patch("pipeline.vectorstore.HuggingFaceEmbeddings"):
                with pytest.raises(ValueError, match="AZURE_SEARCH"):
                    VectorStoreFactory()

    def test_as_retriever_raises_before_build(self):
        from pipeline.vectorstore import VectorStoreFactory
        with patch.dict("os.environ", {
            "AZURE_SEARCH_ENDPOINT": "https://fake.search.windows.net",
            "AZURE_SEARCH_KEY": "fakekey",
        }):
            with patch("pipeline.vectorstore.HuggingFaceEmbeddings"):
                factory = VectorStoreFactory()

        with pytest.raises(ValueError, match="not initialized"):
            factory.as_retriever()

    def test_skips_upload_if_documents_exist(self):
        from pipeline.vectorstore import VectorStoreFactory
        with patch.dict("os.environ", {
            "AZURE_SEARCH_ENDPOINT": "https://fake.search.windows.net",
            "AZURE_SEARCH_KEY": "fakekey",
        }):
            with patch("pipeline.vectorstore.HuggingFaceEmbeddings"):
                factory = VectorStoreFactory()

        mock_vs = MagicMock()
        mock_vs.client.get_document_count.return_value = 100  # already has docs

        with patch("pipeline.vectorstore.AzureSearch", return_value=mock_vs):
            factory.build_or_load([Document(page_content="x", metadata={})])

        # add_documents should NOT be called when index already populated
        mock_vs.add_documents.assert_not_called()
