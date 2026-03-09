"""tests/unit/test_chunker.py"""

import json
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from pipeline.chunker import ChunkingManager


def _make_docs(n=3):
    return [
        Document(
            page_content=f"This is page {i}. " * 50,
            metadata={"paper_id": "test.pdf", "page": i},
        )
        for i in range(1, n + 1)
    ]


def test_token_chunking_creates_chunks():
    docs = _make_docs(2)
    chunker = ChunkingManager()

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        out = Path(tf.name)

    chunks = chunker.token_chunking(docs, output_path=out, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["chunk_strategy"] == "token"
        assert "paper_id" in chunk.metadata

    # Verify JSONL was written
    lines = out.read_text().strip().split("\n")
    assert len(lines) == len(chunks)
    for line in lines:
        record = json.loads(line)
        assert "text" in record
        assert "metadata" in record

    out.unlink()


def test_token_chunking_append_mode():
    docs = _make_docs(1)
    chunker = ChunkingManager()

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        out = Path(tf.name)

    chunker.token_chunking(docs, output_path=out, chunk_size=50, append=False)
    first_count = len(out.read_text().strip().split("\n"))

    docs2 = _make_docs(1)
    docs2[0].metadata["page"] = 99
    chunker.token_chunking(docs2, output_path=out, chunk_size=50, append=True)
    second_count = len(out.read_text().strip().split("\n"))

    assert second_count > first_count
    out.unlink()
