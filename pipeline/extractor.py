"""
extractor.py

PDF text extraction pipeline.
Primary:  pymupdf4llm  — converts text-based PDFs to clean Markdown locally.
                         Handles multi-column layout, bold, headings, tables.
                         No page limits, no API costs, instant.
Fallback: Azure Document Intelligence — OCR for scanned/image-based PDFs only.
          Chunked into 20-page batches to stay under the F0 tier 4 MB limit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pymupdf4llm
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentContentFormat,
)
from azure.core.credentials import AzureKeyCredential
from langchain_core.documents import Document


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_scanned(pdf_path: str, sample_pages: int = 3) -> bool:
    """
    Returns True if the PDF has no extractable text layer (scanned/image PDF).
    Checks the first N pages — if total text < 200 chars it's likely scanned.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for i in range(min(sample_pages, len(doc))):
        text += doc[i].get_text("text")
    doc.close()
    return len(text.strip()) < 200


# ── Main extractor ────────────────────────────────────────────────────────────

class PDFTextExtractor:

    def __init__(self) -> None:
        endpoint = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
        key      = os.getenv("AZURE_DOC_INTEL_KEY")

        if not endpoint or not key:
            raise ValueError(
                "AZURE_DOC_INTEL_ENDPOINT and AZURE_DOC_INTEL_KEY must be set in .env"
            )

        self.client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

    # ── Primary: pymupdf4llm ──────────────────────────────────────────────────

    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """
        Convert text-based PDF to clean Markdown using pymupdf4llm.
        Correctly handles multi-column layouts, bold headers, tables, and lists.
        No page limit, no API cost.
        """
        doc   = fitz.open(pdf_path)
        total = len(doc)
        doc.close()

        print(f"  → pymupdf4llm: converting {total} pages to Markdown...")

        md_text = pymupdf4llm.to_markdown(
            pdf_path,
            show_progress=True,
            page_chunks=False,      # return as single string
            margins=(0, 0, 0, 0),   # use full page area
        )

        print(f"  → Extracted {len(md_text):,} characters")
        return md_text.strip()

    # ── Fallback: Azure OCR ───────────────────────────────────────────────────

    def _extract_with_azure_ocr(self, pdf_path: str) -> str:
        """
        OCR fallback for scanned/image PDFs using Azure Document Intelligence.
        Splits into 20-page chunks to stay under the F0 4 MB per-request limit.
        Returns Markdown output.
        """
        doc        = fitz.open(pdf_path)
        total      = len(doc)
        chunk_size = 20
        all_text   = []

        print(f"  → Azure OCR: processing {total} pages in chunks of {chunk_size}...")

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            print(f"  → OCR pages {start + 1}–{end}/{total}...")

            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
            chunk_bytes = chunk_doc.tobytes()
            chunk_doc.close()

            try:
                poller = self.client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    body=AnalyzeDocumentRequest(bytes_source=chunk_bytes),
                    output_content_format=DocumentContentFormat.MARKDOWN,
                )
                result = poller.result(timeout=600)
                if result.content:
                    all_text.append(result.content.strip())
            except Exception as exc:
                print(f"  ⚠️ OCR chunk {start}–{end} failed: {exc}")
                continue

        doc.close()
        full_text = "\n\n".join(all_text)
        print(f"  → OCR extracted {len(full_text):,} characters")
        return full_text.strip()

    # ── Routing ───────────────────────────────────────────────────────────────

    def _extract(self, pdf_path: str) -> str:
        """
        Route to the correct extractor based on PDF type.
          Text-based → pymupdf4llm (fast, free, best quality)
          Scanned     → Azure Doc Intelligence OCR
        """
        if _is_scanned(pdf_path):
            print("  → Scanned PDF detected — using Azure OCR")
            return self._extract_with_azure_ocr(pdf_path)
        return self._extract_with_pymupdf(pdf_path)

    # ── Public interface ──────────────────────────────────────────────────────

    def extract(
        self,
        pdf_path:           str,
        output_path:        Path,
        existing_documents: List[Document],
    ) -> List[Document]:
        """
        Extract text from a PDF and cache the result to JSONL.
        Returns the cached result immediately if already extracted.
        """
        paper_id = os.path.basename(pdf_path)

        existing_docs: List[Document] = [
            doc for doc in existing_documents
            if doc.metadata["paper_id"] == paper_id
        ]
        if existing_docs:
            print(f"✅ '{paper_id}' already extracted — skipping.")
            return existing_docs

        print(f"🔹 Extracting '{paper_id}'...")
        full_text = self._extract(pdf_path)

        extractor_used = "azure_ocr" if _is_scanned(pdf_path) else "pymupdf4llm"
        document = Document(
            page_content=full_text,
            metadata={
                "paper_id":  paper_id,
                "page":      1,
                "extractor": extractor_used,
            },
        )

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"text": document.page_content, "metadata": document.metadata}
                )
                + "\n"
            )

        print(f"✅ Extracted '{paper_id}' — {len(full_text):,} chars")
        return [document]
