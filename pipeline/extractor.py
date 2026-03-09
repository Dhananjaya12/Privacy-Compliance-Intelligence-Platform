import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Set

import fitz  # PyMuPDF
from langchain_core.documents import Document
from PIL import Image

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False


MIN_TEXT_LENGTH: int = 300  # Minimum chars before OCR fallback is triggered


class PDFTextExtractor:

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        min_text_length: int = MIN_TEXT_LENGTH,
        ocr_dpi: int = 300,
    ) -> None:
        self.min_text_length = min_text_length
        self.ocr_dpi = ocr_dpi
        self._configure_tesseract(tesseract_cmd)

    def _configure_tesseract(self, tesseract_cmd: Optional[str]) -> None:
        """Resolve the Tesseract executable path."""
        if not _TESSERACT_AVAILABLE:
            return

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif sys.platform == "win32":
            # Common Windows install location – override via env var or
            # pass ``tesseract_cmd`` explicitly to avoid hard-coded paths.
            default_win = os.environ.get(
                "TESSERACT_CMD",
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            )
            pytesseract.pytesseract.tesseract_cmd = default_win
        # On Linux/macOS the system PATH is used automatically.

    def _ocr_page(self, page: fitz.Page) -> str:
        """Rasterise *page* and run Tesseract OCR on it."""
        if not _TESSERACT_AVAILABLE:
            raise RuntimeError(
                "pytesseract is not installed; OCR fallback unavailable."
            )

        pix = page.get_pixmap(dpi=self.ocr_dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img).strip()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise extracted text: remove artefacts, fix line breaks."""
        # Remove (cid:XX) font-encoding artefacts
        text = re.sub(r"\(cid:\d+\)", "", text)
        # Rejoin hyphenated line-breaks
        text = re.sub(r"-\n", "", text)
        # Merge soft line-breaks inside paragraphs
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        # Insert missing space between camelCase boundaries
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        # Collapse repeated whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract(
        self,
        pdf_path: str,
        output_path: Path,
        existing_documents: List[Document],
    ) -> List[Document]:
    
        paper_id = os.path.basename(pdf_path)

        existing_docs: List[Document] = [
            doc for doc in existing_documents
            if doc.metadata["paper_id"] == paper_id
        ]
        existing_pages: Set[int] = {
            doc.metadata["page"] for doc in existing_docs
        }

        pdf = fitz.open(pdf_path)
        new_docs: List[Document] = []

        with open(output_path, "a", encoding="utf-8") as f:
            for page_idx, page in enumerate(pdf):
                page_number = page_idx + 1

                if page_number in existing_pages:
                    continue  # Already processed

                text = (page.get_text("text") or "").strip()
                used_ocr = False

                if len(text) < self.min_text_length:
                    text = self._ocr_page(page)
                    used_ocr = True

                text = self._clean_text(text)

                if not text:
                    continue

                document = Document(
                    page_content=text,
                    metadata={
                        "paper_id": paper_id,
                        "page": page_number,
                        "used_ocr": used_ocr,
                    },
                )
                new_docs.append(document)
                f.write(
                    json.dumps(
                        {"text": document.page_content, "metadata": document.metadata}
                    )
                    + "\n"
                )
                f.flush()

        all_docs = sorted(
            existing_docs + new_docs,
            key=lambda d: d.metadata["page"],
        )
        return all_docs