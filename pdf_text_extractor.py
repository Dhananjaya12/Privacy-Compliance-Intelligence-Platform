import os
import fitz  # PyMuPDF
import pytesseract
from langchain_core.documents import Document
import json
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\dp1622\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

MIN_TEXT_LENGTH = 300  # chars


def ocr_page(page):
    pix = page.get_pixmap(dpi=300)

    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    text = pytesseract.image_to_string(img)

    return text.strip()


def clean_text(text: str) -> str:
    import re

    # Remove (cid:XX) artifacts
    text = re.sub(r"\(cid:\d+\)", "", text)

    # Fix hyphenated line breaks
    text = re.sub(r"-\n", "", text)

    # Merge broken line breaks inside paragraphs
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Fix missing spaces between lowercase-uppercase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_pdf_text(pdf_path, output_path, existing_documents):

    paper_id = os.path.basename(pdf_path)

    # Get existing pages for this paper
    existing_docs = [
        doc for doc in existing_documents
        if doc.metadata["paper_id"] == paper_id
    ]

    existing_pages = {
        doc.metadata["page"]
        for doc in existing_docs
    }

    pdf = fitz.open(pdf_path)
    total_pages = len(pdf)

    new_docs = []

    # Extract missing pages
    with open(output_path, "a", encoding="utf-8") as f:

        for page_idx, page in enumerate(pdf):
            page_number = page_idx + 1

            if page_number in existing_pages:

                continue

            text = page.get_text("text") or ""
            text = text.strip()
            used_ocr = False

            if len(text) < MIN_TEXT_LENGTH:
                text = ocr_page(page)
                used_ocr = True

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

            record = {
                "text": document.page_content,
                "metadata": document.metadata,
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

    # Combine existing + new
    full_pdf_docs = existing_docs + new_docs

    # Sort by page
    full_pdf_docs = sorted(
        full_pdf_docs,
        key=lambda d: d.metadata["page"]
    )

    return full_pdf_docs