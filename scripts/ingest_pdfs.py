import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.rag_pipeline import RAGPipeline
from app.core.config import get_settings


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG vector store")
    parser.add_argument("--pdf-dir", required=True, type=Path, help="Directory containing PDF files")
    parser.add_argument("--config", type=Path, default=Path("config/default.json"), help="Config JSON path")
    parser.add_argument("--strategy", choices=["token", "semantic", "agentic"], default="token")
    parser.add_argument("--vs-type", choices=["faiss", "chroma", "pgvector"], default="faiss")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    if not args.pdf_dir.exists():
        logging.error("PDF directory does not exist: %s", args.pdf_dir)
        sys.exit(1)

    pdfs = list(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        logging.error("No PDF files found in %s", args.pdf_dir)
        sys.exit(1)

    logging.info("Found %d PDF(s) in %s", len(pdfs), args.pdf_dir)

    settings = get_settings()
    config = settings.as_pipeline_config()

    # Override from CLI args
    config["chunking"]["strategy"] = args.strategy
    config["vectorstore"]["type"] = args.vs_type
    config["paths"]["save_dir"] = str(args.pdf_dir.parent / "data")

    # Ensure PDFs are in expected location
    import shutil
    pdf_dest = Path(config["paths"]["save_dir"]) / "pdfs"
    pdf_dest.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        shutil.copy2(pdf, pdf_dest / pdf.name)

    pipeline = RAGPipeline(config)
    retriever, generator = pipeline.build()

    logging.info("✅ Ingestion complete. Vector store is ready.")
    logging.info("You can now start the API server: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
