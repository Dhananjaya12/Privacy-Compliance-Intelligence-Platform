"""
spark_ingestion.py

Batch PDF ingestion using Apache Spark.

Runs locally with local[*] (all CPU cores). In production on Azure Synapse,
change SparkSession.builder.master("local[*]") to master("yarn") — everything
else is identical.

Flow
----
1. Scan input_dir for PDFs → build Spark DataFrame (one row per file)
2. Skip already-checkpointed files
3. Distribute across workers: each partition extracts + chunks its PDFs
4. Collect structured chunk rows → batch-upload to Azure AI Search
5. Save checkpoint after each successful file → safe to re-run on failure

DataFrame schema
----------------
file_path    : str   — absolute path to source PDF
file_name    : str   — basename (used as paper_id)
regulation   : str   — GDPR | CCPA | HIPAA | NIST | UNKNOWN
chunk_id     : str   — "{file_name}_{chunk_index:04d}"
chunk_index  : int   — position within the file
text         : str   — chunk content
page         : int   — source page number (1-based; 1 for full-doc extracts)
extractor    : str   — "pymupdf4llm" | "azure_ocr"
chunk_strategy: str  — "token" (fixed) | "semantic" (embedding-based)

Usage
-----
# Local (free, uses all cores):
python -m pipeline.spark_ingestion --input_dir data/regulations --strategy token

# Synapse (swap master in get_spark_session before deploying):
python -m pipeline.spark_ingestion --input_dir abfss://container@account.dfs.core.windows.net/pdfs

Production swap (one line):
  .master("local[*]")  →  .master("yarn")
  Add: .config("spark.executor.instances", "10")
       .config("spark.executor.cores", "4")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("pdf_rag_agent.spark_ingestion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Regulation detection (mirrors kg_builder.py logic) ────────────────────────

REGULATION_PATTERNS = {
    "GDPR":  ["gdpr", "general_data_protection", "general data protection"],
    "CCPA":  ["ccpa", "california_consumer", "california consumer"],
    "HIPAA": ["hipaa", "health_insurance", "health insurance"],
    "NIST":  ["nist", "cybersecurity_framework", "cybersecurity framework"],
}


def detect_regulation(file_name: str) -> str:
    name_lower = file_name.lower()
    for reg, patterns in REGULATION_PATTERNS.items():
        if any(p in name_lower for p in patterns):
            return reg
    return "UNKNOWN"


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

CHECKPOINT_FILE = Path("data/spark_ingestion_checkpoint.json")


def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        return set(data.get("completed", []))
    return set()


def save_checkpoint(completed: set) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps({"completed": sorted(completed)}, indent=2))


# ── Spark session ──────────────────────────────────────────────────────────────

def get_spark_session(app_name: str = "PDFRagIngestion"):
    """
    Build a SparkSession.

    LOCAL  → .master("local[*]")   free, all CPU cores, no cluster needed
    SYNAPSE → change to .master("yarn") and add executor configs
    """
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(app_name)
        # ── swap this one line for production ──────────────────────────────
        .master("local[*]")
        # PRODUCTION (Azure Synapse / Databricks):
        # .master("yarn")
        # .config("spark.executor.instances", "10")
        # .config("spark.executor.cores", "4")
        # .config("spark.executor.memory", "14g")
        # ──────────────────────────────────────────────────────────────────
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        # Silence Spark's verbose INFO logs
        .config("spark.log.level", "WARN")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ── Per-partition processing (runs inside Spark workers) ──────────────────────

def _process_partition(rows: Iterator, strategy: str, azure_ocr_endpoint: str, azure_ocr_key: str) -> Iterator[dict]:
    """
    Called once per Spark partition. Each row is a dict with file_path and regulation.

    We import heavy deps here (not at module level) so Spark workers
    don't fail if the driver imports this module without pyspark installed.
    """
    import fitz
    import pymupdf4llm
    from langchain_text_splitters import TokenTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_experimental.text_splitter import SemanticChunker

    token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=100)

    # Semantic splitter is expensive to init — lazy, only if needed
    _semantic_splitter = None

    def get_semantic_splitter():
        nonlocal _semantic_splitter
        if _semantic_splitter is None:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            _semantic_splitter = SemanticChunker(embeddings=embeddings)
        return _semantic_splitter

    def is_scanned(path: str, sample_pages: int = 3) -> bool:
        doc = fitz.open(path)
        text = "".join(doc[i].get_text("text") for i in range(min(sample_pages, len(doc))))
        doc.close()
        return len(text.strip()) < 200

    def extract_text(path: str) -> tuple[str, str]:
        """Returns (text, extractor_name)."""
        if is_scanned(path):
            # Azure OCR fallback for scanned PDFs
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
            from azure.core.credentials import AzureKeyCredential

            client = DocumentIntelligenceClient(
                endpoint=azure_ocr_endpoint,
                credential=AzureKeyCredential(azure_ocr_key),
            )
            doc = fitz.open(path)
            all_text = []
            for start in range(0, len(doc), 20):
                end = min(start + 20, len(doc))
                chunk = fitz.open()
                chunk.insert_pdf(doc, from_page=start, to_page=end - 1)
                chunk_bytes = chunk.tobytes()
                chunk.close()
                try:
                    poller = client.begin_analyze_document(
                        model_id="prebuilt-layout",
                        body=AnalyzeDocumentRequest(bytes_source=chunk_bytes),
                        output_content_format=DocumentContentFormat.MARKDOWN,
                    )
                    result = poller.result(timeout=600)
                    if result.content:
                        all_text.append(result.content.strip())
                except Exception as e:
                    logger.warning("OCR failed for pages %d-%d in %s: %s", start, end, path, e)
            doc.close()
            return "\n\n".join(all_text), "azure_ocr"
        else:
            text = pymupdf4llm.to_markdown(
                path,
                show_progress=False,
                page_chunks=False,
                margins=(0, 0, 0, 0),
            )
            return text.strip(), "pymupdf4llm"

    def chunk_text(text: str, strategy: str) -> List[str]:
        from langchain_core.documents import Document
        doc = Document(page_content=text)
        if strategy == "semantic":
            splitter = get_semantic_splitter()
            return [c.page_content for c in splitter.split_documents([doc])]
        else:
            return [c.page_content for c in token_splitter.split_documents([doc])]

    for row in rows:
        file_path  = row["file_path"]
        file_name  = row["file_name"]
        regulation = row["regulation"]

        try:
            text, extractor = extract_text(file_path)
            chunks = chunk_text(text, strategy)

            for idx, chunk_text_str in enumerate(chunks):
                yield {
                    "file_path":      file_path,
                    "file_name":      file_name,
                    "regulation":     regulation,
                    "chunk_id":       f"{Path(file_name).stem}_{idx:04d}",
                    "chunk_index":    idx,
                    "text":           chunk_text_str,
                    "page":           1,       # token/semantic splits don't track page
                    "extractor":      extractor,
                    "chunk_strategy": strategy,
                    "status":         "ok",
                    "error":          "",
                }

        except Exception as exc:
            logger.error("Failed to process %s: %s", file_path, exc)
            yield {
                "file_path":      file_path,
                "file_name":      file_name,
                "regulation":     regulation,
                "chunk_id":       f"{Path(file_name).stem}_ERROR",
                "chunk_index":    -1,
                "text":           "",
                "page":           -1,
                "extractor":      "error",
                "chunk_strategy": strategy,
                "status":         "error",
                "error":          str(exc),
            }


# ── Azure AI Search uploader ───────────────────────────────────────────────────

def upload_to_azure_search(
    chunk_rows: List[dict],
    index_name: str,
    doc_type: str,
    batch_size: int = 100,
) -> int:
    """
    Converts chunk rows to LangChain Documents and uploads to Azure AI Search.

    Parameters
    ----------
    index_name : target index ("compliance-regulations" or "compliance-policies")
    doc_type   : "regulation" | "policy" — stored as a filterable field so the
                 retriever can scope by document class.

    Returns number of successfully uploaded chunks.
    """
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores.azuresearch import AzureSearch

    from pipeline.search_schema import build_compliance_fields

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key      = os.getenv("AZURE_SEARCH_KEY")

    if not endpoint or not key:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY must be set.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = AzureSearch(
        azure_search_endpoint=endpoint,
        azure_search_key=key,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        fields=build_compliance_fields(),
    )

    ok_rows = [r for r in chunk_rows if r["status"] == "ok" and r["text"].strip()]
    documents = [
        Document(
            page_content=row["text"],
            metadata={
                "paper_id":       row["file_name"],
                "chunk_id":       row["chunk_id"],
                "regulation":     row["regulation"],
                "doc_type":       doc_type,
                "chunk_index":    row["chunk_index"],
                "page":           row["page"],
                "extractor":      row["extractor"],
                "chunk_strategy": row["chunk_strategy"],
            },
        )
        for row in ok_rows
    ]

    uploaded = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        try:
            vectorstore.add_documents(batch)
            uploaded += len(batch)
            logger.info("Uploaded %d / %d chunks to Azure Search", uploaded, len(documents))
        except Exception as exc:
            logger.error("Batch upload failed at offset %d: %s", i, exc)

    return uploaded


# ── Main ingestion pipeline ────────────────────────────────────────────────────

def _resolve_index_and_doc_type(
    input_dir: str,
    index_name: Optional[str],
    doc_type: Optional[str],
) -> tuple:
    """
    Decide the target index and doc_type. Explicit args win; otherwise infer
    from the input directory ("policy" dirs → policies index, else regulations).
    """
    inferred_policy = "policy_docs" in input_dir.replace("\\", "/").lower() \
        or "policies" in input_dir.replace("\\", "/").lower()

    resolved_doc_type = doc_type or ("policy" if inferred_policy else "regulation")

    if index_name:
        resolved_index = index_name
    elif resolved_doc_type == "policy":
        resolved_index = os.getenv("AZURE_SEARCH_POLICIES_INDEX", "compliance-policies")
    else:
        resolved_index = os.getenv("AZURE_SEARCH_REGULATIONS_INDEX", "compliance-regulations")

    return resolved_index, resolved_doc_type


def run_ingestion(
    input_dir: str,
    strategy: str = "token",
    partitions: int = 4,
    skip_upload: bool = False,
    index_name: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> dict:
    """
    Full ingestion run.

    Parameters
    ----------
    input_dir    : Directory containing PDFs (local path or ADLS abfss:// URI)
    strategy     : "token" | "semantic"
    partitions   : Number of Spark partitions (= parallel workers locally)
    skip_upload  : If True, extract+chunk but don't push to Azure Search (dry run)
    index_name   : Target Azure Search index. Inferred from input_dir if omitted.
    doc_type     : "regulation" | "policy". Inferred from input_dir if omitted.

    Returns
    -------
    Summary dict: {total_files, processed, skipped, total_chunks, uploaded, errors}
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType
    )

    target_index, target_doc_type = _resolve_index_and_doc_type(
        input_dir, index_name, doc_type
    )
    logger.info("Ingestion target: index=%s doc_type=%s", target_index, target_doc_type)

    azure_ocr_endpoint = os.getenv("AZURE_DOC_INTEL_ENDPOINT", "")
    azure_ocr_key      = os.getenv("AZURE_DOC_INTEL_KEY", "")

    # ── Discover PDFs ─────────────────────────────────────────────────────────
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    all_pdfs = sorted(input_path.glob("**/*.pdf"))
    if not all_pdfs:
        logger.warning("No PDF files found in %s", input_dir)
        return {"total_files": 0, "processed": 0, "skipped": 0,
                "total_chunks": 0, "uploaded": 0, "errors": 0}

    # ── Checkpoint: skip already-processed files ──────────────────────────────
    completed = load_checkpoint()
    pending = [p for p in all_pdfs if str(p) not in completed]

    logger.info(
        "Found %d PDFs | %d already processed | %d pending",
        len(all_pdfs), len(completed), len(pending),
    )

    if not pending:
        logger.info("All files already processed. Nothing to do.")
        return {"total_files": len(all_pdfs), "processed": 0, "skipped": len(all_pdfs),
                "total_chunks": 0, "uploaded": 0, "errors": 0}

    # ── Build Spark DataFrame ─────────────────────────────────────────────────
    spark = get_spark_session()

    rows = [
        {
            "file_path":  str(p.resolve()),
            "file_name":  p.name,
            "regulation": detect_regulation(p.name),
        }
        for p in pending
    ]

    input_schema = StructType([
        StructField("file_path",  StringType(), False),
        StructField("file_name",  StringType(), False),
        StructField("regulation", StringType(), False),
    ])

    df_input = spark.createDataFrame(rows, schema=input_schema).repartition(partitions)

    logger.info("Spark DataFrame: %d rows across %d partitions", len(rows), partitions)

    # ── Output schema ─────────────────────────────────────────────────────────
    output_schema = StructType([
        StructField("file_path",       StringType(),  False),
        StructField("file_name",       StringType(),  False),
        StructField("regulation",      StringType(),  False),
        StructField("chunk_id",        StringType(),  False),
        StructField("chunk_index",     IntegerType(), False),
        StructField("text",            StringType(),  False),
        StructField("page",            IntegerType(), False),
        StructField("extractor",       StringType(),  False),
        StructField("chunk_strategy",  StringType(),  False),
        StructField("status",          StringType(),  False),
        StructField("error",           StringType(),  False),
    ])

    # ── Broadcast shared config to workers ────────────────────────────────────
    # Broadcasting avoids serializing large objects per-row
    bc_strategy         = spark.sparkContext.broadcast(strategy)
    bc_ocr_endpoint     = spark.sparkContext.broadcast(azure_ocr_endpoint)
    bc_ocr_key          = spark.sparkContext.broadcast(azure_ocr_key)

    def process_partition_wrapper(rows: Iterator) -> Iterator[dict]:
        return _process_partition(
            rows,
            strategy=bc_strategy.value,
            azure_ocr_endpoint=bc_ocr_endpoint.value,
            azure_ocr_key=bc_ocr_key.value,
        )

    # ── Run distributed extraction + chunking ─────────────────────────────────
    logger.info("Starting Spark distributed processing (strategy=%s)...", strategy)

    df_chunks = df_input.rdd.mapPartitions(process_partition_wrapper).toDF(output_schema)

    # Collect to driver — acceptable for PDFs (chunks are text, not binary)
    # For very large corpora (10k+ PDFs), write to Delta/Parquet instead and
    # upload from there in a second pass.
    chunk_rows = df_chunks.collect()
    chunk_dicts = [row.asDict() for row in chunk_rows]

    # ── Stats ─────────────────────────────────────────────────────────────────
    ok_rows    = [r for r in chunk_dicts if r["status"] == "ok"]
    error_rows = [r for r in chunk_dicts if r["status"] == "error"]
    processed_files = {r["file_path"] for r in chunk_dicts}
    error_files     = {r["file_path"] for r in error_rows}

    logger.info(
        "Extraction complete | files=%d | chunks=%d | errors=%d",
        len(processed_files), len(ok_rows), len(error_files),
    )

    if error_rows:
        for r in error_rows:
            logger.error("  ✗ %s — %s", r["file_name"], r["error"])

    # ── Upload to Azure AI Search ─────────────────────────────────────────────
    uploaded = 0
    if not skip_upload and ok_rows:
        logger.info(
            "Uploading %d chunks to Azure AI Search (index=%s)...",
            len(ok_rows), target_index,
        )
        uploaded = upload_to_azure_search(
            ok_rows, index_name=target_index, doc_type=target_doc_type
        )
    else:
        if skip_upload:
            logger.info("skip_upload=True — skipping Azure Search upload (dry run).")

    # ── Update checkpoint (only files with no errors) ─────────────────────────
    newly_completed = processed_files - error_files
    completed.update(newly_completed)
    save_checkpoint(completed)
    logger.info("Checkpoint saved. %d files marked complete.", len(newly_completed))

    spark.stop()

    summary = {
        "total_files":   len(all_pdfs),
        "processed":     len(processed_files),
        "skipped":       len(all_pdfs) - len(pending),
        "total_chunks":  len(ok_rows),
        "uploaded":      uploaded,
        "errors":        len(error_files),
    }
    logger.info("Ingestion summary: %s", summary)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch PDF ingestion with Spark")
    parser.add_argument(
        "--input_dir", default="data/regulations",
        help="Directory containing PDFs (local or ADLS abfss:// URI)",
    )
    parser.add_argument(
        "--strategy", choices=["token", "semantic"], default="token",
        help="Chunking strategy: token (fast) or semantic (embedding-based)",
    )
    parser.add_argument(
        "--partitions", type=int, default=4,
        help="Number of Spark partitions (= parallel workers)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Extract and chunk but skip Azure Search upload",
    )
    parser.add_argument(
        "--index", default=None,
        help="Target Azure Search index (inferred from input_dir if omitted)",
    )
    parser.add_argument(
        "--doc_type", choices=["regulation", "policy"], default=None,
        help="Document class (inferred from input_dir if omitted)",
    )
    args = parser.parse_args()

    summary = run_ingestion(
        input_dir=args.input_dir,
        strategy=args.strategy,
        partitions=args.partitions,
        skip_upload=args.dry_run,
        index_name=args.index,
        doc_type=args.doc_type,
    )
    print("\nIngestion complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")