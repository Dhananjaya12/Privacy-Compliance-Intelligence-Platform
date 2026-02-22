import json
from pathlib import Path
from langchain_core.documents import Document

from pdf_text_extractor import extract_pdf_text
from chunking import ChunkingManager
from tqdm import tqdm

from vectorstore import VectorStoreFactory
from rag_generate import LLMGenerator

with open("config_test.json", "r", encoding="utf-8") as f:
    config = json.load(f)

DATASET_SAVE_DIR = Path(config['paths']["save_dir"])
PDF_DIR = DATASET_SAVE_DIR / "pdfs"

EXTRACTED_TEXT_DIR = DATASET_SAVE_DIR / "extracted_text"
EXTRACTED_TEXT_PATH = EXTRACTED_TEXT_DIR / "pages.jsonl"

CHUNKS_DIR = DATASET_SAVE_DIR / "chunks"
BASE_VECTORSTORE_DIR = Path(config['paths']["save_dir"]) / "vectorstores"

CHUNK_STRATEGY = config['chunking']['strategy']   # "token" | "semantic" | "agentic"

TOKEN_CHUNK_SIZE = config['chunking']['token_chunk_size']
TOKEN_CHUNK_OVERLAP = config['chunking']['token_chunk_overlap']

VECTORSTORE_TYPE = config['vectorstore']['type']   # "faiss" | "chroma" | "pgvector"
TOP_K = config['vectorstore']['top_k']
COLLECTION_NAME = f"{CHUNK_STRATEGY}_chunks"

PGVECTOR_CONNECTION = config['vectorstore']['pgvector_connection']

EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
BASE_VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

def save_jsonl(docs, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            f.write(json.dumps({
                "id": i,
                "text": doc.page_content,
                "metadata": doc.metadata
            }) + "\n")

def load_extracted_pages(path):
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            documents.append(
                Document(
                    page_content=record["text"],
                    metadata=record["metadata"]
                )
            )
    return documents

def save_extracted_pages(documents, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            record = {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(documents)} extracted pages → {output_path}")

def load_chunks(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(
                Document(
                    page_content=row["text"],
                    metadata=row["metadata"]
                )
            )
    return docs

def main():
    print("🔹 Starting pipeline")

    documents = []

    if EXTRACTED_TEXT_PATH.exists():
        print("✅ Found extracted text. Loading...")

        with open(EXTRACTED_TEXT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)

                doc = Document(
                    page_content=record["text"],
                    metadata=record["metadata"],
                )

                documents.append(doc)

    print("🔹 Processing PDFs...")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    all_documents = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        print(f"   → {pdf_path.name}")

        pdf_docs = extract_pdf_text(
            str(pdf_path),
            EXTRACTED_TEXT_PATH,
            documents
        )

        all_documents.extend(pdf_docs)

    # Final sort
    all_documents = sorted(
        all_documents,
        key=lambda d: (d.metadata["paper_id"], d.metadata["page"]),
    )

    documents = all_documents

    print(f"✅ Total pages available: {len(documents)}")

    save_extracted_pages(documents, EXTRACTED_TEXT_PATH)

    print(f"🔹 Loaded {len(documents)} pages")

    chunk_file = CHUNKS_DIR / f"{CHUNK_STRATEGY}.jsonl"

    completed = set()

    if chunk_file.exists():
        print(f"✅ Found existing chunks for '{CHUNK_STRATEGY}'. Resuming...")

        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    m = r.get("metadata", {})
                    completed.add((m.get("paper_id"), m.get("page")))
                except json.JSONDecodeError:
                    continue

    # Filter only remaining documents
    remaining_documents = [
        d for d in documents
        if (d.metadata.get("paper_id"), d.metadata.get("page")) not in completed
    ]

    print(f"Remaining docs to process: {len(remaining_documents)}")

    chunker = ChunkingManager(embedding_model=config["chunking"]["embedding_model"], 
                                  llm_model=config["chunking"]["llm_model"])

    if CHUNK_STRATEGY == "semantic":

        chunks = chunker.semantic_chunking(
            remaining_documents,
            output_path=chunk_file,
            append=True
        )

    elif CHUNK_STRATEGY == "token":

        chunks = chunker.token_chunking(
            remaining_documents,
            output_path=chunk_file,
            chunk_size=TOKEN_CHUNK_SIZE,
            chunk_overlap=TOKEN_CHUNK_OVERLAP,
            append=True
        )

    elif CHUNK_STRATEGY == "agentic":
        prompt_template = config["llm"]["chunking_prompt"].strip()
        
        chunks = chunker.agentic_chunking(
                documents,
                output_path=chunk_file,
                prompt_template=prompt_template,
                append=True
            )

    else:
        raise ValueError(f"Unknown chunking strategy: {CHUNK_STRATEGY}")

    print(f"✅ Finished chunking with strategy '{CHUNK_STRATEGY}'")

    print(f"🔹 Initializing vector store: {VECTORSTORE_TYPE.upper()}")

    if VECTORSTORE_TYPE == "faiss":
        VECTORSTORE_DIR = BASE_VECTORSTORE_DIR / "faiss" / CHUNK_STRATEGY
    elif VECTORSTORE_TYPE == "chroma":
        VECTORSTORE_DIR = BASE_VECTORSTORE_DIR / "chroma"
    elif VECTORSTORE_TYPE == "pgvector":
        VECTORSTORE_DIR = None
    else:
        raise ValueError(f"Unknown vector store type: {VECTORSTORE_TYPE}")

    vs_factory = VectorStoreFactory(
        vs_type=VECTORSTORE_TYPE,
        embedding_model = config["vectorstore"]["embedding_model"],
        persist_dir=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME,
        pgvector_connection=PGVECTOR_CONNECTION,
    )

    vs_factory.build_or_load(chunks)
    retriever = vs_factory.as_retriever(k=TOP_K)

    print("🎉 Pipeline ready — retrieval can now run")

    generator = LLMGenerator(llm_model_name=config["llm"]["llm_model_name"])

    question="What problem does the transformer architecture solve?"
    docs = retriever.invoke(question)
    
    context = "\n\n---\n\n".join(
        d.page_content for d in docs
    )

    prompt =  config["llm"]["retrieval_prompt"].strip().format(context=context, question=question)

    result = generator.generate_answer(
        docs,
        prompt,
        question,
    )

    print(result["generated_answer"])

if __name__ == "__main__":
    main()
