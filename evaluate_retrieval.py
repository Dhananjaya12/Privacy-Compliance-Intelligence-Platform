import json
from pathlib import Path
from langchain_core.documents import Document
import pandas as pd
from vectorstore import VectorStoreFactory
import random
from rank_bm25 import BM25Okapi
import re

with open("config_test.json", "r", encoding="utf-8") as f:
    config = json.load(f)

DATASET_SAVE_DIR = Path(config['paths']["save_dir"])
HF_DATASET_DIR = DATASET_SAVE_DIR / "hf_dataset"
FILTERED_DATASET_PATH = DATASET_SAVE_DIR / "filtered_dataset.json"
CHUNKS_DIR =  DATASET_SAVE_DIR / "chunks"

BASE_VECTORSTORE_DIR = DATASET_SAVE_DIR / "vectorstores"
BASE_VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR = BASE_VECTORSTORE_DIR / "chroma"

OUTPUT_EXCEL = DATASET_SAVE_DIR / "retrieval_results.xlsx"
PGVECTOR_CONNECTION = config['vectorstore']['pgvector_connection']

# STRATEGIES = ["semantic", "token", "agentic"]
STRATEGIES = ["semantic", "token"]
VECTORSTORES = ["faiss", "chroma"]
# VECTORSTORES = ["pgvector"]
TOP_K = 5

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def bm25_score(document, reference_answer):
    """
    Compute BM25 score between a retrieved document and the reference answer.
    """
    if not reference_answer or not document:
        return 0.0

    corpus = [tokenize(reference_answer)]
    bm25 = BM25Okapi(corpus)

    scores = bm25.get_scores(tokenize(document))
    return float(scores[0])

def load_test_queries_from_json(
    json_path,
    shuffle=True,
):
    """
    Load evaluation queries from a filtered JSON dataset.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = list(
        {item["question"] for item in dataset if "question" in item}
    )

    if shuffle:
        random.shuffle(questions)

    return questions

def load_qa_map(json_path):
    """
    Returns: dict {question -> answer}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    qa_map = {}
    for item in dataset:
        q = item.get("question")
        a = item.get("answer")
        if q and a:
            qa_map[q] = a

    return qa_map

QA_MAP = load_qa_map(FILTERED_DATASET_PATH)

TEST_QUERIES = load_test_queries_from_json(
    json_path=FILTERED_DATASET_PATH
)

print(f"Loaded {len(TEST_QUERIES)} test queries")

def load_chunks(strategy):
    path = CHUNKS_DIR / f"{strategy}.jsonl"
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

def evaluate():
    rows = []

    for strategy in STRATEGIES:
        docs = load_chunks(strategy)

        for vs_type in VECTORSTORES:
            print(f"Running: {strategy} | {vs_type}")

            collection_name = f"{strategy}_{vs_type}"

            if vs_type == "faiss":
                VECTORSTORE_DIR = BASE_VECTORSTORE_DIR / "faiss" / strategy
            elif vs_type == "chroma":
                VECTORSTORE_DIR = BASE_VECTORSTORE_DIR / "chroma"
            elif vs_type == "pgvector":
                VECTORSTORE_DIR = None

            vs_factory = VectorStoreFactory(
                vs_type=vs_type,
                embedding_model = config["vectorstore"]["embedding_model"],
                persist_dir=VECTORSTORE_DIR,
                collection_name=collection_name,
                pgvector_connection=PGVECTOR_CONNECTION,
            )

            vs_factory.build_or_load(docs)

            retriever = vs_factory.as_retriever(k=TOP_K)

            for query in TEST_QUERIES:
                gt_answer = QA_MAP.get(query, "")

                results = retriever.invoke(query)

                for rank, doc in enumerate(results, start=1):
                    score = bm25_score(
                        document=doc.page_content,
                        reference_answer=gt_answer
                    )

                    rows.append({
                        "strategy": strategy,
                        "vectorstore": vs_type,
                        "query": query,
                        "ground_truth_answer": gt_answer,
                        "rank": rank,
                        "paper_id": doc.metadata.get("paper_id"),
                        "page": doc.metadata.get("page"),
                        "used_ocr": doc.metadata.get("used_ocr"),
                        "chunk_strategy": doc.metadata.get("chunk_strategy"),
                        "bm25_score": score,
                        "content_preview": doc.page_content,
                    })

    df = pd.DataFrame(rows)

    def clean_excel_text(text):
        if isinstance(text, str):
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return text

    df = df.astype(str).applymap(clean_excel_text)

    # Write to Excel with formatting
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="RetrievalResults")

        worksheet = writer.sheets["RetrievalResults"]

        # Adjust column widths for readability
        column_widths = {
            "A": 12,  # strategy
            "B": 14,  # vectorstore
            "C": 60,  # query
            "D": 6,   # rank
            "E": 18,  # paper_id
            "F": 8,   # page
            "G": 10,  # used_ocr
            "H": 14,  # chunk_strategy
            "I": 80,  # content_preview
        }

        for col, width in column_widths.items():
            worksheet.column_dimensions[col].width = width

    print(f"\n✅ Retrieval results saved to: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    evaluate()
