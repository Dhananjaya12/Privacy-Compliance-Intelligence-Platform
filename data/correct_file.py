import json
import pandas as pd
from pathlib import Path

with open("config_test.json", "r", encoding="utf-8") as f:
    config = json.load(f)

DATASET_SAVE_DIR = Path(config['paths']["save_dir"])
RETRIEVAL_EXCEL = DATASET_SAVE_DIR / "retrieval_results.xlsx"
FILTERED_DATASET_PATH = DATASET_SAVE_DIR / "filtered_dataset.json"
OUTPUT_EXCEL = DATASET_SAVE_DIR / "retrieval_results_with_support.xlsx"

def load_support_map(json_path):
    """
    Returns: dict {question -> support_list}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    support_map = {}
    for item in dataset:
        q = item.get("question")
        s = item.get("support", [])
        if q:
            support_map[q] = s

    return support_map

def main():
    print("📄 Loading retrieval results...")
    df = pd.read_excel(RETRIEVAL_EXCEL)

    # normalize paper ids (remove .pdf if present)
    df["paper_id"] = df["paper_id"].str.replace(r"\.pdf$", "", regex=True)

    print("📄 Loading support dataset...")
    support_map = load_support_map(FILTERED_DATASET_PATH)

    print("➕ Adding support column...")
    df["support"] = df["query"].apply(
        lambda q: support_map.get(q, [])
    )

    df["hit"] = df.apply(
    lambda r: r["paper_id"] in r["support"],
    axis=1
)

    grouped = (
        df
        .groupby(["query", "strategy", "vectorstore"])
        .agg(
            retrieved_papers=("paper_id", lambda x: list(set(x))),
            support=("support", "first")
        )
        .reset_index()
    )

    def support_coverage(retrieved, support):
        if not support:
            return 0.0
        return len(set(retrieved) & set(support)) / len(set(support))

    grouped["support_coverage"] = grouped.apply(
        lambda r: support_coverage(r["retrieved_papers"], r["support"]),
        axis=1
    )

    grouped["support_coverage_pct"] = grouped["support_coverage"] * 100

    recall_per_query = (
        df
        .groupby(["strategy", "vectorstore", "query"])["hit"]
        .max()
        .reset_index(name="recall_at_k")
    )

    mean_recall = (
        recall_per_query
        .groupby(["strategy", "vectorstore"])["recall_at_k"]
        .mean()
        .reset_index(name="mean_recall_at_k")
    )

    mean_support_coverage = (
        grouped
        .groupby(["strategy", "vectorstore"])["support_coverage"]
        .mean()
        .reset_index(name="mean_support_coverage")
    )

    print("💾 Writing results to Excel...")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="RetrievalResults")
        grouped.to_excel(writer, index=False, sheet_name="SupportCoveragePerQuery")
        recall_per_query.to_excel(writer, index=False, sheet_name="RecallPerQuery")
        mean_recall.to_excel(writer, index=False, sheet_name="MeanRecall")
        mean_support_coverage.to_excel(writer, index=False, sheet_name="MeanSupportCoverage")

    print(f"✅ Done! Saved to {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()
