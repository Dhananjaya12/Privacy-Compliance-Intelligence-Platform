import json
import random
from tqdm import tqdm
import arxiv
from pathlib import Path


with open("config_test.json", "r", encoding="utf-8") as f:
    config = json.load(f)

DATASET_SAVE_DIR = Path(config['paths']["save_dir"])
RAW_DATASET_PATH = DATASET_SAVE_DIR / "MDA-QA.json"
FILTERED_DATASET_PATH = DATASET_SAVE_DIR / "filtered_dataset.json"
PDF_DIR = DATASET_SAVE_DIR / "pdfs"

NUM_ROWS = 100
RANDOM_SEED = 42

PDF_DIR.mkdir(parents=True, exist_ok=True)
random.seed(RANDOM_SEED)

with open(RAW_DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} QA entries")

if len(dataset) < NUM_ROWS:
    raise ValueError("Dataset has fewer rows than requested sample size")

sampled_rows = random.sample(dataset, NUM_ROWS)
print(f"Randomly selected {len(sampled_rows)} QA rows")

selected_paper_ids = set()

for item in sampled_rows:
    for pid in item.get("support", []):
        selected_paper_ids.add(pid)

selected_paper_ids = sorted(selected_paper_ids)

print(f"Collected {len(selected_paper_ids)} unique arXiv papers from sampled questions")


with open(FILTERED_DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(sampled_rows, f, indent=2)

print(f"Saved filtered dataset → {FILTERED_DATASET_PATH}")

print("\n📄 Downloading PDFs from arXiv...\n")

for arxiv_id in tqdm(selected_paper_ids):
    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

    if pdf_path.exists():
        continue

    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        paper.download_pdf(filename=pdf_path)
    except Exception as e:
        print(f"⚠️ Failed to download {arxiv_id}: {e}")

print("\n✅ DONE")
print(f"PDFs saved in: {PDF_DIR}")