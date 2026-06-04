from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "processed" / "personalization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSACTIONS = REPO_ROOT / "data" / "raw" / "transactions_train.csv"
META_FILE = REPO_ROOT / "data" / "processed" / "dataset_qwen_completed.csv"

ARTICLE_EMB_NPY = OUT_DIR / "article_image_emb.npy"
ARTICLE_INDEX = OUT_DIR / "article_index.json"
ARTICLE_POP = OUT_DIR / "article_popularity.csv"
PROFILES = OUT_DIR / "profiles.csv"
TASTE_NPY = OUT_DIR / "taste_vectors.npy"
TASTE_INDEX = OUT_DIR / "taste_index.json"

MIN_PURCHASES = 10
SAMPLE_CUSTOMERS = 8000
SEED = 42

PROFILE_FIELDS = [
    "product_type_name",
    "colour_group_name",
    "style_aesthetic",
    "occasion",
    "index_group_name",
]
TOP_K_PER_FIELD = 5
