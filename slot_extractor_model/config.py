"""Config for the slot-extractor fine-tune (colour/fit/occasion/season).

Mirrors intent_analysis_model: ckey.vn multi-model gen + sonnet judge + DeBERTa, but the
task is multi-head slot filling. KEY design for trustworthy labels: data is generated
LABEL-CONDITIONED — we sample a slot spec and ask the LLM to WRITE a query expressing
exactly it (with varied/rare surface forms), so the label comes from the generator, not
from an LLM reading the query. A blind judge then validates faithfulness.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_VN = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)


def is_english(text: str) -> bool:
    return not bool(_VN.search(str(text)))


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
REPO_ROOT = ROOT.parent


def _load_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


_ENV = _load_env(REPO_ROOT / ".env")
API_BASE = os.getenv("CKEY_API_BASE", _ENV.get("CKEY_API_BASE", "https://ckey.vn/v1"))
API_KEY = os.getenv("CKEY_API_KEY", _ENV.get("CKEY_API_KEY", ""))

GEN_MODELS_TRAIN = ["gpt-5.4-mini", "deepseek-3.2", "glm-5", "claude-haiku-4.5"]
GEN_MODELS_TEST = ["gpt-5.4"]
JUDGE_MODEL = "claude-sonnet-4.5"

# Slot ontology = catalog canonical values the model must predict. "none" = not mentioned.
FIELDS = ["colour_group", "fit", "occasion", "seasonality"]
SLOT_VALUES = {
    "colour_group": [
        "Black", "White", "Off White", "Beige", "Light Beige", "Dark Beige", "Grey", "Dark Grey",
        "Light Grey", "Blue", "Dark Blue", "Light Blue", "Red", "Dark Red", "Pink", "Light Pink",
        "Dark Pink", "Green", "Dark Green", "Greenish Khaki", "Yellow", "Orange", "Dark Orange",
        "Purple", "Turquoise", "Gold", "Silver", "Yellowish Brown",
    ],
    "fit": ["Tight/Skinny/Bodycon", "Oversized/Loose/Relaxed", "Slim/Tailored", "Regular/Straight"],
    "occasion": [
        "Party/Evening/Wedding", "Sport/Active/Workout", "Office/Workwear", "Lounge/Sleep/Nightwear",
        "Beach/Swimwear", "Casual/Everyday", "Intimate/Underwear", "Outdoor/Adventure",
    ],
    "seasonality": ["Spring/Summer", "Autumn/Winter", "Core/All-year"],
}

GARMENTS = [
    "dress", "jacket", "coat", "parka", "blazer", "trousers", "jeans", "shorts", "skirt", "sweater",
    "cardigan", "hoodie", "t-shirt", "shirt", "blouse", "top", "bodysuit", "jumpsuit", "sneakers",
    "boots", "sandals", "heels", "bag", "scarf", "hat", "swimsuit", "bikini", "leggings", "polo shirt",
]
# plausibility constraints (cut nonsense specs the judge would just drop):
#   fit doesn't apply to footwear/accessories; Intimate/Underwear not synthesised with these garments.
NO_FIT_GARMENTS = {"sneakers", "boots", "sandals", "heels", "bag", "scarf", "hat"}
GEN_OCCASIONS = [o for o in SLOT_VALUES["occasion"] if o != "Intimate/Underwear"]

# encourage rare/colloquial surface forms so the model learns surface -> canonical generalisation
RARE_SURFACE_HINT = {
    "Dark Red": "burgundy, oxblood, wine, maroon, crimson", "Dark Blue": "navy, cobalt, indigo, midnight",
    "Greenish Khaki": "khaki, olive, army green", "Dark Green": "emerald, forest, bottle green, jade",
    "Beige": "camel, tan, sand, oatmeal, stone, nude", "Off White": "ivory, cream, eggshell, chalk",
    "Dark Grey": "charcoal, slate, gunmetal", "Yellow": "mustard, lemon, ochre",
    "Dark Orange": "rust, terracotta, burnt orange", "Purple": "lavender, lilac, violet, plum, mauve",
    "Turquoise": "teal, aqua, cyan, seafoam", "Pink": "salmon, coral, rose, blush",
    "Yellowish Brown": "chocolate, mocha, bronze, coffee, tobacco",
}

LENGTH_BUCKETS = {
    "short": {"min_words": 2, "max_words": 5, "ratio": 0.3},
    "medium": {"min_words": 6, "max_words": 12, "ratio": 0.4},
    "long": {"min_words": 13, "max_words": 32, "ratio": 0.3},
}

BACKBONE = "microsoft/deberta-v3-base"
MAX_LENGTH = 64
OUTPUT_MODEL_DIR = REPO_ROOT / "model_cache" / "slot_extractor_deberta"
