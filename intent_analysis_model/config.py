from __future__ import annotations

import os
import re
from pathlib import Path

_VN_DIACRITICS = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)


def is_english(text: str) -> bool:
    return not bool(_VN_DIACRITICS.search(str(text)))


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

GEN_MODELS_TRAIN = [
    "gpt-5.4-mini",
    "deepseek-3.2",
    "glm-5",
    "minimax-m2.5",
    "claude-haiku-4.5",
]
GEN_MODELS_TEST = ["gpt-5.4"]
JUDGE_MODEL_TRAIN = "claude-sonnet-4.5"
JUDGE_MODEL_TEST = "claude-sonnet-4.5"

INTENT_LABELS = [
    "similar_items",
    "graph_pairing",
    "color_variant",
    "composite_intent",
    "chit_chat",
]
LABEL2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

INTENT_DEFINITIONS = {
    "similar_items": (
        "User muốn TÌM MỘT món đồ (đúng 1 loại sản phẩm) theo mô tả. Mô tả có thể RẤT NGẮN "
        "('padded jacket') HOẶC RẤT DÀI & CHI TIẾT ('sun-faded charcoal 14oz boxy hoodie with "
        "cropped hem and dropped shoulders'). Độ dài KHÔNG quyết định nhãn — miễn là CHỈ 1 món "
        "mục tiêu, không yêu cầu phối với món khác, không hỏi màu khác của món đang xem."
    ),
    "graph_pairing": (
        "User muốn tìm món ĐỂ PHỐI / ĐI KÈM với một món khác (thường là item đang xem). "
        "Đặc trưng: quan hệ 'match/pair/go with/style with/complete the look'. CHỈ 1 món cần tìm "
        "(món để phối). Ví dụ: 'a bag that goes with this dress', 'shoes to match these trousers'."
    ),
    "color_variant": (
        "User hỏi món HIỆN TẠI (đang xem) có ở MÀU hoặc PHIÊN BẢN khác không. Cùng một món, đổi màu. "
        "Ví dụ: 'is this available in red?', 'does it come in black', 'other colorways of this'. "
        "PHÂN BIỆT: 'find a red jacket' (món MỚI màu đỏ) = similar_items, KHÔNG phải color_variant."
    ),
    "composite_intent": (
        "Câu chứa TỪ 2 INTENT RIÊNG BIỆT trở lên. KEY: phải có >=2 hành động/mục tiêu tách biệt, "
        "KHÔNG phải vì câu dài. Ví dụ: 'find a white dress AND earrings to match it' "
        "(similar + graph), 'show this in red and also find shoes to pair' (color_variant + graph). "
        "Câu DÀI mô tả chi tiết MỘT món KHÔNG phải composite — đó là similar_items."
    ),
    "chit_chat": (
        "KHÔNG phải intent mua sắm sản phẩm. Small talk, off-topic, câu linh tinh, gibberish, "
        "hỏi về shipping/refund/policy, lời chào. Ví dụ: 'hello', 'how are you', 'it's raining lol', "
        "'where is my order'."
    ),
}

HARD_CASE_RULES = [
    "Câu DÀI + nhiều thuộc tính (màu, chất liệu, fit, dáng) + 1 món => similar_items (KHÔNG composite).",
    "'X to match/pair with Y' => graph_pairing (1 intent pairing, KHÔNG composite).",
    "'X AND Y' với 2 món riêng biệt => composite_intent.",
    "'this/it in <color>' => color_variant; 'find a <color> X' (món mới) => similar_items.",
    "Câu NGẮN vẫn có thể là composite nếu có 2 intent ('red? + matching bag?').",
]

LENGTH_BUCKETS = {
    "short": {"min_words": 1, "max_words": 4, "ratio": 0.25},
    "medium": {"min_words": 5, "max_words": 12, "ratio": 0.40},
    "long": {"min_words": 13, "max_words": 35, "ratio": 0.35},
}

PER_INTENT_TRAIN = 1500
PER_INTENT_TEST = 200
VAL_RATIO = 0.12

BACKBONE = "microsoft/deberta-v3-base"
MAX_LENGTH = 128
OUTPUT_MODEL_DIR = REPO_ROOT / "model_cache" / "intent_classifier_deberta"
