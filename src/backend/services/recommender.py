"""Frontend bootstrap + image decode helper.

LỊCH SỬ: file này từng chứa `FashionAssistantService.handle_chat` (~880 dòng) —
một pipeline recommend riêng KHÔNG được endpoint nào gọi (chat router dùng
`FashionRAGService`). Đã xóa toàn bộ dead code (Step 4b, 2026-06-03).

Còn lại 2 thứ THỰC SỰ được dùng:
- `FashionAssistantService.decode_image` (static) — gọi từ chat_router để decode base64 ảnh upload.
- `FashionAssistantService.frontend_bootstrap` — gọi từ endpoint /api/frontend/bootstrap.

Giữ nguyên tên class + signature `__init__(settings)` để không phá API surface của
chat_router / main.py. (Có thể rename file/class thành `bootstrap.py` sau nếu muốn.)
"""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Dict

from PIL import Image

from src.backend.core.config import Settings


INTENT_SIMILAR = "similar_items"
INTENT_GRAPH = "graph_pairing"
INTENT_VARIANT = "color_variant"

INTENT_UI_META = {
    INTENT_SIMILAR: {
        "label": "Similar Picks",
        "description": "Find the closest items by style, texture, and overall vibe from the current focus item.",
    },
    INTENT_GRAPH: {
        "label": "Outfit Pairing",
        "description": "Prioritize items with strong pairing likelihood based on the co-buy graph.",
    },
    INTENT_VARIANT: {
        "label": "Color Variants",
        "description": "Focus on nearby variants and alternate colors of the same design.",
    },
}


class FashionAssistantService:
    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def decode_image(base64_str: str | None) -> Image.Image | None:
        if not base64_str:
            return None
        try:
            payload = base64_str.split(",", 1)[1] if "," in base64_str else base64_str
            img = Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")
            img.thumbnail((512, 512))
            return img
        except Exception:
            return None

    def frontend_bootstrap(self) -> Dict:
        return {
            "app": {
                "name": "Atelier mRAG",
                "tagline": "Retro Vintage Multimodal Styling Console",
                "api_version": "2",
                "theme": "retro-vintage",
            },
            "tech_stack": [
                {"name": "React 18", "role": "Component UI runtime"},
                {"name": "Next.js App Router", "role": "Route-first architecture and image optimization"},
                {"name": "Tailwind CSS", "role": "Utility-first styling"},
                {"name": "shadcn/ui", "role": "Headless UI primitives with full code ownership"},
                {"name": "Zustand", "role": "Lightweight global state for chat and anchor context"},
                {"name": "Framer Motion", "role": "Interaction animation and transition UX"},
                {"name": "React Query", "role": "API cache, request states, and mutation orchestration"},
                {"name": "FastAPI", "role": "Backend orchestration API"},
                {"name": "Qdrant", "role": "Vector retrieval store"},
                {"name": "Qwen2-VL", "role": "Multimodal reasoning and summary"},
                {"name": "SigLIP", "role": "Joint image-text embedding"},
                {"name": "Graph RAG", "role": "Outfit pairing via co-buy graph"},
            ],
            "capabilities": {
                "image_upload": True,
                "anchor_selection": True,
                "weighted_joint_embedding": True,
                "graph_pairing": True,
                "multi_hop_graph": True,
                "query_understanding_filters": True,
                "query_rewrite": self.settings.use_query_rewrite,
                "quick_actions": True,
            },
            "defaults": {
                "topk_similar": self.settings.topk_similar,
                "topk_graph": self.settings.topk_graph,
                "topk_variants": self.settings.topk_variants,
                "max_vision_images": self.settings.max_vision_images,
                "max_ui_items": 10,
            },
            "intents": [
                {"id": intent_id, **meta}
                for intent_id, meta in INTENT_UI_META.items()
            ],
            "suggested_prompts": [
                "Find pants that match this top",
                "Show another color of #article_id",
                "Build a smart-casual outfit from this item",
                "Find a cleaner minimal version of this look",
            ],
        }
