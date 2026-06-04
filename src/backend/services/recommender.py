from __future__ import annotations

import base64
from io import BytesIO
from typing import Dict

from PIL import Image

from src.backend.core.config import Settings


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
                "tagline": "AI Fashion Styling Studio",
                "api_version": "2",
                "theme": "editorial",
            },
            "suggested_prompts": [
                "Find pants that match this top",
                "Show another color of #article_id",
                "Build a smart-casual outfit from this item",
                "Find a cleaner minimal version of this look",
            ],
        }
