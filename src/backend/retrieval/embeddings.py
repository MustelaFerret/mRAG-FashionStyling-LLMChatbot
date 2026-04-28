from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, SiglipModel

from src.backend.core.config import settings
from src.backend.core.utils import get_local_model_path


class SiglipEmbeddingService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested_name = os.getenv("EMBED_DEVICE", "cpu")
        requested_device = torch.device(requested_name)
        if requested_device.type == "cuda" and torch.cuda.is_available():
            self.device = requested_device
        elif requested_device.type == "cpu":
            self.device = requested_device

        model_path = get_local_model_path(settings.cache_dir, settings.siglip_model_id)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=settings.llm_local_files_only,
        )
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = SiglipModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            local_files_only=settings.llm_local_files_only,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _normalize_weights(text_weight: float | None, image_weight: float | None) -> tuple[float, float]:
        tw = 0.5 if text_weight is None else max(0.0, float(text_weight))
        iw = 0.5 if image_weight is None else max(0.0, float(image_weight))
        total = tw + iw
        if total <= 0:
            return 0.5, 0.5
        return tw / total, iw / total

    @torch.no_grad()
    def encode(
        self,
        text: str,
        image: Image.Image | None = None,
        text_weight: float | None = None,
        image_weight: float | None = None,
    ) -> list[float]:
        if image is not None:
            inputs = self.processor(
                text=[text or ""],
                images=[image],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            inputs = {k: v.to(self.dtype) if v.is_floating_point() else v for k, v in inputs.items()}
            
            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = self.model.get_text_features(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs.get("attention_mask")
            )
            
            if not isinstance(image_features, torch.Tensor):
                image_features = getattr(image_features, "image_embeds", getattr(image_features, "pooler_output", image_features[0]))
            if not isinstance(text_features, torch.Tensor):
                text_features = getattr(text_features, "text_embeds", getattr(text_features, "pooler_output", text_features[0]))

            tw, iw = self._normalize_weights(text_weight, image_weight)
            norm_image = F.normalize(image_features, p=2, dim=-1)
            norm_text = F.normalize(text_features, p=2, dim=-1)
            fused = (iw * norm_image) + (tw * norm_text)
            return F.normalize(fused, p=2, dim=-1)[0].cpu().float().numpy().tolist()

        inputs = self.processor(
            text=[text or "fashion item"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        inputs = {k: v.to(self.dtype) if v.is_floating_point() else v for k, v in inputs.items()}
        
        text_features = self.model.get_text_features(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs.get("attention_mask")
        )
        
        if not isinstance(text_features, torch.Tensor):
            text_features = getattr(text_features, "text_embeds", getattr(text_features, "pooler_output", text_features[0]))

        return F.normalize(text_features, p=2, dim=-1)[0].cpu().float().numpy().tolist()