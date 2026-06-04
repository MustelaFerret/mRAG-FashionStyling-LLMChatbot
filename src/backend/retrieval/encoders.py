from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, SiglipModel

from src.backend.core.config import settings
from src.backend.core.utils import get_local_model_path
from src.backend.retrieval.embeddings import SparseTfidfEncoder


class SigLIPEncoder:
    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model_id = model_id or settings.siglip_model_id
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)

        model_path = get_local_model_path(settings.cache_dir, self.model_id)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=settings.llm_local_files_only,
        )
        self.model = SiglipModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            local_files_only=settings.llm_local_files_only,
        ).to(self.device)
        self.model.eval()
        self.embed_dim: int = int(self.model.config.text_config.hidden_size)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = True,
        progress_desc: str = "encode_texts",
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        out = np.empty((len(texts), self.embed_dim), dtype=np.float32)
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=progress_desc, total=(len(texts) + batch_size - 1) // batch_size)
        for start in iterator:
            batch = [t if t else "fashion item" for t in texts[start : start + batch_size]]
            inputs = self.processor(
                text=batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            features = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            features = self._extract_tensor(features)
            features = F.normalize(features, p=2, dim=-1)
            out[start : start + len(batch)] = features.cpu().float().numpy()
        return out

    @torch.no_grad()
    def encode_images(
        self,
        images: Sequence[Image.Image | str | Path | None],
        batch_size: int = 32,
        show_progress: bool = True,
        progress_desc: str = "encode_images",
    ) -> np.ndarray:
        if not images:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        out = np.zeros((len(images), self.embed_dim), dtype=np.float32)
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=progress_desc, total=(len(images) + batch_size - 1) // batch_size)
        for start in iterator:
            chunk = list(images[start : start + batch_size])
            loaded: List[Image.Image] = []
            keep_idx: List[int] = []
            for offset, item in enumerate(chunk):
                img = self._load_image(item)
                if img is not None:
                    loaded.append(img)
                    keep_idx.append(start + offset)
            if not loaded:
                continue
            inputs = self.processor(images=loaded, return_tensors="pt").to(self.device)
            pixel_values = inputs["pixel_values"].to(self.dtype)
            features = self.model.get_image_features(pixel_values=pixel_values)
            features = self._extract_tensor(features)
            features = F.normalize(features, p=2, dim=-1)
            arr = features.cpu().float().numpy()
            for j, target_idx in enumerate(keep_idx):
                out[target_idx] = arr[j]
            for img in loaded:
                img.close()
        return out

    def encode_text(self, text: str) -> List[float]:
        vec = self.encode_texts([text], batch_size=1, show_progress=False)
        return vec[0].tolist()

    def encode_image(self, image: Image.Image | str | Path) -> List[float]:
        vec = self.encode_images([image], batch_size=1, show_progress=False)
        return vec[0].tolist()

    @staticmethod
    def _extract_tensor(features) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
            value = getattr(features, attr, None)
            if isinstance(value, torch.Tensor):
                if value.ndim == 3:
                    return value[:, 0, :]
                return value
        if hasattr(features, "__getitem__"):
            value = features[0]
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError(f"Cannot extract tensor from features of type {type(features).__name__}")

    @staticmethod
    def _load_image(value: Image.Image | str | Path | None) -> Image.Image | None:
        if value is None:
            return None
        if isinstance(value, Image.Image):
            return value.convert("RGB") if value.mode != "RGB" else value
        path = str(value)
        if not path or not os.path.exists(path):
            return None
        try:
            with Image.open(path) as img:
                return img.convert("RGB").copy()
        except Exception:
            return None

    def free(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class QueryEncoder:
    def __init__(self, siglip: SigLIPEncoder, sparse: SparseTfidfEncoder | None = None) -> None:
        self.siglip = siglip
        self.sparse = sparse

    def encode(
        self,
        text: str | None = None,
        image: Image.Image | str | Path | None = None,
        sparse_text: str | None = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "text_dense": None,
            "image_dense": None,
            "sparse_indices": [],
            "sparse_values": [],
        }
        if text:
            out["text_dense"] = self.siglip.encode_text(text)
        if image is not None:
            out["image_dense"] = self.siglip.encode_image(image)
        if self.sparse is not None:
            sp_text = sparse_text if sparse_text is not None else (text or "")
            if sp_text:
                idx, val = self.sparse.encode(sp_text)
                out["sparse_indices"] = idx
                out["sparse_values"] = val
        return out
