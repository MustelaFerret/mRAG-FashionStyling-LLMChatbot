from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, SiglipModel

from src.backend.core.config import settings
from src.backend.core.utils import get_local_model_path, normalize_text


class SparseTfidfEncoder:
    def __init__(self, min_df: int = 2, max_df_ratio: float = 0.95):
        self.min_df = max(1, int(min_df))
        self.max_df_ratio = float(max_df_ratio)
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []

    def _tokenize(self, text: str) -> list[str]:
        value = normalize_text(text)
        if not value:
            return []
        return [tok for tok in value.split() if tok]

    def fit(self, texts: list[str]) -> "SparseTfidfEncoder":
        doc_freq: dict[str, int] = defaultdict(int)
        doc_count = 0
        for text in texts:
            tokens = set(self._tokenize(text))
            if not tokens:
                continue
            doc_count += 1
            for token in tokens:
                doc_freq[token] += 1

        max_df = int(self.max_df_ratio * doc_count) if doc_count else 0
        vocab_terms = [
            token
            for token, count in doc_freq.items()
            if count >= self.min_df and (max_df == 0 or count <= max_df)
        ]
        vocab_terms.sort()
        self.vocab = {token: idx for idx, token in enumerate(vocab_terms)}
        self.idf = [0.0 for _ in vocab_terms]

        for token, idx in self.vocab.items():
            df_value = doc_freq[token]
            self.idf[idx] = math.log((1.0 + doc_count) / (1.0 + df_value)) + 1.0

        return self

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        if not self.vocab:
            return [], []
        tokens = self._tokenize(text)
        if not tokens:
            return [], []
        counts = Counter(token for token in tokens if token in self.vocab)
        if not counts:
            return [], []
        total = float(sum(counts.values()))
        indices: list[int] = []
        values: list[float] = []
        for token, count in counts.items():
            idx = self.vocab[token]
            tf = float(count) / total
            value = tf * self.idf[idx]
            if value > 0.0:
                indices.append(idx)
                values.append(float(value))
        pairs = sorted(zip(indices, values), key=lambda x: x[0])
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def encode_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [self.encode(text) for text in texts]

    def to_dict(self) -> dict:
        return {
            "min_df": self.min_df,
            "max_df_ratio": self.max_df_ratio,
            "vocab": self.vocab,
            "idf": self.idf,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SparseTfidfEncoder":
        encoder = cls(
            min_df=int(data.get("min_df", 2)),
            max_df_ratio=float(data.get("max_df_ratio", 0.95)),
        )
        encoder.vocab = {str(k): int(v) for k, v in (data.get("vocab") or {}).items()}
        encoder.idf = [float(v) for v in (data.get("idf") or [])]
        return encoder

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "SparseTfidfEncoder":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class HybridEmbeddingService:
    def __init__(self, sparse_encoder: SparseTfidfEncoder | None = None):
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
        self.sparse_encoder = sparse_encoder

    @staticmethod
    def _normalize_weights(text_weight: float | None, image_weight: float | None) -> tuple[float, float]:
        tw = 0.5 if text_weight is None else max(0.0, float(text_weight))
        iw = 0.5 if image_weight is None else max(0.0, float(image_weight))
        total = tw + iw
        if total <= 0:
            return 0.5, 0.5
        return tw / total, iw / total

    def set_sparse_encoder(self, encoder: SparseTfidfEncoder | None) -> None:
        self.sparse_encoder = encoder

    def fit_sparse(self, texts: list[str], min_df: int = 2, max_df_ratio: float = 0.95) -> SparseTfidfEncoder:
        encoder = SparseTfidfEncoder(min_df=min_df, max_df_ratio=max_df_ratio)
        encoder.fit(texts)
        self.sparse_encoder = encoder
        return encoder

    @torch.no_grad()
    def encode_dense(
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
                attention_mask=inputs.get("attention_mask"),
            )

            if not isinstance(image_features, torch.Tensor):
                image_features = getattr(
                    image_features,
                    "image_embeds",
                    getattr(image_features, "pooler_output", image_features[0]),
                )
            if not isinstance(text_features, torch.Tensor):
                text_features = getattr(
                    text_features,
                    "text_embeds",
                    getattr(text_features, "pooler_output", text_features[0]),
                )

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
            attention_mask=inputs.get("attention_mask"),
        )

        if not isinstance(text_features, torch.Tensor):
            text_features = getattr(
                text_features,
                "text_embeds",
                getattr(text_features, "pooler_output", text_features[0]),
            )

        return F.normalize(text_features, p=2, dim=-1)[0].cpu().float().numpy().tolist()

    def encode_sparse(self, text: str) -> tuple[list[int], list[float]]:
        if self.sparse_encoder is None:
            return [], []
        return self.sparse_encoder.encode(text)

    def encode_hybrid(
        self,
        dense_text: str,
        sparse_text: str,
        image: Image.Image | None = None,
        text_weight: float | None = None,
        image_weight: float | None = None,
    ) -> tuple[list[float], list[int], list[float]]:
        dense_vector = self.encode_dense(dense_text, image=image, text_weight=text_weight, image_weight=image_weight)
        sparse_indices, sparse_values = self.encode_sparse(sparse_text)
        return dense_vector, sparse_indices, sparse_values

    def encode(
        self,
        text: str,
        image: Image.Image | None = None,
        text_weight: float | None = None,
        image_weight: float | None = None,
    ) -> list[float]:
        return self.encode_dense(text, image=image, text_weight=text_weight, image_weight=image_weight)


class SiglipEmbeddingService(HybridEmbeddingService):
    pass