"""all-MiniLM-L6-v2 text encoder (mean-pooled, L2-normalized) for the OOV span linker.
Same encode_texts() signature as SigLIPEncoder. Loaded via plain transformers (no
sentence-transformers dep). ~90MB download once into model_cache."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.backend.core.config import settings

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMEncoder:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=settings.cache_dir)
        self.model = AutoModel.from_pretrained(MODEL_ID, cache_dir=settings.cache_dir).eval().to(self.device)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        out = []
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
            tok_emb = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(F.normalize(mean, p=2, dim=1).cpu().numpy())
        return np.vstack(out)
