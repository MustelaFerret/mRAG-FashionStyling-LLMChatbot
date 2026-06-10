"""Cross-encoder reranker stage (relevance re-ranking after hybrid retrieval).

The hybrid retriever (SigLIP dense + BM25 sparse, RRF) is a bi-encoder: query and item
are embedded independently, so fine relevance ordering at the very top is weak (gold-set
recall@10 high but rank@1 lower). A cross-encoder reads (query, item_text) jointly with
full attention and produces a single relevance logit -- the standard "biggest quality jump
per cost" stage. Used to re-order a small candidate pool, not to retrieve.

Kept dependency-light: plain transformers AutoModelForSequenceClassification, no
FlagEmbedding. Lazy single load; scores in batches on GPU when available.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.backend.core.config import settings


class CrossEncoderReranker:
    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 64,
    ) -> None:
        self.model_id = model_id or settings.reranker_model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=settings.cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, cache_dir=settings.cache_dir
        ).to(self.device).eval()

    @torch.no_grad()
    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        """Relevance logit for (query, doc) per doc. Higher = more relevant."""
        if not docs:
            return []
        out: List[float] = []
        for start in range(0, len(docs), self.batch_size):
            batch = [[query, d or ""] for d in docs[start : start + self.batch_size]]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            logits = self.model(**inputs).logits.view(-1).float().cpu().tolist()
            out.extend(logits)
        return out

    def rerank(self, query: str, docs: Sequence[str], ids: Sequence[str]) -> List[str]:
        """Return ids re-ordered by descending relevance of their doc text."""
        scores = self.score(query, docs)
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
        return [ids[i] for i in order]

    def free(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
