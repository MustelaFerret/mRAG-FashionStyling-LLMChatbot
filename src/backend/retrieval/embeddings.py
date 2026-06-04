from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

from src.backend.core.utils import normalize_text


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
