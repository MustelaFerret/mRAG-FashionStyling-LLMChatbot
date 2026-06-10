from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict

from src.backend.core.utils import normalize_text

# Split on any non-alphanumeric run so slash-joined attribute values
# (e.g. "Leather/Faux Leather", "Silk/Satin/Chiffon") become separate, queryable
# tokens instead of one un-matchable token.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


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


class SparseBM25Encoder:
    """BM25 sparse encoder. Doc weights carry idf + length-saturation; query weights are
    binary (idf folded into the doc side) so a sparse dot product yields the BM25 score."""

    def __init__(self, min_df: int = 2, max_df_ratio: float = 0.95, k1: float = 1.5, b: float = 0.75):
        self.min_df = max(1, int(min_df))
        self.max_df_ratio = float(max_df_ratio)
        self.k1 = float(k1)
        self.b = float(b)
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []
        self.avgdl: float = 0.0

    def _tokenize(self, text: str) -> list[str]:
        value = normalize_text(text)
        return _TOKEN_RE.findall(value) if value else []

    def fit(self, texts: list[str]) -> "SparseBM25Encoder":
        token_lists = [self._tokenize(t) for t in texts]
        doc_freq: dict[str, int] = defaultdict(int)
        doc_count = 0
        for toks in token_lists:
            if not toks:
                continue
            doc_count += 1
            for token in set(toks):
                doc_freq[token] += 1

        max_df = int(self.max_df_ratio * doc_count) if doc_count else 0
        vocab_terms = sorted(
            t for t, c in doc_freq.items() if c >= self.min_df and (max_df == 0 or c <= max_df)
        )
        self.vocab = {t: i for i, t in enumerate(vocab_terms)}
        self.idf = [math.log(1.0 + (doc_count - doc_freq[t] + 0.5) / (doc_freq[t] + 0.5)) for t in vocab_terms]

        total_len, n_nonempty = 0, 0
        for toks in token_lists:
            dl = sum(1 for t in toks if t in self.vocab)
            if dl > 0:
                total_len += dl
                n_nonempty += 1
        self.avgdl = total_len / n_nonempty if n_nonempty else 0.0
        return self

    def encode_doc(self, text: str) -> tuple[list[int], list[float]]:
        if not self.vocab or self.avgdl <= 0:
            return [], []
        counts = Counter(t for t in self._tokenize(text) if t in self.vocab)
        if not counts:
            return [], []
        dl = float(sum(counts.values()))
        pairs = []
        for token, c in counts.items():
            idx = self.vocab[token]
            denom = c + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            weight = self.idf[idx] * (c * (self.k1 + 1.0)) / denom if denom else 0.0
            if weight > 0.0:
                pairs.append((idx, float(weight)))
        pairs.sort(key=lambda x: x[0])
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        idxs = sorted({self.vocab[t] for t in self._tokenize(text) if t in self.vocab})
        return idxs, [1.0] * len(idxs)

    # alias so callers expecting `.encode` still get the query form
    def encode(self, text: str) -> tuple[list[int], list[float]]:
        return self.encode_query(text)

    def to_dict(self) -> dict:
        return {"method": "bm25", "min_df": self.min_df, "max_df_ratio": self.max_df_ratio,
                "k1": self.k1, "b": self.b, "avgdl": self.avgdl, "vocab": self.vocab, "idf": self.idf}

    @classmethod
    def from_dict(cls, data: dict) -> "SparseBM25Encoder":
        enc = cls(min_df=int(data.get("min_df", 2)), max_df_ratio=float(data.get("max_df_ratio", 0.95)),
                  k1=float(data.get("k1", 1.5)), b=float(data.get("b", 0.75)))
        enc.vocab = {str(k): int(v) for k, v in (data.get("vocab") or {}).items()}
        enc.idf = [float(v) for v in (data.get("idf") or [])]
        enc.avgdl = float(data.get("avgdl", 0.0))
        return enc

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "SparseBM25Encoder":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def load_sparse_encoder(path: str):
    """Load either sparse encoder based on the saved `method` field (default tfidf)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("method") == "bm25":
        return SparseBM25Encoder.from_dict(data)
    return SparseTfidfEncoder.from_dict(data)
