"""In-process hybrid search index over the (embedded) Qdrant collection.

Qdrant's local mode scans pure-Python per query: measured 1.7s for the sparse branch and
~2.7s more when payload filters apply (70k points), i.e. ~4.6s per hybrid_search call.
This index pulls the collection into numpy once (cached to disk as npz) and serves the
same query: dense cosine via fp16 matmul (~30ms), sparse via scipy CSR dot, payload
filters via vectorized string-equality masks, then the same per-branch prefetch + RRF
fusion. Equivalence vs the Qdrant path is verified in temp/verify_local_hybrid.py and the
gold-set eval; QdrantStore falls back to the native path if the index is unavailable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from scipy.sparse import csr_matrix

_RRF_K = 2  # qdrant's RRF constant (verified by output equivalence)
_FILTER_ALIASES = {
    "product_type": ["product_type", "product_type_name"],
    "colour_group": ["colour_group", "colour_group_name"],
}


@dataclass
class LocalPoint:
    id: int
    score: float
    payload: Dict[str, Any]


class LocalHybridIndex:
    def __init__(self, client, collection_name: str, cache_path: str):
        self.ready = False
        self.payloads: List[Dict[str, Any]] = []
        try:
            if not self._load_cache(client, collection_name, cache_path):
                self._build(client, collection_name)
                self._save_cache(cache_path)
            self._finalize()
            self.ready = True
        except Exception:
            self.ready = False

    # -- construction ------------------------------------------------------
    def _build(self, client, collection_name: str) -> None:
        ids: List[int] = []
        payloads: List[Dict[str, Any]] = []
        text_rows: List[np.ndarray] = []
        image_rows: List[np.ndarray] = []
        sp_indptr, sp_indices, sp_data = [0], [], []
        offset = None
        dim = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name, limit=2048, offset=offset,
                with_payload=True, with_vectors=True,
            )
            for p in points:
                vec = p.vector or {}
                t = vec.get("text_emb")
                i = vec.get("image_emb")
                if t is None or i is None:
                    continue
                if dim is None:
                    dim = len(t)
                ids.append(int(p.id))
                payloads.append(dict(p.payload or {}))
                text_rows.append(np.asarray(t, dtype=np.float16))
                image_rows.append(np.asarray(i, dtype=np.float16))
                sp = vec.get("sparse_bm25")
                idx = list(getattr(sp, "indices", []) or [])
                val = list(getattr(sp, "values", []) or [])
                sp_indices.extend(int(x) for x in idx)
                sp_data.extend(float(x) for x in val)
                sp_indptr.append(len(sp_indices))
            if offset is None:
                break
        self.ids = np.asarray(ids, dtype=np.int64)
        self.text = np.vstack(text_rows) if text_rows else np.zeros((0, dim or 768), np.float16)
        self.image = np.vstack(image_rows) if image_rows else np.zeros((0, dim or 768), np.float16)
        n_vocab = (max(sp_indices) + 1) if sp_indices else 1
        self.sparse = csr_matrix(
            (np.asarray(sp_data, np.float32), np.asarray(sp_indices, np.int32), np.asarray(sp_indptr, np.int32)),
            shape=(len(ids), n_vocab),
        )
        self.payloads = payloads

    def _save_cache(self, cache_path: str) -> None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            ids=self.ids, text=self.text, image=self.image,
            sp_data=self.sparse.data, sp_indices=self.sparse.indices,
            sp_indptr=self.sparse.indptr, sp_shape=np.asarray(self.sparse.shape),
        )
        with open(cache_path + ".payloads.json", "w", encoding="utf-8") as f:
            json.dump(self.payloads, f, ensure_ascii=False)

    def _load_cache(self, client, collection_name: str, cache_path: str) -> bool:
        pj = cache_path + ".payloads.json"
        if not (os.path.exists(cache_path) and os.path.exists(pj)):
            return False
        z = np.load(cache_path)
        ids = z["ids"]
        # staleness check: the cache must mirror the live collection's size
        try:
            live = client.count(collection_name=collection_name).count
        except Exception:
            live = None
        if live is not None and live != len(ids):
            return False
        self.ids = ids
        self.text = z["text"]
        self.image = z["image"]
        self.sparse = csr_matrix((z["sp_data"], z["sp_indices"], z["sp_indptr"]), shape=tuple(z["sp_shape"]))
        self.payloads = json.load(open(pj, encoding="utf-8"))
        return True

    def _finalize(self) -> None:
        # L2-normalize dense once -> cosine becomes a dot product
        def norm(m):
            m32 = m.astype(np.float32)
            n = np.linalg.norm(m32, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return (m32 / n).astype(np.float16)
        self.text = norm(self.text)
        self.image = norm(self.image)
        # filterable payload columns as object arrays for vectorized equality
        keys = set()
        for aliases in _FILTER_ALIASES.values():
            keys.update(aliases)
        for p in self.payloads[:50]:
            keys.update(k for k, v in p.items() if isinstance(v, str))
        self.columns: Dict[str, np.ndarray] = {}
        for k in keys:
            self.columns[k] = np.asarray([str(p.get(k, "") or "") for p in self.payloads], dtype=object)

    # -- query -------------------------------------------------------------
    def _mask(self, must: Dict | None, must_not: Dict | None) -> np.ndarray | None:
        n = len(self.ids)
        mask = None

        def col_eq(key: str, values: List[str]) -> np.ndarray:
            hit = np.zeros(n, dtype=bool)
            for alias in _FILTER_ALIASES.get(key, [key]):
                col = self.columns.get(alias)
                if col is None:
                    continue
                for v in values:
                    hit |= col == v
            return hit

        def vals_of(value) -> List[str]:
            if isinstance(value, (list, tuple)):
                return [str(v).strip() for v in value if str(v).strip()]
            s = str(value).strip()
            return [s] if s else []

        for key, value in (must or {}).items():
            vals = vals_of(value)
            if not vals:
                continue
            m = col_eq(key, vals)
            mask = m if mask is None else (mask & m)
        for key, value in (must_not or {}).items():
            vals = vals_of(value)
            if not vals:
                continue
            m = ~col_eq(key, vals)
            mask = m if mask is None else (mask & m)
        return mask

    def _top_ranks(self, scores: np.ndarray, mask: np.ndarray | None, k: int) -> List[int]:
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(scores))
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part])]
        return [int(i) for i in order if np.isfinite(scores[i])]

    def search(
        self,
        text_dense: List[float] | None,
        image_dense: List[float] | None,
        sparse_indices: List[int] | None,
        sparse_values: List[float] | None,
        limit: int = 10,
        must_filters: Dict | None = None,
        must_not_filters: Dict | None = None,
        prefetch_multiplier: int = 4,
    ) -> List[LocalPoint]:
        mask = self._mask(must_filters, must_not_filters)
        prefetch_limit = max(limit * prefetch_multiplier, limit + 5)
        branches: List[List[int]] = []

        if text_dense:
            q = np.asarray(text_dense, dtype=np.float32)
            q /= (np.linalg.norm(q) or 1.0)
            branches.append(self._top_ranks(self.text.astype(np.float32) @ q, mask, prefetch_limit))
        if image_dense:
            q = np.asarray(image_dense, dtype=np.float32)
            q /= (np.linalg.norm(q) or 1.0)
            branches.append(self._top_ranks(self.image.astype(np.float32) @ q, mask, prefetch_limit))
        if sparse_indices and sparse_values:
            qv = np.zeros(self.sparse.shape[1], dtype=np.float32)
            for i, v in zip(sparse_indices, sparse_values):
                if 0 <= int(i) < len(qv):
                    qv[int(i)] = float(v)
            branches.append(self._top_ranks(self.sparse @ qv, mask, prefetch_limit))

        branches = [b for b in branches if b]
        if not branches:
            return []
        if len(branches) == 1:
            chosen = branches[0][:limit]
            return [LocalPoint(id=int(self.ids[i]), score=0.0, payload=self.payloads[i]) for i in chosen]

        rrf: Dict[int, float] = {}
        for branch in branches:
            for rank, i in enumerate(branch):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chosen = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:limit]
        return [LocalPoint(id=int(self.ids[i]), score=rrf[i], payload=self.payloads[i]) for i in chosen]
