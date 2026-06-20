"""Personalization re-ranker — a LIGHT tie-break toward a customer's taste, applied AFTER
relevance (the retrieval/rerank order stays dominant).

The blend uses the signals a leak-free leave-last-out evaluation found to actually carry
preference: max-similarity to the nearest single past purchase, a recency-weighted taste vector,
and in-category popularity. It replaces the earlier hand-set blend, whose weights were never
measured and whose mean-vector term the evaluation assigned a *negative* weight (the mean blurs a
multi-faceted customer). The mean-vector and category-match terms are therefore dropped.

Offline inputs: per-customer purchase history (chronological article_ids, from
export_purchases.py) + the frozen SigLIP image embeddings. Profiles are static snapshots — this
nudges relevant results toward taste, it does not predict the next purchase.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.backend.core.utils import normalize_article_id


@dataclass
class RerankWeights:
    # Relevance (retrieval order) is dominant; the personalization terms are a light additive
    # nudge. Their ratio follows the learned logistic blend (recency >= popularity > maxsim);
    # the total is deliberately small so taste only reorders near-ties.
    retrieval: float = 1.0
    recency: float = 0.14
    popularity: float = 0.13
    maxsim: float = 0.07


_RECENCY_SPAN = (-2.0, 0.0)  # exp() weights over chronological purchases -> newest weighted up


class PersonalizationStore:
    def __init__(self, data_dir: str, meta_file: str):
        d = Path(data_dir)
        self.art_emb = None
        self.art_index: Dict[str, int] = {}
        self.pop: Dict[str, float] = {}
        self.cust_purch_rows: Dict[str, np.ndarray] = {}
        self.cust_recency: Dict[str, np.ndarray] = {}
        self.available = False
        self._load(d, meta_file)

    def _load(self, d: Path, meta_file: str):
        prof_path = d / "profiles.csv"
        purch_path = d / "customer_purchases.json"
        if not (prof_path.exists() and purch_path.exists()):
            return
        self.art_index = json.loads((d / "article_index.json").read_text())
        emb = np.load(d / "article_image_emb.npy").astype(np.float32)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)  # unit-norm -> dot == cosine
        self.art_emb = emb

        pop = pd.read_csv(d / "article_popularity.csv", dtype={"article_id": str})
        pop["article_id"] = pop["article_id"].astype(str).str.zfill(10)
        max_log = float(np.log1p(pop["count"].max())) or 1.0
        self.pop = {a: float(np.log1p(c) / max_log) for a, c in zip(pop["article_id"], pop["count"])}

        purch = json.loads(purch_path.read_text())
        for cid, aids in purch.items():
            rows = [self.art_index[a] for a in aids if a in self.art_index]
            if not rows:
                continue
            rows = np.asarray(rows, dtype=np.int64)
            self.cust_purch_rows[cid] = rows
            w = np.exp(np.linspace(_RECENCY_SPAN[0], _RECENCY_SPAN[1], len(rows))).astype(np.float32)
            rv = (w[:, None] * self.art_emb[rows]).sum(0)
            norm = float(np.linalg.norm(rv))
            self.cust_recency[cid] = rv / norm if norm > 0 else rv
        self.available = True

    def has_profile(self, customer_id: str) -> bool:
        return bool(customer_id) and customer_id in self.cust_purch_rows

    def rerank(self, customer_id: str, article_ids: List[str], weights: RerankWeights | None = None) -> List[str]:
        if not self.has_profile(customer_id) or not article_ids:
            return article_ids
        w = weights or RerankWeights()
        past = self.art_emb[self.cust_purch_rows[customer_id]]   # [P, 768], unit-norm
        rec = self.cust_recency[customer_id]                     # [768], unit-norm
        n = len(article_ids)

        # taste signals only defined for candidates we have an embedding for
        cand_rows = [self.art_index.get(normalize_article_id(a)) for a in article_ids]
        maxsim = np.zeros(n, dtype=np.float32)
        recency = np.zeros(n, dtype=np.float32)
        known_i = [i for i, r in enumerate(cand_rows) if r is not None]
        if known_i:
            ce = self.art_emb[np.asarray([cand_rows[i] for i in known_i])]   # [K, 768]
            ms = (ce @ past.T).max(axis=1)                                   # nearest past item
            rc = ce @ rec                                                    # recency-weighted taste
            for j, i in enumerate(known_i):
                maxsim[i] = ms[j]
                recency[i] = rc[j]

        scored = []
        for rank, raw in enumerate(article_ids):
            retrieval_prior = (n - rank) / n
            s = (w.retrieval * retrieval_prior
                 + w.recency * float(recency[rank])
                 + w.popularity * self.pop.get(normalize_article_id(raw), 0.0)
                 + w.maxsim * float(maxsim[rank]))
            scored.append((raw, s))
        scored.sort(key=lambda x: -x[1])
        return [a for a, _ in scored]
