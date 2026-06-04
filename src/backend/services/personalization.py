"""Personalization re-ranker — re-rank candidates theo taste profile của customer.

Profile (offline, từ purchase history): categorical preference + taste vector (mean image_emb)
+ price tier. Re-rank blend: retrieval prior + taste cosine + categorical match + popularity.
Snapshot thesis: profile TĨNH, không dự đoán tương lai — chỉ ưu tiên item khớp gu khi search.
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
    retrieval: float = 1.0
    taste: float = 0.6
    categorical: float = 0.5
    popularity: float = 0.2
    price: float = 0.3


PROFILE_FIELD_MAP = {
    "product_type_name": "product_type",
    "colour_group_name": "colour_group",
    "style_aesthetic": "style_aesthetic",
    "occasion": "occasion",
    "index_group_name": "index_group",
}


class PersonalizationStore:
    def __init__(self, data_dir: str, meta_file: str):
        d = Path(data_dir)
        self.profiles: Dict[str, dict] = {}
        self.taste = None
        self.taste_index: Dict[str, int] = {}
        self.art_emb = None
        self.art_index: Dict[str, int] = {}
        self.pop: Dict[str, float] = {}
        self.art_meta: Dict[str, dict] = {}
        self.art_price: Dict[str, float] = {}
        self.available = False
        self._load(d, meta_file)

    def _load(self, d: Path, meta_file: str):
        prof_path = d / "profiles.csv"
        if not prof_path.exists():
            return
        df = pd.read_csv(prof_path, dtype={"customer_id": str})
        fields = list(PROFILE_FIELD_MAP.keys())
        for _, r in df.iterrows():
            cid = str(r["customer_id"])
            self.profiles[cid] = {
                "price_median": float(r.get("price_median", 0.0)),
                "cats": {f: json.loads(r[f]) if isinstance(r.get(f), str) else {} for f in fields},
            }
        self.taste = np.load(d / "taste_vectors.npy").astype(np.float32)
        self.taste_index = json.loads((d / "taste_index.json").read_text())
        self.art_emb = np.load(d / "article_image_emb.npy").astype(np.float32)
        self.art_index = json.loads((d / "article_index.json").read_text())

        pop = pd.read_csv(d / "article_popularity.csv", dtype={"article_id": str})
        pop["article_id"] = pop["article_id"].astype(str).str.zfill(10)
        max_log = np.log1p(pop["count"].max())
        self.pop = {a: float(np.log1p(c) / max_log) for a, c in zip(pop["article_id"], pop["count"])}
        if "mean_price" in pop.columns:
            self.art_price = {a: float(p) for a, p in zip(pop["article_id"], pop["mean_price"]) if p == p}

        meta = pd.read_csv(meta_file, usecols=["article_id"] + list(PROFILE_FIELD_MAP.keys()), dtype={"article_id": str})
        meta["article_id"] = meta["article_id"].str.zfill(10)
        for _, r in meta.iterrows():
            self.art_meta[r["article_id"]] = {f: str(r.get(f, "") or "") for f in PROFILE_FIELD_MAP}
        self.available = True

    def has_profile(self, customer_id: str) -> bool:
        return bool(customer_id) and customer_id in self.profiles

    def _cat_match(self, profile: dict, aid: str) -> float:
        meta = self.art_meta.get(aid)
        if not meta:
            return 0.0
        score = 0.0
        for field in PROFILE_FIELD_MAP:
            weights = profile["cats"].get(field, {})
            val = meta.get(field, "")
            if val and val in weights:
                score += weights[val]
        return score / len(PROFILE_FIELD_MAP)

    def _taste_cos(self, customer_id: str, aid: str) -> float:
        ti = self.taste_index.get(customer_id)
        ai = self.art_index.get(aid)
        if ti is None or ai is None:
            return 0.0
        return float(np.dot(self.taste[ti], self.art_emb[ai]))

    def _price_match(self, profile: dict, aid: str) -> float:
        if not self.art_price:
            return 0.0
        ap = self.art_price.get(aid)
        pm = profile.get("price_median", 0.0)
        if ap is None or pm <= 0:
            return 0.0
        return 1.0 - min(abs(ap - pm) / pm, 1.0)

    def rerank(self, customer_id: str, article_ids: List[str], weights: RerankWeights | None = None) -> List[str]:
        if not self.has_profile(customer_id) or not article_ids:
            return article_ids
        w = weights or RerankWeights()
        profile = self.profiles[customer_id]
        n = len(article_ids)
        scored = []
        for rank, raw in enumerate(article_ids):
            aid = normalize_article_id(raw)
            retrieval_prior = (n - rank) / n
            s = (
                w.retrieval * retrieval_prior
                + w.taste * self._taste_cos(customer_id, aid)
                + w.categorical * self._cat_match(profile, aid)
                + w.popularity * self.pop.get(aid, 0.0)
                + w.price * self._price_match(profile, aid)
            )
            scored.append((raw, s))
        scored.sort(key=lambda x: -x[1])
        return [a for a, _ in scored]
