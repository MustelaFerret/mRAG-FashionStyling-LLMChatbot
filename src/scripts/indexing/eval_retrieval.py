"""DEPRECATED as a decision gate — self-retrieval leaks (the query derives from the
indexed text itself, so scores overstate real ranking quality). Use eval_goldset.py
(hand-written queries, graded relevance) for accept/reject decisions; keep this script
only as a cheap smoke check that indexing isn't catastrophically broken.

Label-free retrieval eval (self-retrieval). For a sample of items, build a short
natural query from the item's own attributes + a description snippet, run the hybrid
search, and measure how well the exact article is recovered (recall@k, MRR).

Run: python -m src.scripts.indexing.eval_retrieval
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

import numpy as np

from src.backend.core.config import settings
from src.backend.core.utils import normalize_article_id
from src.backend.retrieval.qdrant import QdrantStore
from src.backend.retrieval.encoders import QueryEncoder, SigLIPEncoder
from src.backend.retrieval.embeddings import load_sparse_encoder
from src.backend.services.catalog import FashionCatalog

SAMPLE = 300
TOPK = 10
SEED = 42
K_VALUES = (1, 5, 10)


def _query_for(meta: Dict) -> str:
    desc = str(meta.get("refined_description", "") or "")
    words = re.findall(r"[A-Za-z]+", desc)[:12]
    parts = [meta.get("colour_group_name", ""), meta.get("product_type_name", "")] + words
    return " ".join(p for p in parts if p).strip()


def _metrics(ranks: List[int]) -> Dict[str, float]:
    n = len(ranks)
    out = {f"recall@{k}": round(sum(1 for r in ranks if r <= k) / n, 4) for k in K_VALUES}
    out["mrr"] = round(sum(1.0 / r for r in ranks if r != 10**9) / n, 4)
    out["n"] = n
    return out


def main() -> None:
    catalog = FashionCatalog(settings.meta_file, settings.graph_file, settings.image_dir)
    store = QdrantStore(settings.db_path, settings.collection_name)
    sparse = load_sparse_encoder(settings.sparse_model_path) if os.path.exists(settings.sparse_model_path) else None
    embedder = QueryEncoder(siglip=SigLIPEncoder(), sparse=sparse)

    rng = np.random.default_rng(SEED)
    aids = [a for a, m in catalog.meta_by_article.items() if str(m.get("refined_description", "")).strip()]
    sample = [aids[i] for i in rng.choice(len(aids), size=min(SAMPLE, len(aids)), replace=False)]

    ranks: List[int] = []
    for n, aid in enumerate(sample, 1):
        if n % 25 == 0:
            print(f"  ...{n}/{len(sample)}", flush=True)
        meta = catalog.get_meta(aid)
        q = _query_for(meta)
        if not q:
            continue
        enc = embedder.encode(text=q, image=None, sparse_text=q)
        points = store.hybrid_search(
            text_dense=enc.get("text_dense"),
            image_dense=None,
            sparse_indices=enc.get("sparse_indices", []),
            sparse_values=enc.get("sparse_values", []),
            limit=TOPK,
        ) or []
        ids = [normalize_article_id(str((getattr(p, "payload", {}) or {}).get("article_id", ""))) for p in points]
        ranks.append(ids.index(aid) + 1 if aid in ids else 10**9)

    m = _metrics(ranks)
    print(f"self-retrieval eval | sample={len(ranks)} topk={TOPK} seed={SEED}")
    print(" ", "  ".join(f"{k}={v}" for k, v in m.items()))
    return m


if __name__ == "__main__":
    main()
