"""Smoke test retrieval sau khi rebuild vector DB.

10 query đa dạng (text-only natural language, structured, code-switching, image).
Print top-5 với article_id + prod_name + PT + color.

Mục đích: verify hybrid retrieval hoạt động + sparse vocab mới cover natural query.

Run:
    conda activate mRAG
    python -m src.scripts.smoke_test_retrieval
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import List

from src.backend.core.config import settings
from src.backend.retrieval.embeddings import SparseTfidfEncoder
from src.backend.retrieval.encoders import QueryEncoder, SigLIPEncoder
from src.backend.retrieval.qdrant import QdrantStore


QUERIES = [
    # Structured (metadata-driven) — sparse + dense both work
    "black blazer office",
    "white t-shirt casual",
    # Natural language — dense must carry, sparse should also hit some
    "elegant flowy summer dress for beach vacation",
    "chunky oversized knit sweater cozy winter vibe",
    "y2k aesthetic vintage graphic tee",
    "minimalist scandinavian linen jumpsuit",
    "streetwear baggy cargo pants techwear",
    # Cross-PT outfit query
    "red floral midi dress with denim jacket",
    # Color compound + style
    "burnt orange linen wide-leg trousers",
    # Brand/specific (rare token)
    "denim slim ankle jeans high waist",
]


def print_results(query: str, points: List):
    print(f"\n>>> {query!r}")
    if not points:
        print("  (no results)")
        return
    for rank, p in enumerate(points, 1):
        payload = getattr(p, "payload", {}) or {}
        score = getattr(p, "score", 0.0)
        aid = payload.get("article_id", "")
        prod_name = (payload.get("prod_name", "") or "")[:35]
        pt = payload.get("product_type", "")
        color = payload.get("colour_group", "")
        sect = payload.get("section_name", "")[:20]
        print(f"  #{rank} score={score:.4f} #{aid}  {prod_name:35s}  PT={pt:18s}  color={color:14s}  {sect}")


def main() -> None:
    sparse_path = settings.sparse_model_path
    if not os.path.exists(sparse_path):
        raise SystemExit(f"sparse encoder not found at {sparse_path} — run build_vector_db.py first")
    sparse = SparseTfidfEncoder.load(sparse_path)
    print(f"sparse encoder loaded: vocab={len(sparse.vocab):,}")

    siglip = SigLIPEncoder()
    print(f"siglip loaded on {siglip.device}, dtype={siglip.dtype}, embed_dim={siglip.embed_dim}")

    qe = QueryEncoder(siglip=siglip, sparse=sparse)
    store = QdrantStore(settings.db_path, settings.collection_name)

    try:
        info = store.client.get_collection(settings.collection_name)
        print(f"collection '{settings.collection_name}': points={info.points_count}, vectors_config keys={list(info.config.params.vectors.keys()) if hasattr(info.config.params.vectors, 'keys') else 'n/a'}")
    except Exception as ex:
        raise SystemExit(f"collection not found — run build_vector_db.py first: {ex}")

    print(f"\nstart loop over {len(QUERIES)} queries", flush=True)
    for i, query in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] encoding query {query!r}", flush=True)
        try:
            encoded = qe.encode(text=query, image=None)
            td = encoded.get("text_dense")
            sd_i = encoded.get("sparse_indices") or []
            print(f"  text_dense: len={len(td) if td else 0}, sparse_idx: len={len(sd_i)}", flush=True)
            print(f"  calling hybrid_search...", flush=True)
            points = store.hybrid_search(
                text_dense=td,
                image_dense=encoded.get("image_dense"),
                sparse_indices=sd_i,
                sparse_values=encoded.get("sparse_values"),
                limit=5,
            )
            print(f"  -> got {len(points) if points else 0} points", flush=True)
        except Exception as ex:
            print(f"  ERROR: {type(ex).__name__}: {ex}", flush=True)
            traceback.print_exc()
            continue
        print_results(query, points)
    print("\nloop done", flush=True)


if __name__ == "__main__":
    main()
