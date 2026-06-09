"""A2 build — fit a BM25 sparse encoder on the rich-text corpus, save it, and re-index
ONLY the sparse vector of every point in Qdrant (dense vectors untouched).

Keeps the old TF-IDF model file for revert. Run a small batch first with --limit.

Run: python -m src.scripts.indexing.build_bm25 [--limit N]
"""
from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
from qdrant_client import models

from src.backend.core.config import settings
from src.backend.core.utils import normalize_article_id, parse_numeric_ids
from src.backend.retrieval.embeddings import SparseBM25Encoder
from src.backend.retrieval.qdrant import QdrantStore
from src.scripts.indexing.build_vector_db import VectorIndexBuilder

BATCH = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="re-index only first N points (0 = all)")
    args = ap.parse_args()

    builder = VectorIndexBuilder.__new__(VectorIndexBuilder)
    df = pd.read_csv(settings.meta_file).fillna("")
    df["aid_norm"] = df["article_id"].map(lambda x: normalize_article_id(str(x)))
    df["rich_text"] = df.apply(builder.build_sparse_text, axis=1)

    encoder = SparseBM25Encoder(min_df=2, max_df_ratio=0.95)
    encoder.fit(df["rich_text"].tolist())
    encoder.save(settings.sparse_model_path)
    print(f"[bm25] fitted vocab={len(encoder.vocab)} avgdl={encoder.avgdl:.2f} -> {settings.sparse_model_path}")

    store = QdrantStore(settings.db_path, settings.collection_name)
    sub = df if args.limit <= 0 else df.head(args.limit)

    batch: List[models.PointVectors] = []
    updated = 0
    for row in sub.itertuples(index=False):
        aid = getattr(row, "aid_norm")
        pids = parse_numeric_ids([aid])
        if not pids:
            continue
        idx, val = encoder.encode_doc(getattr(row, "rich_text"))
        batch.append(models.PointVectors(
            id=pids[0],
            vector={settings.vector_name_sparse: models.SparseVector(indices=idx, values=val)},
        ))
        if len(batch) >= BATCH:
            store.client.update_vectors(collection_name=settings.collection_name, points=batch)
            updated += len(batch)
            batch = []
            print(f"  updated {updated}", end="\r")
    if batch:
        store.client.update_vectors(collection_name=settings.collection_name, points=batch)
        updated += len(batch)
    print(f"\n[bm25] re-indexed sparse vectors for {updated} points")


if __name__ == "__main__":
    main()
