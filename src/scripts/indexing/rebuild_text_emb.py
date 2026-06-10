"""Re-encode ONLY the SigLIP text_emb vector of every point with the new front-loaded
build_dense_text (audit_metadata bug A: attributes were truncated out of the 64-token
window for 71% of items). Image + sparse vectors untouched.

No on-disk backup of the old dense vectors -- to revert, restore build_dense_text and
re-run this script.

Run alone (embedded Qdrant single-process):
    PYTORCH_JIT=0 python -m src.scripts.indexing.rebuild_text_emb [--limit N]
"""
from __future__ import annotations

import argparse
from typing import List

import pandas as pd
from qdrant_client import models

from src.backend.core.config import settings
from src.backend.core.utils import normalize_article_id, parse_numeric_ids
from src.backend.retrieval.encoders import SigLIPEncoder
from src.backend.retrieval.qdrant import QdrantStore
from src.scripts.indexing.build_vector_db import VectorIndexBuilder

BATCH = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="re-encode only first N points (0 = all)")
    args = ap.parse_args()

    builder = VectorIndexBuilder.__new__(VectorIndexBuilder)
    df = pd.read_csv(settings.meta_file).fillna("")
    df["aid_norm"] = df["article_id"].map(lambda x: normalize_article_id(str(x)))
    df["dense_text"] = df.apply(builder.build_dense_text, axis=1)
    if args.limit > 0:
        df = df.head(args.limit)

    siglip = SigLIPEncoder()
    embs = siglip.encode_texts(
        df["dense_text"].tolist(),
        batch_size=settings.encode_batch_text,
        progress_desc="text_emb",
    )
    siglip.free()

    store = QdrantStore(settings.db_path, settings.collection_name)
    batch: List[models.PointVectors] = []
    updated = 0
    for pos, row in enumerate(df.itertuples(index=False)):
        pids = parse_numeric_ids([getattr(row, "aid_norm")])
        if not pids:
            continue
        batch.append(models.PointVectors(
            id=pids[0],
            vector={settings.vector_name_text: embs[pos].tolist()},
        ))
        if len(batch) >= BATCH:
            store.client.update_vectors(collection_name=settings.collection_name, points=batch)
            updated += len(batch)
            batch = []
            print(f"  updated {updated}", end="\r")
    if batch:
        store.client.update_vectors(collection_name=settings.collection_name, points=batch)
        updated += len(batch)
    print(f"\n[text_emb] re-encoded + re-indexed {updated} points")


if __name__ == "__main__":
    main()
