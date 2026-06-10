"""Gap C (audit_metadata) backfill: add the usable-but-absent attribute fields to the
payload of every existing point WITHOUT re-encoding vectors -- graphical_appearance
(pattern), dominant_material, colour_value (shade), colour_master (colour family). Also
create keyword payload indexes for the filterable ones (pattern, material, colour_master)
so they can back hard/soft NLU filters.

A full build_vector_db run already writes these (mapping updated); this script patches the
live index in place. Idempotent.

Run alone (embedded Qdrant single-process):
    PYTORCH_JIT=0 python -m src.scripts.indexing.backfill_payload [--limit N]
"""
from __future__ import annotations

import argparse
from typing import List

import pandas as pd
from qdrant_client import models

from src.backend.core.config import settings
from src.backend.core.utils import normalize_article_id, parse_numeric_ids
from src.backend.retrieval.qdrant import QdrantStore
from src.scripts.indexing.build_vector_db import VectorIndexBuilder

BATCH = 256
FIELD_MAP = {
    "graphical_appearance": "graphical_appearance_name",
    "dominant_material": "dominant_material",
    "colour_value": "perceived_colour_value_name",
    "colour_master": "perceived_colour_master_name",
}
INDEX_FIELDS = ["graphical_appearance", "dominant_material", "colour_master"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="patch only first N points (0 = all)")
    args = ap.parse_args()

    builder = VectorIndexBuilder.__new__(VectorIndexBuilder)
    df = pd.read_csv(settings.meta_file).fillna("")
    df["aid_norm"] = df["article_id"].map(lambda x: normalize_article_id(str(x)))
    if args.limit > 0:
        df = df.head(args.limit)

    store = QdrantStore(settings.db_path, settings.collection_name)

    for field in INDEX_FIELDS:
        try:
            store.client.create_payload_index(
                collection_name=settings.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:  # already exists / race -> non-fatal
            print(f"  index {field}: {type(exc).__name__} (likely exists)")

    ops: List[models.UpdateOperation] = []
    patched = 0
    for row in df.itertuples(index=False):
        pids = parse_numeric_ids([getattr(row, "aid_norm")])
        if not pids:
            continue
        payload = {pk: builder.clean_value(getattr(row, src)) for pk, src in FIELD_MAP.items()}
        ops.append(models.SetPayloadOperation(
            set_payload=models.SetPayload(payload=payload, points=[pids[0]])
        ))
        if len(ops) >= BATCH:
            store.client.batch_update_points(collection_name=settings.collection_name, update_operations=ops)
            patched += len(ops)
            ops = []
            print(f"  patched {patched}", end="\r")
    if ops:
        store.client.batch_update_points(collection_name=settings.collection_name, update_operations=ops)
        patched += len(ops)
    print(f"\n[payload] backfilled {patched} points with {list(FIELD_MAP)} (+indexes {INDEX_FIELDS})")


if __name__ == "__main__":
    main()
