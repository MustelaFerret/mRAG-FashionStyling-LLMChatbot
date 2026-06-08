"""Qualitative check — compat_emb vs SigLIP-kNN pairings for given anchors.

For each anchor: top-k by compat_emb (cross-slot complement only, via ALLOWED_SLOT_PAIRS)
vs top-k by raw SigLIP. Shows whether the learned metric returns plausible outfit
partners (esp. for cold / out-of-graph items the co-buy holdout cannot measure).

Run: python -m src.scripts.compat.inspect_pairing 0569973001 0569973006 ...
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from src.backend.core.config import settings
from src.backend.core.utils import normalize_article_id
from src.backend.services.catalog import FashionCatalog
from src.scripts.graph.outfit_slots import slot_pair_allowed

DATA_DIR = os.path.join(os.path.dirname(settings.meta_file), "compat")
TOPK = 6


def load():
    compat = np.load(os.path.join(DATA_DIR, "compat_emb.npy"))
    siglip = np.load(os.path.join(DATA_DIR, "node_features.npy"))
    ids = json.load(open(os.path.join(DATA_DIR, "node_ids.json"), encoding="utf-8"))
    slots = json.load(open(os.path.join(DATA_DIR, "node_slots.json"), encoding="utf-8"))
    return compat, siglip, ids["article_ids"], ids["aid_to_idx"], slots


def top_compat(emb, idx, slots, complement_only, k):
    sims = emb @ emb[idx]
    order = np.argsort(-sims)
    anchor_slot = slots[idx]
    out = []
    for j in order:
        if j == idx:
            continue
        if complement_only and not slot_pair_allowed(anchor_slot, slots[j]):
            continue
        out.append((int(j), float(sims[j])))
        if len(out) >= k:
            break
    return out


def main():
    anchors = [normalize_article_id(a) for a in sys.argv[1:]] or ["0569973001"]
    compat, siglip, article_ids, aid_to_idx, slots = load()
    catalog = FashionCatalog(settings.meta_file, settings.graph_file, settings.image_dir)

    def describe(j):
        aid = article_ids[j]
        m = catalog.get_meta(aid)
        return f"{aid} | slot={slots[j]:9s} | {m.get('product_type_name','?'):14s} | {m.get('colour_group_name','?')}"

    for a in anchors:
        idx = aid_to_idx.get(a)
        if idx is None:
            print(f"\n### {a} — NOT in index"); continue
        in_graph = a in catalog.graph_adj
        print(f"\n### anchor {a} | slot={slots[idx]} | {catalog.get_meta(a).get('product_type_name','?')} {catalog.get_meta(a).get('colour_group_name','')} | in_graph={in_graph}")
        print("  -- compat (complement slots only) --")
        for j, s in top_compat(compat, idx, slots, True, TOPK):
            print(f"    {s:.3f}  {describe(j)}")
        print("  -- siglip-kNN (raw, any slot) --")
        for j, s in top_compat(siglip, idx, slots, False, TOPK):
            print(f"    {s:.3f}  {describe(j)}")


if __name__ == "__main__":
    main()
