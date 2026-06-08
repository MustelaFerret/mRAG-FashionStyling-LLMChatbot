from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from src.backend.core.utils import normalize_article_id
from src.scripts.graph.outfit_slots import slot_pair_allowed


class CompatPairingIndex:
    """In-memory learned-compatibility pairing index (M2 compat_emb).

    Loads the trained compat embeddings + slot labels and serves complement-slot
    nearest neighbours by cosine. Used as a fallback for graph_pairing when the
    co-buy graph has no usable neighbours (cold / out-of-graph anchors), so those
    items get learned pairings instead of an apology.
    """

    def __init__(self, compat_dir: str):
        self.ready = False
        emb_path = os.path.join(compat_dir, "compat_emb.npy")
        ids_path = os.path.join(compat_dir, "node_ids.json")
        slots_path = os.path.join(compat_dir, "node_slots.json")
        if not (os.path.exists(emb_path) and os.path.exists(ids_path) and os.path.exists(slots_path)):
            return
        self.emb = np.load(emb_path)
        ids = json.load(open(ids_path, encoding="utf-8"))
        self.article_ids = ids["article_ids"]
        self.aid_to_idx = {k: int(v) for k, v in ids["aid_to_idx"].items()}
        self.slots = json.load(open(slots_path, encoding="utf-8"))
        self.ready = self.emb.shape[0] == len(self.article_ids) == len(self.slots)

    def complement_ids(self, anchor_id: str, limit: int) -> List[str]:
        if not self.ready:
            return []
        idx = self.aid_to_idx.get(normalize_article_id(anchor_id))
        if idx is None:
            return []
        anchor_slot = self.slots[idx]
        sims = self.emb @ self.emb[idx]
        order = np.argsort(-sims)
        out: List[str] = []
        for j in order:
            j = int(j)
            if j == idx:
                continue
            if not slot_pair_allowed(anchor_slot, self.slots[j]):
                continue
            out.append(self.article_ids[j])
            if len(out) >= limit:
                break
        return out
