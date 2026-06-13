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
        self.features = None
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
        self.ready = (
            self.emb.shape[0] == len(self.article_ids) == len(self.slots)
            and bool(np.isfinite(self.emb).all())  # corrupt/NaN embeddings must not serve
        )
        # frozen SigLIP features for cold-anchor twin search (fp16, ~108MB)
        feat_path = os.path.join(compat_dir, "node_features.npy")
        if self.ready and os.path.exists(feat_path):
            feats = np.load(feat_path).astype(np.float16)
            if feats.shape[0] == len(self.article_ids):
                norms = np.linalg.norm(feats.astype(np.float32), axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.features = (feats.astype(np.float32) / norms).astype(np.float16)

    def nearest_warm_twins(self, anchor_id: str, warm_ids, k: int = 5):
        """Same-slot warm items most visually similar to the anchor (frozen SigLIP cosine).
        Used for cold-anchor neighbor borrowing: the twins' co-buy partners are served as
        the cold anchor's pairings (verified: cold MRR 0.459 -> 0.529, md/refine_4.MD)."""
        if not self.ready or self.features is None:
            return []
        idx = self.aid_to_idx.get(normalize_article_id(anchor_id))
        if idx is None:
            return []
        anchor_slot = self.slots[idx]
        pool = [i for i, (aid, s) in enumerate(zip(self.article_ids, self.slots))
                if s == anchor_slot and i != idx and aid in warm_ids]
        if not pool:
            return []
        pool_arr = np.asarray(pool, dtype=np.int64)
        sims = self.features[pool_arr].astype(np.float32) @ self.features[idx].astype(np.float32)
        top = pool_arr[np.argsort(-sims)[:k]]
        top_sims = self.features[top].astype(np.float32) @ self.features[idx].astype(np.float32)
        return [(self.article_ids[i], float(s)) for i, s in zip(top.tolist(), top_sims.tolist())]

    def complement_ids(self, anchor_id: str, limit: int, target_slot: str = "") -> List[str]:
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
            cand_slot = self.slots[j]
            if target_slot:
                # User named a category (e.g. "shoes") -> restrict to that slot.
                if cand_slot != target_slot:
                    continue
            elif not slot_pair_allowed(anchor_slot, cand_slot):
                continue
            out.append(self.article_ids[j])
            if len(out) >= limit:
                break
        return out
