"""GraphSAGE on the AUGMENTED graph: co-buy train edges + P3alpha-cold edges, in BOTH the
aggregation adjacency AND the positive set. M3 GraphSAGE lost to the MLP on cold only
because cold nodes had no neighbors to aggregate (it degenerated to the content MLP). With
P3alpha edges, cold nodes now have a neighborhood -> test whether propagation finally helps.

Compares against the shipped compat_emb.npy (P3alpha-aug MLP). Run:
  python -m src.scripts.compat.train_compat_graphsage_aug
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from src.scripts.compat.eval_pairing import PairingEvaluator
from src.scripts.compat.train_compat_graphsage import DATA_DIR, GraphSAGETrainer

FRACTION = float(os.getenv("P3A_FRACTION", "0.3"))
SEED = 42


def _p3a_cold_edge_index() -> np.ndarray:
    ids = json.load(open(os.path.join(DATA_DIR, "node_ids.json"), encoding="utf-8"))
    aid_to_idx = {k: int(v) for k, v in ids["aid_to_idx"].items()}
    edges = np.load(os.path.join(DATA_DIR, "edges.npz"))
    ei_train = edges["edge_index"][:, edges["train_idx"]]
    warm = set(ei_train.flatten().tolist())
    p3 = pd.read_csv("data/processed/p3a_cold_edges.csv")
    for c in ("item_a", "item_b"):
        p3[c] = p3[c].astype(str).str.zfill(10)
    pairs = []
    for a, b in zip(p3.item_a, p3.item_b):
        ia, ib = aid_to_idx.get(a), aid_to_idx.get(b)
        if ia is None or ib is None:
            continue
        if ia not in warm or ib not in warm:
            pairs.append((ia, ib))
    pairs = np.asarray(pairs, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    keep = rng.choice(len(pairs), size=int(len(pairs) * FRACTION), replace=False)
    return pairs[keep].T  # (2, E)


class GraphSAGEAugTrainer(GraphSAGETrainer):
    def __init__(self, device=None):
        super().__init__(device=device)
        p3 = _p3a_cold_edge_index()
        edges = np.load(os.path.join(DATA_DIR, "edges.npz"))
        ei_train = edges["edge_index"][:, edges["train_idx"]]
        aug = np.concatenate([ei_train, p3], axis=1)
        # rebuild adjacency over the augmented edge set (cold nodes now get neighbors)
        self._build_adjacency(aug)
        # add P3alpha-cold edges to the positive training set
        extra = torch.from_numpy(p3.T.copy()).long()
        self.train_edges = torch.cat([self.train_edges, extra], dim=0)
        print(f"[graphsage-aug] train edges={self.train_edges.shape[0]} (p3a-cold +{extra.shape[0]})", flush=True)


def main():
    tr = GraphSAGEAugTrainer()
    tr.train()
    emb = tr.embed_all()
    np.save(os.path.join(DATA_DIR, "compat_emb_graphsage_aug.npy"), emb)
    ship = np.load(os.path.join(DATA_DIR, "compat_emb.npy"))  # P3alpha-aug MLP (current ship)
    for mode in ("hard_random", "pop_matched"):
        rep = PairingEvaluator(extra_methods={"compat_aug": ship, "sage_aug": emb}, neg_mode=mode).evaluate()
        print(f"\n===== {mode} =====")
        for g in ("warm", "cold"):
            print(f"[{g}]")
            for m in ("compat_aug", "sage_aug"):
                v = rep[g][m]
                print(f"  {m:11s} AUC={v['auc']:.4f} MRR={v['mrr']:.4f} H@1={v['hit@1']:.3f} H@10={v['hit@10']:.3f}")


if __name__ == "__main__":
    main()
