"""Experiment: can P3alpha transaction edges improve the compat metric?

Augment M2 training positives with P3alpha edges that touch a co-buy-cold node (the only
nodes that gain NEW supervision — warm nodes already have co-buy edges). These edges are
noisier as labels (half the same-basket precision on the temporal harness) so they are
down-sampled by FRACTION. Everything else identical to train_compat_metric.

Hypothesis (md/refine_5/6): near-neutral, because the cold nodes P3alpha can supervise are
already served better by the direct P3alpha edge at serve time, and the truly-cold
(no-transaction) nodes get no P3alpha edge here either. Run to settle it with a number.

Run: python -m src.scripts.compat.train_compat_aug
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from src.scripts.compat.eval_pairing import PairingEvaluator
from src.scripts.compat.train_compat_metric import DATA_DIR, CompatTrainer

FRACTION = float(os.getenv("P3A_FRACTION", "0.3"))
SEED = 42


class AugTrainer(CompatTrainer):
    def __init__(self, device=None):
        super().__init__(device=device)
        ids = json.load(open(os.path.join(DATA_DIR, "node_ids.json"), encoding="utf-8"))
        aid_to_idx = {k: int(v) for k, v in ids["aid_to_idx"].items()}
        warm = set()
        for a, b in self.train_edges.tolist():
            warm.add(a); warm.add(b)

        p3 = pd.read_csv("data/processed/p3a_cold_edges.csv")
        for c in ("item_a", "item_b"):
            p3[c] = p3[c].astype(str).str.zfill(10)
        extra = []
        for a, b in zip(p3.item_a, p3.item_b):
            ia, ib = aid_to_idx.get(a), aid_to_idx.get(b)
            if ia is None or ib is None:
                continue
            if ia not in warm or ib not in warm:   # must inject new (cold) supervision
                extra.append((ia, ib))
        extra = np.asarray(extra, dtype=np.int64)
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(extra), size=int(len(extra) * FRACTION), replace=False)
        extra = torch.from_numpy(extra[keep]).long()
        print(f"same-basket edges={self.train_edges.shape[0]} | p3a-cold added={extra.shape[0]} "
              f"(frac={FRACTION})", flush=True)
        self.train_edges = torch.cat([self.train_edges, extra], dim=0)

    @torch.no_grad()
    def export(self, path):
        self.model.eval()
        out = [self.model(self.X[s:s + 4096]).cpu().numpy() for s in range(0, self.num_nodes, 4096)]
        emb = np.vstack(out).astype(np.float32)
        np.save(path, emb)
        return emb


def main():
    tr = AugTrainer()
    tr.train()
    emb_aug = tr.export(os.path.join(DATA_DIR, "compat_emb_aug.npy"))
    emb_old = np.load(os.path.join(DATA_DIR, "compat_emb.npy"))
    for mode in ("hard_random", "pop_matched"):
        rep = PairingEvaluator(extra_methods={"compat": emb_old, "compat_aug": emb_aug}, neg_mode=mode).evaluate()
        print(f"\n===== {mode} =====")
        for g in ("warm", "cold"):
            print(f"[{g}]")
            for m in ("compat", "compat_aug"):
                v = rep[g][m]
                print(f"  {m:11s} AUC={v['auc']:.4f} MRR={v['mrr']:.4f} H@1={v['hit@1']:.3f} H@10={v['hit@10']:.3f}")


if __name__ == "__main__":
    main()
