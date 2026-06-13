"""compat with RICH features (image⊕text, 1536d) + P3alpha-cold augmentation.
Reuses AugTrainer (which adds the P3alpha-cold positives), only swaps the input feature
matrix and rebuilds the head for the new input dim. Eval vs the shipped compat.

Run: python -m src.scripts.compat.train_compat_rich
"""
from __future__ import annotations

import os

import numpy as np
import torch

from src.scripts.compat.eval_pairing import PairingEvaluator
from src.scripts.compat.train_compat_aug import AugTrainer
from src.scripts.compat.train_compat_metric import DROPOUT, EMB_DIM, HIDDEN, LR, SEED, DATA_DIR, MetricHead


class RichAugTrainer(AugTrainer):
    def __init__(self, device=None):
        super().__init__(device=device)  # image feats + edges (incl P3alpha-cold)
        rich = np.load(os.path.join(DATA_DIR, "node_features_rich.npy"))
        self.X = torch.from_numpy(rich).to(self.device)
        torch.manual_seed(SEED)
        self.model = MetricHead(self.X.shape[1], HIDDEN, EMB_DIM, DROPOUT).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        print(f"rich feature dim={self.X.shape[1]}", flush=True)

    @torch.no_grad()
    def export(self, path):
        self.model.eval()
        out = [self.model(self.X[s:s + 4096]).cpu().numpy() for s in range(0, self.num_nodes, 4096)]
        emb = np.vstack(out).astype(np.float32)
        np.save(path, emb)
        return emb


def main():
    tr = RichAugTrainer()
    tr.train()
    emb_rich = tr.export(os.path.join(DATA_DIR, "compat_emb_rich.npy"))
    emb_ship = np.load(os.path.join(DATA_DIR, "compat_emb.npy"))  # current = P3alpha-aug
    for mode in ("hard_random", "pop_matched"):
        rep = PairingEvaluator(extra_methods={"compat_aug": emb_ship, "compat_rich": emb_rich}, neg_mode=mode).evaluate()
        print(f"\n===== {mode} =====")
        for g in ("warm", "cold"):
            print(f"[{g}]")
            for m in ("compat_aug", "compat_rich"):
                v = rep[g][m]
                print(f"  {m:12s} AUC={v['auc']:.4f} MRR={v['mrr']:.4f} H@1={v['hit@1']:.3f} H@10={v['hit@10']:.3f}")


if __name__ == "__main__":
    main()
