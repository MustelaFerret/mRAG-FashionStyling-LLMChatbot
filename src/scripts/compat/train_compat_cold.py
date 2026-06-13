"""Experiment B — cold-robust retrain of the compat metric head.

Hypothesis for the warm/cold gap (AUC 0.984 vs 0.915): uniform edge sampling
overrepresents popular anchors, so the embedding space is tuned to dense (warm) feature
regions; cold items live in underrepresented regions. Two changes vs train_compat_metric:

  1. degree-debiased positive sampling: edge (a,b) drawn with prob ~ 1/sqrt(deg a * deg b)
  2. input feature augmentation: gaussian noise + light same-slot mixup on the anchor,
     simulating unseen items near the training manifold

Everything else (architecture, InfoNCE, negatives, split, seed) identical. Output
compat_emb_cold.npy — only replaces compat_emb.npy if it WINS on the eval harness.

Run: python -m src.scripts.compat.train_compat_cold
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.scripts.compat.eval_pairing import PairingEvaluator
from src.scripts.compat.train_compat_metric import BATCH, DATA_DIR, EPOCHS, TEMP, CompatTrainer

NOISE_STD = 0.03   # relative to per-dim feature std
MIXUP_P = 0.25     # fraction of anchors mixed with a same-slot random item
MIXUP_LAM = 0.8    # anchor keeps 80% of itself


class ColdRobustTrainer(CompatTrainer):
    def __init__(self, device: str | None = None):
        super().__init__(device=device)
        deg = torch.zeros(self.num_nodes)
        for a, b in self.train_edges.tolist():
            deg[a] += 1
            deg[b] += 1
        e = self.train_edges
        w = 1.0 / torch.sqrt(deg[e[:, 0]].clamp(min=1) * deg[e[:, 1]].clamp(min=1))
        self.edge_weights = w / w.sum()  # CPU, like train_edges (multinomial index must match)
        self.feat_std = self.X.float().std(dim=0, keepdim=True)

    def _augment(self, feats: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        out = feats + torch.randn_like(feats) * (NOISE_STD * self.feat_std)
        mix = torch.rand(feats.shape[0], device=self.device) < MIXUP_P
        if mix.any():
            partners = self._sample_from(self.same_mat, self.same_len, slot_ids[mix], 1).squeeze(1)
            out[mix] = MIXUP_LAM * out[mix] + (1 - MIXUP_LAM) * self.X[partners]
        return out

    def train(self) -> None:
        self.model.train()
        n_edges = self.train_edges.shape[0]
        for epoch in range(1, EPOCHS + 1):
            total, batches = 0.0, 0
            order = torch.multinomial(self.edge_weights, n_edges, replacement=True)
            for start in range(0, n_edges, BATCH):
                idx = order[start:start + BATCH]
                edge = self.train_edges[idx].to(self.device)
                flip = torch.rand(edge.shape[0], device=self.device) < 0.5
                a = torch.where(flip, edge[:, 1], edge[:, 0])
                p = torch.where(flip, edge[:, 0], edge[:, 1])
                neg = self._sample_negatives(a)

                fa = self.model(self._augment(self.X[a], self.slot_of[a]))
                fp = self.model(self.X[p])
                fn = self.model(self.X[neg.reshape(-1)]).reshape(neg.shape[0], neg.shape[1], -1)
                s_pos = (fa * fp).sum(-1, keepdim=True)
                s_neg = (fa.unsqueeze(1) * fn).sum(-1)
                logits = torch.cat([s_pos, s_neg], dim=1) / TEMP
                target = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
                loss = F.cross_entropy(logits, target)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total += float(loss)
                batches += 1
            print(f"  epoch {epoch:02d}/{EPOCHS} loss={total / max(1, batches):.4f}", flush=True)

    @torch.no_grad()
    def export(self, path: str) -> np.ndarray:
        self.model.eval()
        out = []
        for s in range(0, self.num_nodes, 4096):
            out.append(self.model(self.X[s:s + 4096]).cpu().numpy())
        emb = np.vstack(out).astype(np.float32)
        np.save(path, emb)
        return emb


def main() -> None:
    trainer = ColdRobustTrainer()
    trainer.train()
    out_path = os.path.join(DATA_DIR, "compat_emb_cold.npy")
    emb_cold = trainer.export(out_path)
    print(f"saved -> {out_path}")

    emb_old = np.load(os.path.join(DATA_DIR, "compat_emb.npy"))
    for mode in ("hard_random", "pop_matched"):
        report = PairingEvaluator(
            extra_methods={"compat": emb_old, "compat_cold": emb_cold}, neg_mode=mode
        ).evaluate()
        with open(os.path.join(DATA_DIR, f"eval_compat_cold_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump({"neg_mode": mode, "report": report}, f, indent=2)
        print(f"\n===== neg_mode={mode} =====")
        for group in ("warm", "cold"):
            print(f"[{group}]")
            for m in ("compat", "compat_cold"):
                v = report[group][m]
                print(f"  {m:12s} AUC={v['auc']:.4f} MRR={v['mrr']:.4f} H@1={v['hit@1']:.3f} H@10={v['hit@10']:.3f} (n={v['n']})")


if __name__ == "__main__":
    main()
