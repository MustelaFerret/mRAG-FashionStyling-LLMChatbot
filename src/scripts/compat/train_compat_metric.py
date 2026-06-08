"""Milestone 2 — learned compatibility metric head on frozen SigLIP features.

A small MLP projects frozen SigLIP image_emb (768d) into a compatibility space
(128d, L2-normalised), trained with InfoNCE (temperature) on co-buy train edges.
Negatives mix three kinds so the metric learns item-level (not just slot-level)
compatibility and the score space does not collapse:
  - same-slot-as-anchor   : substitutes of the anchor (must be far)
  - complement-slot wrong  : plausible-role but non-partner item (item-level hard)
  - random
No graph aggregation (that is M3 / GraphSAGE). Output compat_emb is evaluated
against the M1 baselines on both negative-sampling modes.

Run: python -m src.scripts.compat.train_compat_metric
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backend.core.config import settings
from src.scripts.compat.eval_pairing import PairingEvaluator
from src.scripts.graph.outfit_slots import slot_pair_allowed

DATA_DIR = os.path.join(os.path.dirname(settings.meta_file), "compat")
EMB_DIM = 128
HIDDEN = 256
EPOCHS = 25
BATCH = 2048
LR = 1e-3
DROPOUT = 0.1
TEMP = 0.1
N_SUB = 3
N_COMP = 4
N_RAND = 3
SEED = 42


class MetricHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class CompatTrainer:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.X = torch.from_numpy(np.load(os.path.join(DATA_DIR, "node_features.npy"))).to(self.device)
        slots = json.load(open(os.path.join(DATA_DIR, "node_slots.json"), encoding="utf-8"))
        edges = np.load(os.path.join(DATA_DIR, "edges.npz"))
        self.train_edges = torch.from_numpy(edges["edge_index"][:, edges["train_idx"]].T.copy()).long()
        self.num_nodes = self.X.shape[0]
        self._build_slot_tensors(slots)
        torch.manual_seed(SEED)
        self.model = MetricHead(self.X.shape[1], HIDDEN, EMB_DIM, DROPOUT).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=LR)

    @staticmethod
    def _pad_buckets(buckets: Dict[int, List[int]], num_slots: int, device) -> tuple:
        max_len = max((len(v) for v in buckets.values()), default=1)
        mat = torch.zeros(num_slots, max_len, dtype=torch.long, device=device)
        lens = torch.zeros(num_slots, dtype=torch.long, device=device)
        for s, nodes in buckets.items():
            if nodes:
                mat[s, :len(nodes)] = torch.tensor(nodes, device=device)
                lens[s] = len(nodes)
        return mat, lens

    def _build_slot_tensors(self, slots: List[str]) -> None:
        slot_ids = sorted(set(slots))
        n = len(slot_ids)
        self.slot_of = torch.tensor([slot_ids.index(s) for s in slots], device=self.device)
        same: Dict[int, List[int]] = defaultdict(list)
        for i, s in enumerate(slots):
            same[slot_ids.index(s)].append(i)
        self.same_mat, self.same_len = self._pad_buckets(same, n, self.device)

        # complement bucket: for each slot, all nodes whose slot forms an allowed pair.
        comp: Dict[int, List[int]] = {sid: [] for sid in range(n)}
        for sid_a, sa in enumerate(slot_ids):
            comp_slots = {slot_ids.index(sb) for sb in slot_ids if slot_pair_allowed(sa, sb)}
            for i, s in enumerate(slots):
                if slot_ids.index(s) in comp_slots:
                    comp[sid_a].append(i)
        self.comp_mat, self.comp_len = self._pad_buckets(comp, n, self.device)

    def _sample_from(self, mat: torch.Tensor, lens: torch.Tensor, slot_ids: torch.Tensor, k: int) -> torch.Tensor:
        b = slot_ids.shape[0]
        length = lens[slot_ids].clamp(min=1).unsqueeze(1)
        rand = (torch.rand(b, k, device=self.device) * length).long()
        return mat[slot_ids.unsqueeze(1), rand]

    def _sample_negatives(self, anchor: torch.Tensor) -> torch.Tensor:
        a_slot = self.slot_of[anchor]
        negs = []
        if N_SUB > 0:
            negs.append(self._sample_from(self.same_mat, self.same_len, a_slot, N_SUB))
        if N_COMP > 0:
            negs.append(self._sample_from(self.comp_mat, self.comp_len, a_slot, N_COMP))
        if N_RAND > 0:
            negs.append(torch.randint(0, self.num_nodes, (anchor.shape[0], N_RAND), device=self.device))
        return torch.cat(negs, dim=1)

    def train(self) -> None:
        self.model.train()
        for epoch in range(1, EPOCHS + 1):
            perm = torch.randperm(self.train_edges.shape[0])
            total = 0.0
            for start in range(0, perm.shape[0], BATCH):
                idx = perm[start:start + BATCH]
                edge = self.train_edges[idx].to(self.device)
                flip = torch.rand(edge.shape[0], device=self.device) < 0.5
                a = torch.where(flip, edge[:, 1], edge[:, 0])
                p = torch.where(flip, edge[:, 0], edge[:, 1])
                neg = self._sample_negatives(a)

                fa = self.model(self.X[a])
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
                total += loss.item() * idx.shape[0]
            print(f"epoch {epoch:02d}  infonce_loss={total / perm.shape[0]:.4f}")

    @torch.no_grad()
    def embed_all(self) -> np.ndarray:
        self.model.eval()
        out = []
        for start in range(0, self.num_nodes, 8192):
            out.append(self.model(self.X[start:start + 8192]).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


def main() -> None:
    trainer = CompatTrainer()
    print(f"[compat-train] device={trainer.device} edges={trainer.train_edges.shape[0]} dim={EMB_DIM}")
    trainer.train()
    compat = trainer.embed_all()
    np.save(os.path.join(DATA_DIR, "compat_emb.npy"), compat)
    torch.save(trainer.model.state_dict(), os.path.join(DATA_DIR, "compat_head.pt"))

    summary = {}
    for mode in ("hard_random", "pop_matched"):
        report = PairingEvaluator(extra_methods={"compat": compat}, neg_mode=mode).evaluate()
        summary[mode] = report
        print(f"\n===== neg_mode={mode} =====")
        for group in ("warm", "cold"):
            print(f"[{group}]")
            for m, v in report[group].items():
                print(f"  {m:11s} AUC={v['auc']:.3f} MRR={v['mrr']:.3f} H@1={v['hit@1']:.3f} H@10={v['hit@10']:.3f}")
    with open(os.path.join(DATA_DIR, "eval_compat.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
