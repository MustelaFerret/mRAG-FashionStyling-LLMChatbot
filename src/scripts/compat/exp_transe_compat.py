"""Experiment ③: translational compatibility (TransE per type-pair) vs the type-aware winner.

Knowledge-graph embedding view of "goes-with": model it as a type-pair-conditioned TRANSLATION
    score(a,b) = -|| g(a) + r_{slot_a -> slot_b} - g(b) ||
so each ordered category pair has its own relation vector. This is the unusual-but-grounded angle
(relational, directional) vs the metric/masked-cosine models. Trained on the same co-buy edges +
InfoNCE + same/complement/random negatives, then scored on the same held-out link-prediction eval
against the shipped single-space `compat` and the type-aware model (loaded from its saved artifacts).

Run: python -m src.scripts.compat.exp_transe_compat
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.scripts.compat.eval_pairing import PairingEvaluator
from src.scripts.compat.exp_typeaware_disentangled import (
    DATA_DIR, EMB_DIM, HIDDEN, DROPOUT, EPOCHS, BATCH, LR, SEED,
    _load_common, _NegSampler,
)

TEMP = 0.5


class TransEHead(nn.Module):
    def __init__(self, in_dim, n_slot):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                                 nn.Linear(HIDDEN, EMB_DIM))
        self.rel = nn.Parameter(torch.zeros(n_slot, n_slot, EMB_DIM))  # ordered type-pair relation

    def g(self, x):
        return F.normalize(self.net(x), dim=-1)  # unit-norm entities (anti-collapse, standard TransE)

    def score(self, ga, gx, sa, sx):
        r = self.rel[sa, sx]
        return -((ga + r - gx) ** 2).sum(-1)  # negative squared L2 (higher = more compatible)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, slots, slot_of, n_slot, pair_of, n_pairs, train_edges, cat_ids, cat_sizes, article_ids = _load_common(device)
    neg = _NegSampler(slots, slot_of, device)
    print(f"[transe] device={device} nodes={X.shape[0]} edges={train_edges.shape[0]} slots={n_slot}")

    torch.manual_seed(SEED)
    model = TransEHead(X.shape[1], n_slot).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(train_edges.shape[0]); total = 0.0
        for start in range(0, perm.shape[0], BATCH):
            idx = perm[start:start + BATCH]; edge = train_edges[idx].to(device)
            flip = torch.rand(edge.shape[0], device=device) < 0.5
            a = torch.where(flip, edge[:, 1], edge[:, 0]); p = torch.where(flip, edge[:, 0], edge[:, 1])
            ng = neg.sample(a)
            ga, gp = model.g(X[a]), model.g(X[p])
            gn = model.g(X[ng.reshape(-1)]).reshape(ng.shape[0], ng.shape[1], -1)
            sa, sp = slot_of[a], slot_of[p]
            s_pos = model.score(ga, gp, sa, sp)
            sn = slot_of[ng]
            s_neg = model.score(ga.unsqueeze(1), gn, sa.unsqueeze(1).expand_as(sn), sn)
            logits = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1) / TEMP
            loss = F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=device))
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * idx.shape[0]
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [transe] epoch {epoch:02d} loss={total / perm.shape[0]:.4f}")

    model.eval()
    with torch.no_grad():
        G = torch.cat([model.g(X[s:s + 8192]) for s in range(0, X.shape[0], 8192)]).cpu().numpy().astype(np.float32)
        R = model.rel.detach().cpu().numpy().astype(np.float32)
    slot_np = slot_of.cpu().numpy()

    def score_transe(anchor, cands):
        r = R[slot_np[anchor], slot_np[cands]]
        return -((G[anchor] + r - G[cands]) ** 2).sum(axis=1)

    # type-aware scorer from saved artifacts (the shipped winner) for head-to-head
    ta_g = np.load(os.path.join(DATA_DIR, "compat_typeaware_g.npy"))
    ta_mask = np.load(os.path.join(DATA_DIR, "compat_typeaware_mask.npy"))
    ta_meta = json.load(open(os.path.join(DATA_DIR, "compat_typeaware_meta.json"), encoding="utf-8"))
    ta_pair = np.asarray(ta_meta["pair_of"], dtype=np.int64)
    svidx = {s: i for i, s in enumerate(ta_meta["slot_vocab"])}
    ta_slot = np.asarray([svidx.get(s, 0) for s in slots], dtype=np.int64)

    def score_typeaware(anchor, cands):
        m = ta_mask[ta_pair[ta_slot[anchor], ta_slot[cands]]]
        return np.sum(ta_g[anchor] * ta_g[cands] * m, axis=1)

    compat = np.load(os.path.join(DATA_DIR, "compat_emb.npy"))
    extra = {"compat": compat, "typeaware": score_typeaware, "transe": score_transe}
    summary = {}
    for mode in ("hard_random", "pop_matched"):
        report = PairingEvaluator(extra_methods=extra, neg_mode=mode).evaluate()
        summary[mode] = report
        print(f"\n===== neg_mode={mode} =====")
        for group in ("warm", "cold"):
            print(f"[{group}]")
            for m in ("compat", "typeaware", "transe"):
                v = report[group][m]
                print(f"  {m:11s} AUC={v['auc']:.3f} MRR={v['mrr']:.3f} H@1={v['hit@1']:.3f} H@10={v['hit@10']:.3f}")
    with open(os.path.join(DATA_DIR, "exp_transe.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote exp_transe.json")


if __name__ == "__main__":
    main()
