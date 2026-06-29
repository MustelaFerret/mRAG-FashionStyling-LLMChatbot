"""Experiment: two theoretically-grounded compatibility heads vs the M2 single-space metric.

(1) Type-conditioned (CSA-Net, Vasileva et al. ECCV'18): a general embedding g(x) plus a learnable
    diagonal mask m_{ab} per slot-pair, so compatibility is measured in a type-pair subspace:
        compat(a,b) = < g(a) ⊙ m_{ab}, g(b) ⊙ m_{ab} >.
    Fixes the single-space metric's blind spot (top-bottom, top-shoe, bag-dress geometry differ).

(2) Disentangled aspect-compatibility (data-driven, interpretable -- NOT hand colour rules): a learned
    affinity per attribute aspect (colour, occasion, season, pattern) via small category embeddings,
    plus a learned SigLIP residual, combined with learned non-negative weights:
        compat(a,b) = Σ_aspect w_aspect · <E_aspect[c_a], E_aspect[c_b]> + w_resid · cos(g(a), g(b)).
    Every aspect affinity is learned from co-buy (which colour/occasion/season pairs actually co-occur),
    so the model stays interpretable (read the weights + the colour-pair matrix) without hard rules.

Both are trained with the same InfoNCE + same/complement/random negatives as train_compat_metric, then
scored on the held-out co-buy link-prediction eval (PairingEvaluator) against the shipped `compat`.

Run: python -m src.scripts.compat.exp_typeaware_disentangled
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backend.core.config import settings
from src.scripts.compat.eval_pairing import PairingEvaluator
from src.scripts.graph.outfit_slots import slot_pair_allowed

DATA_DIR = os.path.join(os.path.dirname(settings.meta_file), "compat")
EMB_DIM = 128
HIDDEN = 256
ASPECT_DIM = 24
EPOCHS = 25
BATCH = 2048
LR = 1e-3
DROPOUT = 0.1
TEMP = 0.1
N_SUB, N_COMP, N_RAND = 3, 4, 3
SEED = 42
ASPECTS = ["colour_group_name", "occasion", "seasonality", "graphical_appearance_name"]


def _load_common(device):
    X = torch.from_numpy(np.load(os.path.join(DATA_DIR, "node_features.npy"))).to(device)
    slots = json.load(open(os.path.join(DATA_DIR, "node_slots.json"), encoding="utf-8"))
    ids = json.load(open(os.path.join(DATA_DIR, "node_ids.json"), encoding="utf-8"))
    article_ids = ids["article_ids"]
    edges = np.load(os.path.join(DATA_DIR, "edges.npz"))
    train_edges = torch.from_numpy(edges["edge_index"][:, edges["train_idx"]].T.copy()).long()
    # slot ids + unordered slot-pair id matrix
    slot_vocab = sorted(set(slots))
    sidx = {s: i for i, s in enumerate(slot_vocab)}
    slot_of = torch.tensor([sidx[s] for s in slots], device=device)
    n_slot = len(slot_vocab)
    pair_of = torch.full((n_slot, n_slot), 0, dtype=torch.long)
    pid, seen = {}, 0
    for a in range(n_slot):
        for b in range(n_slot):
            key = (min(a, b), max(a, b))
            if key not in pid:
                pid[key] = len(pid)
            pair_of[a, b] = pid[key]
    n_pairs = len(pid)
    pair_of = pair_of.to(device)
    # per-node attribute category ids (aligned to node order via article_id -> meta)
    df = pd.read_csv(settings.meta_file, dtype=str).fillna("")
    df["article_id"] = df["article_id"].str.zfill(10)
    meta = df.drop_duplicates("article_id").set_index("article_id")
    cat_ids, cat_sizes = {}, {}
    for asp in ASPECTS:
        col = meta[asp] if asp in meta.columns else pd.Series("", index=meta.index)
        vocab = {v: i for i, v in enumerate(["<unk>"] + sorted(set(col.values)))}
        arr = []
        for aid in article_ids:
            v = meta.at[aid, asp] if aid in meta.index and asp in meta.columns else ""
            arr.append(vocab.get(v, 0))
        cat_ids[asp] = torch.tensor(arr, device=device)
        cat_sizes[asp] = len(vocab)
    return X, slots, slot_of, n_slot, pair_of, n_pairs, train_edges, cat_ids, cat_sizes, article_ids


class _NegSampler:
    """same-slot / complement-slot / random negatives (as in train_compat_metric)."""
    def __init__(self, slots, slot_of, device):
        self.slot_of, self.device, self.num_nodes = slot_of, device, len(slots)
        slot_vocab = sorted(set(slots))
        sidx = {s: i for i, s in enumerate(slot_vocab)}
        same = defaultdict(list)
        for i, s in enumerate(slots):
            same[sidx[s]].append(i)
        self.same_mat, self.same_len = self._pad(same, len(slot_vocab))
        comp = {i: [] for i in range(len(slot_vocab))}
        for ai, sa in enumerate(slot_vocab):
            cs = {sidx[sb] for sb in slot_vocab if slot_pair_allowed(sa, sb)}
            for i, s in enumerate(slots):
                if sidx[s] in cs:
                    comp[ai].append(i)
        self.comp_mat, self.comp_len = self._pad(comp, len(slot_vocab))

    def _pad(self, buckets, n):
        m = max((len(v) for v in buckets.values()), default=1)
        mat = torch.zeros(n, m, dtype=torch.long, device=self.device)
        lens = torch.zeros(n, dtype=torch.long, device=self.device)
        for s, nodes in buckets.items():
            if nodes:
                mat[s, :len(nodes)] = torch.tensor(nodes, device=self.device)
                lens[s] = len(nodes)
        return mat, lens

    def _from(self, mat, lens, sl, k):
        length = lens[sl].clamp(min=1).unsqueeze(1)
        rand = (torch.rand(sl.shape[0], k, device=self.device) * length).long()
        return mat[sl.unsqueeze(1), rand]

    def sample(self, anchor):
        sl = self.slot_of[anchor]
        return torch.cat([
            self._from(self.same_mat, self.same_len, sl, N_SUB),
            self._from(self.comp_mat, self.comp_len, sl, N_COMP),
            torch.randint(0, self.num_nodes, (anchor.shape[0], N_RAND), device=self.device),
        ], dim=1)


class TypeAwareHead(nn.Module):
    def __init__(self, in_dim, n_pairs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                                 nn.Linear(HIDDEN, EMB_DIM))
        self.mask = nn.Parameter(torch.zeros(n_pairs, EMB_DIM))  # sigmoid -> [0,1] diagonal mask

    def g(self, x):
        return F.normalize(self.net(x), dim=-1)

    def pair_score(self, ga, gx, pair_ids):
        m = torch.sigmoid(self.mask[pair_ids])
        return (ga * gx * m).sum(-1)


class DisentangledModel(nn.Module):
    def __init__(self, in_dim, cat_sizes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                                 nn.Linear(HIDDEN, EMB_DIM))
        self.emb = nn.ModuleDict({a: nn.Embedding(cat_sizes[a], ASPECT_DIM) for a in ASPECTS})
        self.w = nn.Parameter(torch.zeros(len(ASPECTS) + 1))  # softplus -> non-neg weights

    def g(self, x):
        return F.normalize(self.net(x), dim=-1)

    def aspect_affinity(self, asp, ca, cx):
        ea = F.normalize(self.emb[asp](ca), dim=-1)
        ex = F.normalize(self.emb[asp](cx), dim=-1)
        return (ea * ex).sum(-1)


def _info_nce(s_pos, s_neg):
    logits = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1) / TEMP
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def train_type_aware(X, slot_of, pair_of, n_pairs, train_edges, neg, device):
    torch.manual_seed(SEED)
    model = TypeAwareHead(X.shape[1], n_pairs).to(device)
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
            s_pos = model.pair_score(ga, gp, pair_of[sa, sp])
            pr = pair_of[sa.unsqueeze(1), slot_of[ng]]
            s_neg = model.pair_score(ga.unsqueeze(1), gn, pr)
            loss = _info_nce(s_pos, s_neg)
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * idx.shape[0]
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [typeaware] epoch {epoch:02d} loss={total / perm.shape[0]:.4f}")
    model.eval()
    with torch.no_grad():
        G = torch.cat([model.g(X[s:s + 8192]) for s in range(0, X.shape[0], 8192)]).cpu().numpy()
        mask = torch.sigmoid(model.mask).cpu().numpy()
    return G.astype(np.float32), mask.astype(np.float32)


def train_disentangled(X, cat_ids, cat_sizes, train_edges, neg, device):
    torch.manual_seed(SEED)
    model = DisentangledModel(X.shape[1], cat_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(train_edges.shape[0]); total = 0.0
        for start in range(0, perm.shape[0], BATCH):
            idx = perm[start:start + BATCH]; edge = train_edges[idx].to(device)
            flip = torch.rand(edge.shape[0], device=device) < 0.5
            a = torch.where(flip, edge[:, 1], edge[:, 0]); p = torch.where(flip, edge[:, 0], edge[:, 1])
            ng = neg.sample(a); nflat = ng.reshape(-1)
            w = F.softplus(model.w)
            ga, gp = model.g(X[a]), model.g(X[p])
            gn = model.g(X[nflat]).reshape(ng.shape[0], ng.shape[1], -1)
            s_pos = w[-1] * (ga * gp).sum(-1)
            s_neg = w[-1] * (ga.unsqueeze(1) * gn).sum(-1)
            for k, asp in enumerate(ASPECTS):
                ca, cp = cat_ids[asp][a], cat_ids[asp][p]
                s_pos = s_pos + w[k] * model.aspect_affinity(asp, ca, cp)
                cn = cat_ids[asp][nflat].reshape(ng.shape[0], ng.shape[1])
                s_neg = s_neg + w[k] * model.aspect_affinity(asp, ca.unsqueeze(1).expand_as(cn), cn)
            loss = _info_nce(s_pos, s_neg)
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * idx.shape[0]
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [disent]    epoch {epoch:02d} loss={total / perm.shape[0]:.4f}")
    model.eval()
    with torch.no_grad():
        G = torch.cat([model.g(X[s:s + 8192]) for s in range(0, X.shape[0], 8192)]).cpu().numpy().astype(np.float32)
        emb = {a: F.normalize(model.emb[a].weight, dim=-1).cpu().numpy().astype(np.float32) for a in ASPECTS}
        w = F.softplus(model.w).cpu().numpy().astype(np.float32)
    return G, emb, w


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, slots, slot_of, n_slot, pair_of, n_pairs, train_edges, cat_ids, cat_sizes, article_ids = _load_common(device)
    neg = _NegSampler(slots, slot_of, device)
    print(f"[exp] device={device} nodes={X.shape[0]} edges={train_edges.shape[0]} slot_pairs={n_pairs}")

    print("[exp] training type-aware ...")
    G_ta, mask = train_type_aware(X, slot_of, pair_of, n_pairs, train_edges, neg, device)
    # persist type-aware artifacts for production serving (CompatPairingIndex)
    np.save(os.path.join(DATA_DIR, "compat_typeaware_g.npy"), G_ta)
    np.save(os.path.join(DATA_DIR, "compat_typeaware_mask.npy"), mask)
    with open(os.path.join(DATA_DIR, "compat_typeaware_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"slot_vocab": sorted(set(slots)), "pair_of": pair_of.cpu().numpy().tolist()}, f)
    print("[exp] saved compat_typeaware_g/mask/meta")
    print("[exp] training disentangled ...")
    G_dis, emb_dis, w_dis = train_disentangled(X, cat_ids, cat_sizes, train_edges, neg, device)
    print("[exp] disentangled aspect weights:",
          {a: round(float(w_dis[k]), 3) for k, a in enumerate(ASPECTS)}, "resid", round(float(w_dis[-1]), 3))

    # numpy scorers for the eval harness
    slot_np = slot_of.cpu().numpy(); pair_np = pair_of.cpu().numpy()
    cat_np = {a: cat_ids[a].cpu().numpy() for a in ASPECTS}

    def score_typeaware(anchor, cands):
        m = mask[pair_np[slot_np[anchor], slot_np[cands]]]
        return np.sum(G_ta[anchor] * G_ta[cands] * m, axis=1)

    def score_disent(anchor, cands):
        s = w_dis[-1] * (G_dis[cands] @ G_dis[anchor])
        for k, asp in enumerate(ASPECTS):
            E = emb_dis[asp]
            s = s + w_dis[k] * np.sum(E[cat_np[asp][anchor]] * E[cat_np[asp][cands]], axis=1)
        return s

    compat = np.load(os.path.join(DATA_DIR, "compat_emb.npy"))
    extra = {"compat": compat, "typeaware": score_typeaware, "disent": score_disent}
    summary = {}
    for mode in ("hard_random", "pop_matched"):
        report = PairingEvaluator(extra_methods=extra, neg_mode=mode).evaluate()
        summary[mode] = report
        print(f"\n===== neg_mode={mode} =====")
        for group in ("warm", "cold"):
            print(f"[{group}]")
            for m in ("compat", "typeaware", "disent"):
                v = report[group][m]
                print(f"  {m:11s} AUC={v['auc']:.3f} MRR={v['mrr']:.3f} H@1={v['hit@1']:.3f} H@10={v['hit@10']:.3f}")
    with open(os.path.join(DATA_DIR, "exp_typeaware_disent.json"), "w", encoding="utf-8") as f:
        json.dump({"weights": {a: float(w_dis[k]) for k, a in enumerate(ASPECTS)} | {"resid": float(w_dis[-1])},
                   "report": summary}, f, indent=2)
    print("\nWrote exp_typeaware_disent.json")

    # --- blind VLM-judge set: compat vs typeaware top-1 complement for sample cold anchors ---
    _dump_judgeset(train_edges, slots, slot_np, article_ids, compat, score_typeaware)


def _dump_judgeset(train_edges, slots, slot_np, article_ids, compat, score_typeaware, n=14):
    import random
    from src.scripts.graph.outfit_slots import slot_pair_allowed
    rng = random.Random(SEED)
    deg = np.zeros(len(slots), dtype=np.int64)
    for a, b in train_edges.cpu().numpy().tolist():
        deg[a] += 1; deg[b] += 1
    slot_vocab = sorted(set(slots))
    def img(aid): return f"data/raw/images/{aid[:3]}/{aid}.jpg"
    def has_img(aid): return os.path.exists(img(aid))
    judge_slots = {"top", "bottom", "dress", "outerwear", "shoes", "bag", "inner"}
    cold = [i for i in range(len(slots)) if deg[i] == 0 and slots[i] in judge_slots and has_img(article_ids[i])]
    rng.shuffle(cold)
    by_slot = defaultdict(list)
    for i in range(len(slots)):
        by_slot[slots[i]].append(i)
    pairs, key = [], []
    for ai in cold:
        sa = slots[ai]
        comp_pool = [j for sb in slot_vocab if slot_pair_allowed(sa, sb) for j in by_slot[sb]
                     if has_img(article_ids[j]) and article_ids[j][:7] != article_ids[ai][:7]]
        if len(comp_pool) < 50:
            continue
        cand = np.asarray(comp_pool, dtype=np.int64)
        c_top = int(cand[np.argmax(compat[cand] @ compat[ai])])
        t_top = int(cand[np.argmax(score_typeaware(ai, cand))])
        if c_top == t_top:
            continue  # only show anchors where the two methods disagree
        opts = [("compat", c_top), ("typeaware", t_top)]
        rng.shuffle(opts)
        pairs.append({"anchor": article_ids[ai], "anchor_img": img(article_ids[ai]), "slot": sa,
                      "A": img(article_ids[opts[0][1]]), "B": img(article_ids[opts[1][1]])})
        key.append({"anchor": article_ids[ai], "A": opts[0][0], "B": opts[1][0]})
        if len(pairs) >= n:
            break
    with open(os.path.join(DATA_DIR, "judgeset.json"), "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
    with open(os.path.join(DATA_DIR, "judgeset_key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2)
    print(f"Wrote judgeset.json ({len(pairs)} disagreement pairs) + judgeset_key.json")


if __name__ == "__main__":
    main()
