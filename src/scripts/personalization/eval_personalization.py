"""Leak-free evaluation of personalization signals — the piece the shipped reranker lacks
(its blend weights were hand-set, never measured).

Protocol: temporal leave-last-out. For each eval customer, hold out their LAST purchased
item h; build the taste profile from the EARLIER purchases only; rank h against negatives.
Two negative regimes:
  pop_matched : negatives from h's log-popularity band  -> neutralises popularity bias
  same_pt     : negatives share h's product_type         -> the hard, rerank-realistic test
                ("within a category, does taste pick the right item?")

Signals scored (cosine/score of candidate vs profile):
  random, popularity (non-personalised baseline),
  taste_mean  : cos(mean image_emb of past purchases, candidate)
  taste_maxsim: max cos to ANY past purchase (handles multi-faceted taste)
  cat_match   : current categorical-preference score
  blend_ship  : the shipped linear blend (retrieval prior excluded — no query here)
Metrics: MRR, Hit@1/5/10, AUC (tie-aware). Reported mean over eval customers.

Run: python -m src.scripts.personalization.eval_personalization
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.backend.core.config import settings

PDIR = os.path.join(os.path.dirname(settings.meta_file), "personalization")
TX = "data/raw/transactions_train.csv"
N_EVAL = 3000
NUM_NEG = 100
MIN_HISTORY = 6      # need enough past to build a profile after holding one out
SEED = 42
K = (1, 5, 10)
CAT_FIELDS = ["product_type_name", "colour_group_name", "style_aesthetic", "occasion", "index_group_name"]


def _load_assets():
    art_index = {k: int(v) for k, v in json.loads(open(os.path.join(PDIR, "article_index.json")).read()).items()}
    art_emb = np.load(os.path.join(PDIR, "article_image_emb.npy")).astype(np.float32)
    art_emb /= (np.linalg.norm(art_emb, axis=1, keepdims=True) + 1e-8)
    pop = pd.read_csv(os.path.join(PDIR, "article_popularity.csv"), dtype={"article_id": str})
    pop["article_id"] = pop["article_id"].str.zfill(10)
    popcount = {a: int(c) for a, c in zip(pop["article_id"], pop["count"])}
    meta = pd.read_csv(settings.meta_file, usecols=["article_id"] + CAT_FIELDS, dtype={"article_id": str})
    meta["article_id"] = meta["article_id"].str.zfill(10)
    cat = {r.article_id: {f: str(getattr(r, f) or "") for f in CAT_FIELDS} for r in meta.itertuples()}
    return art_index, art_emb, popcount, cat


def _eval_customers(rng):
    prof = pd.read_csv(os.path.join(PDIR, "profiles.csv"), dtype={"customer_id": str})
    cids = prof["customer_id"].tolist()
    rng.shuffle(cids)
    return set(cids[: N_EVAL * 2])  # over-sample; some drop after history filter


def _collect_history(sample: set):
    """Ordered unique purchases per customer (article_id by date)."""
    hist = defaultdict(list)
    seen = defaultdict(set)
    for chunk in pd.read_csv(TX, chunksize=3_000_000, usecols=["customer_id", "article_id", "t_dat"],
                             dtype={"article_id": str, "customer_id": str, "t_dat": "string"}):
        sub = chunk[chunk["customer_id"].isin(sample)]
        for cid, aid, t in zip(sub["customer_id"].values, sub["article_id"].str.zfill(10).values, sub["t_dat"].values):
            if aid not in seen[cid]:
                hist[cid].append((t, aid)); seen[cid].add(aid)
    for cid in hist:
        hist[cid].sort()
    return hist


def main():
    rng = np.random.default_rng(SEED)
    art_index, art_emb, popcount, cat = _load_assets()
    sample = _eval_customers(np.random.default_rng(SEED))
    hist = _collect_history(sample)

    # popularity bins for matched negatives
    arts = [a for a in art_index if a in popcount]
    art_arr = np.array(arts)
    logpop = np.array([np.log2(popcount[a] + 1) for a in arts])
    bin_of = np.floor(logpop).astype(int)
    bin_to_arts = defaultdict(list)
    for a, b in zip(arts, bin_of):
        bin_to_arts[b].append(a)
    pt_of = {a: cat.get(a, {}).get("product_type_name", "") for a in arts}
    pt_to_arts = defaultdict(list)
    for a in arts:
        pt_to_arts[pt_of[a]].append(a)
    maxpop = max(popcount.values())

    def cat_profile(past):
        prof = {}
        for f in CAT_FIELDS:
            c = Counter(cat.get(a, {}).get(f, "") for a in past)
            c.pop("", None); c.pop("Unknown", None)
            tot = sum(c.values()) or 1
            prof[f] = {k: v / tot for k, v in c.most_common(5)}
        return prof

    def sample_negs(target, mode):
        pool = bin_to_arts[int(np.floor(np.log2(popcount.get(target, 1) + 1)))] if mode == "pop_matched" else pt_to_arts[pt_of.get(target, "")]
        if len(pool) <= NUM_NEG:
            pool = arts
        picks = rng.choice(len(pool), size=min(NUM_NEG * 3, len(pool)), replace=False)
        out = []
        for i in picks.tolist():
            a = pool[i]
            if a != target:
                out.append(a)
            if len(out) >= NUM_NEG:
                break
        return out

    FEATS = ["popularity", "taste_mean", "taste_maxsim", "taste_recency", "cat_match"]

    def features(past_emb, recency_w, cprof, cands):
        ci = np.array([art_index[a] for a in cands])
        E = art_emb[ci]
        mean_v = past_emb.mean(0); mean_v /= (np.linalg.norm(mean_v) + 1e-8)
        rec_v = (recency_w[:, None] * past_emb).sum(0); rec_v /= (np.linalg.norm(rec_v) + 1e-8)
        sims = E @ past_emb.T
        catm = np.array([sum(cprof[f].get(cat.get(a, {}).get(f, ""), 0.0) for f in CAT_FIELDS) / len(CAT_FIELDS)
                         for a in cands])
        return {
            "popularity": np.array([np.log1p(popcount.get(a, 0)) / np.log1p(maxpop) for a in cands]),
            "taste_mean": E @ mean_v,
            "taste_maxsim": sims.max(1),
            "taste_recency": E @ rec_v,
            "cat_match": catm,
        }

    methods = ["random", "popularity", "taste_mean", "taste_maxsim", "taste_recency",
               "cat_match", "blend_ship", "learned"]

    # Phase 1: collect per-(customer, mode) candidate feature matrices (target = row 0).
    # 50/50 customer split: logistic 'learned' blend is fit on TRAIN customers only.
    collected = {"pop_matched": [], "same_pt": []}
    n_used = 0
    for cid, h in hist.items():
        if len(h) < MIN_HISTORY:
            continue
        past = [a for _, a in h[:-1] if a in art_index]
        target = h[-1][1]
        if target not in art_index or len(past) < MIN_HISTORY - 1:
            continue
        n_used += 1
        if n_used > N_EVAL:
            break
        past_emb = art_emb[[art_index[a] for a in past]]
        recency_w = np.exp(np.linspace(-2.0, 0.0, len(past))).astype(np.float32)  # recent weighted up
        cprof = cat_profile(past)
        # stable hash (md5) so the train/test customer split is reproducible across runs and
        # machines; Python's builtin hash() is salted by PYTHONHASHSEED -> the learned-blend
        # numbers were unrepeatable before this.
        is_test = (int(hashlib.md5(cid.encode()).hexdigest(), 16) & 1) == 0
        for mode in ("pop_matched", "same_pt"):
            negs = sample_negs(target, mode)
            if len(negs) < 10:
                continue
            cands = [target] + negs
            feat = features(past_emb, recency_w, cprof, cands)
            collected[mode].append((feat, len(cands), is_test, rng.random(len(cands))))

    # Phase 2: fit logistic blend per mode on TRAIN rows
    from sklearn.linear_model import LogisticRegression
    learned_models = {}
    for mode in ("pop_matched", "same_pt"):
        Xtr, ytr = [], []
        for feat, n, is_test, _ in collected[mode]:
            if is_test:
                continue
            M = np.stack([feat[f] for f in FEATS], axis=1)
            lab = np.zeros(n); lab[0] = 1.0
            Xtr.append(M); ytr.append(lab)
        X = np.vstack(Xtr); y = np.concatenate(ytr)
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xz, y)
        learned_models[mode] = (lr, X.mean(0), X.std(0))
        print(f"[learned:{mode}] weights " + " ".join(f"{f}={w:+.2f}" for f, w in zip(FEATS, lr.coef_[0])), flush=True)
    weights_out = {mode: {f: float(w) for f, w in zip(FEATS, learned_models[mode][0].coef_[0])}
                   for mode in learned_models}

    # Phase 3: eval all methods on TEST customers
    out = {}
    print(f"\neval customers used: {n_used} (test split scored)\n")
    for mode in ("pop_matched", "same_pt"):
        lr, mu, sd = learned_models[mode]
        agg = {m: {"mrr": 0.0, "auc": 0.0, **{f"hit@{k}": 0.0 for k in K}, "n": 0} for m in methods}
        for feat, n, is_test, randsc in collected[mode]:
            if not is_test:
                continue
            M = np.stack([feat[f] for f in FEATS], axis=1)
            sc = {f: feat[f] for f in FEATS}
            sc["random"] = randsc
            sc["blend_ship"] = 0.6 * feat["taste_mean"] + 0.5 * feat["cat_match"] + 0.2 * feat["popularity"]
            sc["learned"] = lr.predict_proba((M - mu) / (sd + 1e-8))[:, 1]
            for m in methods:
                s = sc[m]; tgt = s[0]; negv = s[1:]
                rank = int(np.sum(negv > tgt)) + 1 + int(np.sum(negv == tgt)) / 2.0
                a = agg[m]; a["n"] += 1; a["mrr"] += 1.0 / rank
                a["auc"] += float((np.sum(negv < tgt) + 0.5 * np.sum(negv == tgt)) / len(negv))
                for k in K:
                    if rank <= k:
                        a[f"hit@{k}"] += 1.0
        print(f"===== neg_mode={mode} =====")
        for m in methods:
            a = agg[m]; nn = max(1, a["n"])
            row = {"mrr": a["mrr"] / nn, "auc": a["auc"] / nn, **{f"hit@{k}": a[f"hit@{k}"] / nn for k in K}, "n": a["n"]}
            out[f"{mode}|{m}"] = row
            print(f"  {m:13s} AUC={row['auc']:.4f} MRR={row['mrr']:.4f} H@1={row['hit@1']:.3f} H@5={row['hit@5']:.3f} H@10={row['hit@10']:.3f}")
        print()
    json.dump({"metrics": out, "learned_weights": weights_out, "feats": FEATS,
               "n_eval": n_used, "num_neg": NUM_NEG},
              open(os.path.join(PDIR, "eval_personalization.json"), "w"), indent=2)

    # qualitative sample for the notebook: a few customers' past purchases + held-out target
    qual = []
    for cid, h in list(hist.items()):
        if len(h) < 8 or len(qual) >= 12:
            continue
        past = [a for _, a in h[:-1] if a in art_index]
        target = h[-1][1]
        if target in art_index and len(past) >= 6:
            qual.append({"customer_id": cid, "past": past[-8:], "target": target})
    json.dump(qual, open(os.path.join(PDIR, "eval_qual_sample.json"), "w"))


if __name__ == "__main__":
    main()
