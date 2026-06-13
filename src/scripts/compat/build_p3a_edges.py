"""Build the P3alpha transaction-edge file shipped as the cold pairing tier
(data/processed/p3a_cold_edges.csv).

P3alpha = two-hop random walk on the user-item bipartite graph (item -> users who bought it
-> their other items), top-K partners per item, undirected, max-aggregated. Filtered by the
redesigned slot whitelist, then restricted to edges touching >=1 co-buy-cold node (those are
the items the co-buy graph cannot serve). Promoted here from the temp campaign so the shipped
artifact is reproducible. Methodology + head-to-head numbers: md/refine_5.MD.

Run: python -m src.scripts.compat.build_p3a_edges
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.backend.core.config import settings
from src.scripts.graph.build_graph import BuilderConfig, OutfitGraphBuilder

TOPK = 50
CHUNK = 1000
FULL_OUT = "data/processed/p3a_full.csv"
COLD_OUT = "data/processed/p3a_cold_edges.csv"


def _build_p3a_full() -> None:
    meta = pd.read_csv(settings.meta_file, usecols=["article_id"])
    valid = sorted(set(meta["article_id"].astype(str).str.zfill(10)))
    aid_to_idx = {a: i for i, a in enumerate(valid)}
    N = len(valid)

    df = pd.read_csv("data/raw/transactions_train.csv", usecols=["customer_id", "article_id"],
                     dtype={"article_id": str, "customer_id": "category"})
    df["article_id"] = df["article_id"].str.zfill(10)
    df = df[df["article_id"].isin(aid_to_idx)].drop_duplicates(subset=["customer_id", "article_id"])
    u = df["customer_id"].cat.codes.to_numpy(np.int32)
    i = df["article_id"].map(aid_to_idx).to_numpy(np.int32)
    del df
    R = csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(int(u.max()) + 1, N))
    deg_u = np.asarray(R.sum(1)).ravel()
    deg_i = np.asarray(R.sum(0)).ravel()
    P_ui = csr_matrix(R.multiply(1.0 / np.maximum(deg_u, 1)[:, None]))
    P_iu = csr_matrix(R.T.multiply(1.0 / np.maximum(deg_i, 1)[:, None]))
    del R

    rows, cols, vals = [], [], []
    for s in range(0, N, CHUNK):
        S = (P_iu[s:s + min(CHUNK, N - s)] @ P_ui).tocsr()
        for r in range(S.shape[0]):
            lo, hi = S.indptr[r], S.indptr[r + 1]
            idx, val = S.indices[lo:hi], S.data[lo:hi]
            mask = idx != (s + r)
            idx, val = idx[mask], val[mask]
            if idx.size > TOPK:
                top = np.argpartition(-val, TOPK - 1)[:TOPK]
                idx, val = idx[top], val[top]
            rows.append(np.full(idx.size, s + r, np.int32)); cols.append(idx.astype(np.int32)); vals.append(val.astype(np.float32))
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    lo = np.minimum(rows, cols).astype(np.int64); hi = np.maximum(rows, cols).astype(np.int64)
    codes = lo * N + hi
    order = np.argsort(codes, kind="stable"); codes, vals = codes[order], vals[order]
    uniq, start = np.unique(codes, return_index=True)
    agg = np.maximum.reduceat(vals, start)

    builder = OutfitGraphBuilder(BuilderConfig(
        transactions_file="data/raw/transactions_train.csv", meta_file=settings.meta_file,
        output_file=FULL_OUT, method="cobuy", content_filter="redesigned"))
    builder.load_meta()
    with open(FULL_OUT, "w", encoding="utf-8") as f:
        f.write("item_a,item_b,weight\n")
        for c, v in zip(uniq.tolist(), agg.tolist()):
            a, b = valid[c // N], valid[c % N]
            if builder._passes_content_filter(a, b, "redesigned"):
                f.write(f"{a},{b},{v * 1000:.4f}\n")
    print(f"[p3a] full edges -> {FULL_OUT}", flush=True)


def main() -> None:
    if not os.path.exists(FULL_OUT):
        _build_p3a_full()
    p3 = pd.read_csv(FULL_OUT)
    for c in ("item_a", "item_b"):
        p3[c] = p3[c].astype(str).str.zfill(10)
    prod = pd.read_csv(settings.graph_file)
    for c in ("item_a", "item_b"):
        prod[c] = prod[c].astype(str).str.zfill(10)
    warm = set(prod.item_a) | set(prod.item_b)
    cold = p3[(~p3.item_a.isin(warm)) | (~p3.item_b.isin(warm))]
    cold.to_csv(COLD_OUT, index=False)
    covered = len((set(cold.item_a) | set(cold.item_b)) - warm)
    print(f"[p3a] cold-touching edges: {len(cold)} | cold nodes covered: {covered} -> {COLD_OUT}")


if __name__ == "__main__":
    main()
