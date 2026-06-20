"""Per-customer purchase history (chronological article_ids) for the profiled customers, so the
re-ranker can compute taste_maxsim (nearest single past item) and taste_recency (recency-weighted
mean) -- the learned-blend signals that the mean taste vector alone cannot express. Reuses the
EXISTING profiles.csv customer set (does not touch the shipped profiles/taste artifacts).

    python -m src.scripts.personalization.export_purchases
"""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from src.scripts.personalization import config as C

CHUNK = 3_000_000
OUT = C.OUT_DIR / "customer_purchases.json"


def main():
    prof = pd.read_csv(C.PROFILES, dtype={"customer_id": str})
    sample = set(prof["customer_id"].astype(str))
    print(f"profiled customers: {len(sample):,}")

    seen = defaultdict(set)
    hist = defaultdict(list)  # cid -> list of (t_dat, aid) for first occurrence of each aid
    for chunk in pd.read_csv(C.TRANSACTIONS, chunksize=CHUNK,
                             usecols=["customer_id", "article_id", "t_dat"],
                             dtype={"article_id": str, "customer_id": str, "t_dat": "string"}):
        sub = chunk[chunk["customer_id"].isin(sample)]
        for cid, aid, t in zip(sub["customer_id"].values,
                               sub["article_id"].str.zfill(10).values,
                               sub["t_dat"].values):
            if aid not in seen[cid]:
                hist[cid].append((t, aid))
                seen[cid].add(aid)

    out = {}
    for cid, items in hist.items():
        items.sort()  # by date (t_dat is YYYY-MM-DD -> lexical == chronological)
        out[cid] = [a for _, a in items]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f)
    sizes = [len(v) for v in out.values()]
    print(f"wrote {OUT} | customers={len(out):,} | median purchases={int(pd.Series(sizes).median())}")


if __name__ == "__main__":
    main()
