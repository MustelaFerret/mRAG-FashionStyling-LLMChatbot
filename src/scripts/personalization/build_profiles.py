"""Build user taste profiles (sample) + article popularity từ transactions.

Pass 1: đếm purchase/customer, chọn >=MIN_PURCHASES, sample SAMPLE_CUSTOMERS.
Pass 2: thu thập history (article_id, price) cho sample + popularity toàn cục.
Output: profiles.parquet, taste_vectors.npy + taste_index.json, article_popularity.parquet.

    python -m src.scripts.personalization.build_profiles
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.scripts.personalization import config as C

CHUNK = 3_000_000


def pass1_sample() -> set:
    counts = Counter()
    for chunk in pd.read_csv(C.TRANSACTIONS, chunksize=CHUNK, usecols=["customer_id"]):
        counts.update(chunk["customer_id"].values)
    eligible = [c for c, n in counts.items() if n >= C.MIN_PURCHASES]
    rng = random.Random(C.SEED)
    rng.shuffle(eligible)
    sample = set(eligible[: C.SAMPLE_CUSTOMERS])
    print(f"eligible(>= {C.MIN_PURCHASES}): {len(eligible):,} | sampled: {len(sample):,}")
    return sample


def pass2_collect(sample: set):
    history = defaultdict(list)
    prices = defaultdict(list)
    popularity = Counter()
    for chunk in tqdm(pd.read_csv(C.TRANSACTIONS, chunksize=CHUNK,
                                  usecols=["customer_id", "article_id", "price"],
                                  dtype={"article_id": str}), desc="pass2"):
        popularity.update(chunk["article_id"].str.zfill(10).values)
        mask = chunk["customer_id"].isin(sample)
        sub = chunk[mask]
        for cid, aid, pr in zip(sub["customer_id"].values, sub["article_id"].values, sub["price"].values):
            history[cid].append(str(aid).zfill(10))
            prices[cid].append(float(pr))
    return history, prices, popularity


def build():
    sample = pass1_sample()
    history, prices, popularity = pass2_collect(sample)

    meta = pd.read_csv(C.META_FILE, usecols=["article_id"] + C.PROFILE_FIELDS, dtype={"article_id": str})
    meta["article_id"] = meta["article_id"].str.zfill(10)
    meta_idx = meta.set_index("article_id")

    art_index = json.loads(C.ARTICLE_INDEX.read_text())
    art_emb = np.load(C.ARTICLE_EMB_NPY).astype(np.float32)

    rows = []
    taste_vecs = []
    taste_index = {}
    for cid, articles in tqdm(history.items(), desc="profiles"):
        valid = [a for a in articles if a in meta_idx.index]
        if not valid:
            continue
        sub = meta_idx.loc[valid]
        prof = {"customer_id": cid, "n_purchases": len(articles), "price_median": float(np.median(prices[cid]))}
        for field in C.PROFILE_FIELDS:
            vc = sub[field].dropna().astype(str)
            vc = vc[~vc.isin(["", "Unknown", "nan"])]
            top = vc.value_counts(normalize=True).head(C.TOP_K_PER_FIELD)
            prof[field] = json.dumps({k: round(float(v), 4) for k, v in top.items()}, ensure_ascii=False)

        emb_rows = [art_index[a] for a in valid if a in art_index]
        if emb_rows:
            tv = art_emb[emb_rows].mean(axis=0)
            tv /= (np.linalg.norm(tv) + 1e-8)
            taste_index[cid] = len(taste_vecs)
            taste_vecs.append(tv)
            rows.append(prof)

    profiles = pd.DataFrame(rows)
    profiles.to_csv(C.PROFILES, index=False)
    np.save(C.TASTE_NPY, np.asarray(taste_vecs, dtype=np.float16))
    C.TASTE_INDEX.write_text(json.dumps(taste_index))

    pop_df = pd.DataFrame({"article_id": list(popularity.keys()), "count": list(popularity.values())})
    pop_df.to_csv(C.ARTICLE_POP, index=False)

    print(f"profiles: {len(profiles)} | taste_vectors: {len(taste_vecs)} | popularity: {len(pop_df)} articles")
    print("sample profile:")
    print(profiles.iloc[0].to_dict())


if __name__ == "__main__":
    build()
