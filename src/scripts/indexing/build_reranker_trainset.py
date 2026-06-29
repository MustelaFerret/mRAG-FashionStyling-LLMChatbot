"""Build an in-domain cross-encoder training set for fine-tuning the reranker (bge-reranker-base).

Goal: teach the reranker the exact skill the gold eval measures -- discriminating WITHIN a product
type by attributes (a "black slim trousers" query must rank the black-slim-trousers item above other
trousers). So each example is:
  query   : a gold-style attribute phrase synthesised from the positive item's metadata
  positive: that item's serving doc text (must match rag_service._rerank_doc EXACTLY)
  hard neg: same product_type, >=1 attribute differs (the grade-1 confusers the reranker must beat)
  rand neg: a different product_type (easy)

Outputs a self-contained bundle (train.jsonl + eval.jsonl + manifest) and zips it for Colab.
Leak note: queries are template-generated, never the 30 hand-written gold query strings; items overlap
the catalogue exactly as at serving, which is intended.

Run: python -m src.scripts.indexing.build_reranker_trainset
"""
from __future__ import annotations

import json
import os
import random
import zipfile
from collections import defaultdict

import pandas as pd

from src.backend.core.config import settings

OUT_DIR = os.path.join(os.path.dirname(settings.meta_file), "reranker_bundle")
N_ITEMS = 36000      # items sampled to build training queries from
N_EVAL = 1500        # held-out items for an in-notebook ranking sanity check (disjoint from train)
N_HARD = 4
N_RAND = 2
DESC_CAP = 320
SEED = 42

# serving doc fields, in the order rag_service._rerank_doc uses (via payload <- CSV *_name)
HEAD_COLS = ["colour_group_name", "graphical_appearance_name", "dominant_material",
             "product_type_name", "fit", "occasion", "seasonality"]
OCC = {"Party/Evening/Wedding": "a party", "Office/Workwear": "the office",
       "Sport/Active/Workout": "the gym", "Lounge/Sleep/Nightwear": "lounge",
       "Beach/Swimwear": "the beach", "Outdoor/Adventure": "outdoors", "Casual/Everyday": "everyday"}
SEA = {"Autumn/Winter": "winter", "Spring/Summer": "summer"}
_SKIP = {"", "unknown", "solid", "regular", "regular fit", "no pattern"}
# only put fit / pattern in the QUERY when they are salient, gold-style descriptors (the catalogue's
# technical values like "Regular/Straight", "Other structure", "Front print" are noise in a query).
_FIT_OK = ("slim", "skinny", "wide", "loose", "oversized", "relaxed", "tailored", "baggy", "tight", "fitted")
_PAT_OK = ("floral", "stripe", "check", "dot", "animal", "leopard", "camo", "plaid", "paisley", "graphic", "tie dye")


def _salient(val: str, allow) -> str:
    v = _clean(val).lower()
    return v if any(a in v for a in allow) else ""


def _doc(row: dict) -> str:
    head = " ".join(str(row.get(k, "") or "").strip() for k in HEAD_COLS if str(row.get(k, "") or "").strip())
    desc = (str(row.get("refined_description", "") or "").strip()
            or str(row.get("detail_desc", "") or "").strip())[:DESC_CAP]
    return f"{head}. {desc}".strip()


def _clean(v: str) -> str:
    v = str(v or "").strip()
    return "" if v.lower() in _SKIP else v


def _make_query(row: dict, rng: random.Random) -> str:
    pt = _clean(row.get("product_type_name")).lower()
    if not pt:
        return ""
    col = _clean(row.get("colour_group_name")).lower()
    fit = _salient(row.get("fit"), _FIT_OK)
    pat = _salient(row.get("graphical_appearance_name"), _PAT_OK)
    parts = [p for p in (col, pat, fit, pt) if p]
    q = " ".join(parts)
    r = rng.random()
    occ, sea = OCC.get(str(row.get("occasion", "")), ""), SEA.get(str(row.get("seasonality", "")), "")
    if r < 0.30 and occ:
        q = f"{q} for {occ}"
    elif r < 0.50 and sea:
        q = f"{sea} {q}"
    return q.strip()


def _attr_key(row: dict) -> tuple:
    return (_clean(row.get("colour_group_name")).lower(), _clean(row.get("fit")).lower(),
            str(row.get("occasion", "")), _clean(row.get("graphical_appearance_name")).lower())


def main() -> None:
    df = pd.read_csv(settings.meta_file, dtype=str).fillna("")
    df["article_id"] = df["article_id"].str.zfill(10)
    df = df.drop_duplicates("article_id")
    rows = df.to_dict("records")
    rng = random.Random(SEED)
    rng.shuffle(rows)

    by_pt = defaultdict(list)
    for r in rows:
        by_pt[_clean(r.get("product_type_name"))].append(r)
    all_idx = list(range(len(rows)))

    def build(records, want_pairs: bool):
        out = []
        for row in records:
            pt = _clean(row.get("product_type_name"))
            q = _make_query(row, rng)
            if not q or not pt:
                continue
            pos_doc = _doc(row)
            pkey = _attr_key(row)
            pool = by_pt.get(pt, [])
            hard = []
            tries = 0
            while len(hard) < N_HARD and tries < 40 and len(pool) > 1:
                cand = pool[rng.randrange(len(pool))]
                tries += 1
                if cand is row or _attr_key(cand) == pkey:
                    continue
                hard.append(_doc(cand))
            rand = [_doc(rows[rng.randrange(len(rows))]) for _ in range(N_RAND)]
            negs = [d for d in hard + rand if d and d != pos_doc]
            if not negs:
                continue
            if want_pairs:
                out.append({"query": q, "doc": pos_doc, "label": 1})
                for d in negs:
                    out.append({"query": q, "doc": d, "label": 0})
            else:
                out.append({"query": q, "pos": pos_doc, "negs": negs})
        return out

    eval_records = rows[:N_EVAL]
    train_records = rows[N_EVAL:N_EVAL + N_ITEMS]
    train = build(train_records, want_pairs=True)
    evalset = build(eval_records, want_pairs=False)

    os.makedirs(OUT_DIR, exist_ok=True)
    train_path = os.path.join(OUT_DIR, "reranker_train.jsonl")
    eval_path = os.path.join(OUT_DIR, "reranker_eval.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for r in evalset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = sum(1 for r in train if r["label"] == 1)
    manifest = {"base_model": settings.reranker_model_id, "train_pairs": len(train),
                "train_positives": pos, "train_negatives": len(train) - pos,
                "eval_groups": len(evalset), "n_hard": N_HARD, "n_rand": N_RAND, "seed": SEED,
                "head_cols": HEAD_COLS, "desc_cap": DESC_CAP}
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    zip_path = os.path.join(OUT_DIR, "reranker_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (train_path, eval_path, os.path.join(OUT_DIR, "manifest.json")):
            z.write(p, os.path.basename(p))
    print(json.dumps(manifest, indent=2))
    print(f"\nbundle -> {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")
    print("\n-- sample train pairs --")
    for r in train[:6]:
        print(f"  label={r['label']} q={r['query']!r}\n     doc={r['doc'][:90]!r}")


if __name__ == "__main__":
    main()
