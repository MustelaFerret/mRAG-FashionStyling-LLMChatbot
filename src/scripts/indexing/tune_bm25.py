"""Grid-search BM25 (k1, b) on the offline sparse-only self-retrieval metric. Tokenizes
the corpus once, then sweeps configs without touching Qdrant (doc weights are recomputed
in numpy per config). Pick the winner, then re-index once with build_bm25.

Caveat: tuned on self-retrieval (query=refined_description) — a relative proxy; we keep the
grid within standard BM25 ranges to avoid overfitting the proxy.

Run: python -m src.scripts.indexing.tune_bm25
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.backend.core.config import settings
from src.backend.core.utils import normalize_article_id
from src.backend.retrieval.embeddings import SparseTfidfEncoder
from src.scripts.indexing.build_vector_db import VectorIndexBuilder

SAMPLE = 300
TOPK = 10
SEED = 42
K1_GRID = [0.8, 1.2, 1.5, 2.0]
B_GRID = [0.25, 0.5, 0.75, 1.0]


def _qwords(text: str, n: int = 12) -> List[str]:
    return re.findall(r"[A-Za-z]+", str(text or ""))[:n]


def main() -> None:
    enc = SparseTfidfEncoder.load(settings.sparse_model_path.replace("sparse_bm25", "sparse_tfidf")) \
        if "bm25" in settings.sparse_model_path else SparseTfidfEncoder.load(settings.sparse_model_path)
    vocab = enc.vocab
    V = len(vocab)
    builder = VectorIndexBuilder.__new__(VectorIndexBuilder)
    df = pd.read_csv(settings.meta_file).fillna("")
    df["aid_norm"] = df["article_id"].map(lambda x: normalize_article_id(str(x)))

    # Tokenize corpus once -> per-doc vocab counts + df + lengths.
    counts: List[Dict[int, int]] = []
    aids: List[str] = []
    dfreq = np.zeros(V, dtype=np.int64)
    for row in df.itertuples(index=False):
        c: Dict[int, int] = {}
        for tok in enc._tokenize(builder.build_sparse_text(row._asdict())):
            j = vocab.get(tok)
            if j is not None:
                c[j] = c.get(j, 0) + 1
        counts.append(c)
        aids.append(getattr(row, "aid_norm"))
        for j in c:
            dfreq[j] += 1
    N = len(aids)
    dl = np.array([sum(c.values()) for c in counts], dtype=np.float64)
    avgdl = dl[dl > 0].mean()
    idf_bm25 = np.log(1.0 + (N - dfreq + 0.5) / (dfreq + 0.5))
    aid_to_doc = {a: i for i, a in enumerate(aids)}

    # Flatten (doc, term, raw_count) once; weights recomputed per (k1,b).
    rows_d, cols_t, raw_c = [], [], []
    for d, c in enumerate(counts):
        for j, cnt in c.items():
            rows_d.append(d); cols_t.append(j); raw_c.append(cnt)
    rows_d = np.asarray(rows_d); cols_t = np.asarray(cols_t); raw_c = np.asarray(raw_c, dtype=np.float64)
    dl_e = dl[rows_d]; idf_e = idf_bm25[cols_t]

    sample = df[df["refined_description"].astype(str).str.strip() != ""].sample(n=SAMPLE, random_state=SEED)
    queries = []
    for row in sample.itertuples(index=False):
        aid = getattr(row, "aid_norm")
        if aid not in aid_to_doc:
            continue
        q = " ".join([str(row.colour_group_name), str(row.product_type_name)] + _qwords(row.refined_description))
        qcols = sorted({vocab[t] for t in set(enc._tokenize(q)) if t in vocab})
        if qcols:
            queries.append((aid, qcols))

    print(f"docs={N} queries={len(queries)} grid={len(K1_GRID)}x{len(B_GRID)}")
    best = None
    for k1 in K1_GRID:
        for b in B_GRID:
            denom = raw_c + k1 * (1.0 - b + b * dl_e / avgdl)
            w = idf_e * (raw_c * (k1 + 1.0)) / denom
            D = csr_matrix((w, (rows_d, cols_t)), shape=(N, V))
            hits10, mrr = 0, 0.0
            for aid, qcols in queries:
                qv = np.zeros(V); qv[qcols] = 1.0
                order = np.argsort(-D.dot(qv))[:TOPK]
                top = [aids[i] for i in order]
                if aid in top:
                    r = top.index(aid) + 1
                    hits10 += 1
                    mrr += 1.0 / r
            n = len(queries)
            rec10, mrr_v = round(hits10 / n, 4), round(mrr / n, 4)
            tag = f"k1={k1} b={b}"
            print(f"  {tag:18s} recall@10={rec10}  mrr={mrr_v}")
            if best is None or mrr_v > best[0]:
                best = (mrr_v, rec10, k1, b)
    print(f"\nBEST: k1={best[2]} b={best[3]}  mrr={best[0]} recall@10={best[1]}")


if __name__ == "__main__":
    main()
