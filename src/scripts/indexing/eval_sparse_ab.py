"""A2 — offline sparse-only A/B: current TF-IDF vs BM25, isolated (no Qdrant mutation).

Two views:
  A. self-retrieval, query from refined_description. Same query for BOTH schemes, so leakage
     inflates absolute recall equally — the TF-IDF vs BM25 DELTA is a valid relative signal.
     (detail_desc is NOT used: it is the raw H&M text, shared across colourways and not
     article-unique, so it cannot identify a specific article.)
  C. qualitative: realistic natural-language queries, top-5 for each scheme — leak-free, the
     decisive "real query" check.

Run: python -m src.scripts.indexing.eval_sparse_ab
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
K_VALUES = (1, 5, 10)
K1, B = 1.5, 0.75

NL_QUERIES = [
    "oversized beige knit sweater",
    "flowy floral summer dress",
    "black leather chelsea boots",
    "dark blue padded parka with hood",
    "high waisted ripped skinny jeans",
    "elegant red evening gown",
]


def _query_words(text: str, n: int = 12) -> List[str]:
    return re.findall(r"[A-Za-z]+", str(text or ""))[:n]


def _metrics(ranks: List[int]) -> Dict[str, float]:
    n = max(1, len(ranks))
    out = {f"recall@{k}": round(sum(1 for r in ranks if r <= k) / n, 4) for k in K_VALUES}
    out["mrr"] = round(sum(1.0 / r for r in ranks if r != 10**9) / n, 4)
    return out


class SparseAB:
    def __init__(self):
        self.enc = SparseTfidfEncoder.load(settings.sparse_model_path)
        self.vocab, self.V = self.enc.vocab, len(self.enc.vocab)
        builder = VectorIndexBuilder.__new__(VectorIndexBuilder)
        self.df = pd.read_csv(settings.meta_file).fillna("")
        self.df["_aid"] = self.df["article_id"].map(lambda x: normalize_article_id(str(x)))

        self.doc_aids: List[str] = []
        self.doc_meta: List[tuple] = []
        counts_list: List[Dict[int, int]] = []
        dfreq = np.zeros(self.V, dtype=np.int64)
        for _, row in self.df.iterrows():
            counts: Dict[int, int] = {}
            for tok in self.enc._tokenize(builder.build_sparse_text(row)):
                j = self.vocab.get(tok)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            self.doc_aids.append(row["_aid"])
            self.doc_meta.append((str(row.get("prod_name", "")), str(row.get("product_type_name", "")), str(row.get("colour_group_name", ""))))
            counts_list.append(counts)
            for j in counts:
                dfreq[j] += 1

        N = len(self.doc_aids)
        dl = np.array([sum(c.values()) for c in counts_list], dtype=np.float64)
        avgdl = dl[dl > 0].mean()
        idf_bm25 = np.log(1.0 + (N - dfreq + 0.5) / (dfreq + 0.5))
        idf_arr = np.asarray(self.enc.idf, dtype=np.float64)

        rt, rb, cols, vt, vb = [], [], [], [], []
        for d, counts in enumerate(counts_list):
            for j, c in counts.items():
                cols.append(j); rt.append(d); rb.append(d)
                vt.append((c / dl[d]) * idf_arr[j] if dl[d] else 0.0)
                denom = c + K1 * (1.0 - B + B * dl[d] / avgdl)
                vb.append(idf_bm25[j] * (c * (K1 + 1.0)) / denom if denom else 0.0)
        self.D_tfidf = csr_matrix((vt, (rt, cols)), shape=(N, self.V))
        self.D_bm25 = csr_matrix((vb, (rb, cols)), shape=(N, self.V))
        self.aid_to_doc = {a: i for i, a in enumerate(self.doc_aids)}

    def _q_tfidf(self, q: str) -> np.ndarray:
        qi, qv = self.enc.encode(q)
        v = np.zeros(self.V); v[list(qi)] = qv
        return v

    def _q_bm25(self, q: str) -> np.ndarray:
        v = np.zeros(self.V)
        for t in set(self.enc._tokenize(q)):
            j = self.vocab.get(t)
            if j is not None:
                v[j] = 1.0
        return v

    def _topk(self, D, qvec, k) -> List[int]:
        return list(np.argsort(-D.dot(qvec))[:k])

    def self_retrieval(self, query_field: str):
        cand = self.df[self.df[query_field].astype(str).str.strip() != ""]
        sample = cand.sample(n=min(SAMPLE, len(cand)), random_state=SEED)
        rt, rb = [], []
        for _, row in sample.iterrows():
            aid = row["_aid"]
            if aid not in self.aid_to_doc:
                continue
            words = _query_words(row[query_field])
            q = " ".join([str(row.get("colour_group_name", "")), str(row.get("product_type_name", ""))] + words).strip()
            if not self.enc._tokenize(q):
                continue
            for D, qvec, ranks in ((self.D_tfidf, self._q_tfidf(q), rt), (self.D_bm25, self._q_bm25(q), rb)):
                top = [self.doc_aids[i] for i in self._topk(D, qvec, TOPK)]
                ranks.append(top.index(aid) + 1 if aid in top else 10**9)
        return _metrics(rt), _metrics(rb)

    def qualitative(self, query: str, k=5):
        out = {}
        for name, D, qvec in (("TF-IDF", self.D_tfidf, self._q_tfidf(query)), ("BM25", self.D_bm25, self._q_bm25(query))):
            rows = []
            for i in self._topk(D, qvec, k):
                pn, pt, cg = self.doc_meta[i]
                rows.append(f"{pt}/{cg} ({pn[:22]})")
            out[name] = rows
        return out


def main() -> None:
    ab = SparseAB()
    print(f"docs={len(ab.doc_aids)} sample={SAMPLE} topk={TOPK} k1={K1} b={B}\n")

    print("[A] self-retrieval — query from refined_description (same query both schemes -> relative A/B)")
    mt, mb = ab.self_retrieval("refined_description")
    print("    TF-IDF", mt, "\n    BM25  ", mb, "\n")

    print("[C] qualitative — realistic NL queries, top-5 (leak-free)")
    for q in NL_QUERIES:
        res = ab.qualitative(q)
        print(f"  Q: {q}")
        print("     TF-IDF:", " | ".join(res["TF-IDF"]))
        print("     BM25  :", " | ".join(res["BM25"]))


if __name__ == "__main__":
    main()
