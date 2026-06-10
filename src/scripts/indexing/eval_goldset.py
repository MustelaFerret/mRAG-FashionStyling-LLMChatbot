"""Gold-set retrieval eval with graded relevance + nDCG (leak-free).

Unlike the self-retrieval harness ([eval_retrieval.py]), relevance here is NOT taken
from the item's own indexed text. Each query is a hand-written natural-language request
([data/eval/gold_queries.json]); relevance is defined by a structured attribute predicate
over the catalog schema (product_type + colour / occasion / season / fit). The query
phrasing is independent of both the retriever's scoring and the gold definition, so there
is no text leak.

Graded relevance (for nDCG):
    2 = product_type matches AND every attr constraint matches   (perfect)
    1 = product_type matches only                                (right category)
    0 = otherwise
gain = 2**grade - 1  ->  grade2=3, grade1=1, grade0=0

Metrics (averaged over queries):
    nDCG@10   primary, graded ordering quality
    P@10      precision of grade>=1 and of grade==2 in the top-10
    hit@k     fraction of queries with >=1 grade-2 item in the top-k (k=1,5,10)
    MRR2      mean reciprocal rank of the first grade-2 item

Pure hybrid retrieval (dense text + BM25 sparse, RRF), NO NLU hard filter -- the score
reflects raw ranking quality, the substrate a reranker improves. Candidate lists are
cached to [data/eval/candidates.json] so a reranker A/B can re-score the same pools
without re-querying Qdrant.

Run alone (embedded Qdrant is single-process):
    PYTORCH_JIT=0 python -m src.scripts.indexing.eval_goldset
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

from src.backend.core.config import BASE_DIR, settings
from src.backend.core.utils import normalize_article_id
from src.backend.retrieval.qdrant import QdrantStore
from src.backend.retrieval.encoders import QueryEncoder, SigLIPEncoder
from src.backend.retrieval.embeddings import load_sparse_encoder
from src.backend.services.catalog import FashionCatalog

GOLD_FILE = str(BASE_DIR / "data" / "eval" / "gold_queries.json")
CAND_FILE = str(BASE_DIR / "data" / "eval" / "candidates.json")
RESULT_MD = str(BASE_DIR / "md" / "exp_eval_goldset.md")
TOPK = 10
CAND_DEPTH = 50  # candidates cached per query for reranker reuse


def grade_of(meta: Dict, q: Dict) -> int:
    """Graded relevance of one catalog item for one gold query."""
    pt = str(meta.get("product_type_name", "")).strip().lower()
    if pt not in {t.lower() for t in q["type"]}:
        return 0
    for field, vals in (q.get("attrs") or {}).items():
        cur = str(meta.get(field, "")).strip().lower()
        if cur not in {str(v).lower() for v in vals}:
            return 1
    return 2


def dcg(grades: List[int], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg(ranked_grades: List[int], corpus_grades: List[int], k: int) -> float:
    ideal = sorted(corpus_grades, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(ranked_grades, k) / idcg if idcg > 0 else 0.0


def main() -> Dict:
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        gold = json.load(f)["queries"]

    catalog = FashionCatalog(settings.meta_file, settings.graph_file, settings.image_dir)
    store = QdrantStore(settings.db_path, settings.collection_name)
    sparse = load_sparse_encoder(settings.sparse_model_path) if os.path.exists(settings.sparse_model_path) else None
    embedder = QueryEncoder(siglip=SigLIPEncoder(), sparse=sparse)

    items = list(catalog.meta_by_article.items())
    per_query: List[Dict] = []
    candidates: Dict[str, List[str]] = {}

    for n, q in enumerate(gold, 1):
        corpus_grades = [grade_of(m, q) for _, m in items]
        n2 = sum(1 for g in corpus_grades if g == 2)
        n1 = sum(1 for g in corpus_grades if g == 1)
        grade_by_aid = {aid: g for (aid, _), g in zip(items, corpus_grades) if g > 0}

        enc = embedder.encode(text=q["query"], image=None, sparse_text=q["query"])
        points = store.hybrid_search(
            text_dense=enc.get("text_dense"),
            image_dense=None,
            sparse_indices=enc.get("sparse_indices", []),
            sparse_values=enc.get("sparse_values", []),
            limit=CAND_DEPTH,
        ) or []
        ranked = [normalize_article_id(str((getattr(p, "payload", {}) or {}).get("article_id", ""))) for p in points]
        candidates[q["id"]] = ranked
        ranked_grades = [grade_by_aid.get(a, 0) for a in ranked]

        first2 = next((i + 1 for i, g in enumerate(ranked_grades) if g == 2), None)
        rec = {
            "id": q["id"],
            "query": q["query"],
            "gold2": n2,
            "gold1": n1,
            "ndcg@10": round(ndcg(ranked_grades, corpus_grades, TOPK), 4),
            "p@10>=1": round(sum(1 for g in ranked_grades[:TOPK] if g >= 1) / TOPK, 4),
            "p@10==2": round(sum(1 for g in ranked_grades[:TOPK] if g == 2) / TOPK, 4),
            "hit@1": int(any(g == 2 for g in ranked_grades[:1])),
            "hit@5": int(any(g == 2 for g in ranked_grades[:5])),
            "hit@10": int(any(g == 2 for g in ranked_grades[:10])),
            "rr2": round(1.0 / first2, 4) if first2 else 0.0,
        }
        per_query.append(rec)
        print(f"  [{n}/{len(gold)}] {q['id']} ndcg@10={rec['ndcg@10']} p@10>=1={rec['p@10>=1']} "
              f"hit@5={rec['hit@5']} (gold2={n2})", flush=True)

    agg = _aggregate(per_query)
    with open(CAND_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f)
    _write_md(per_query, agg)

    print("\n== gold-set eval (baseline: dense+BM25 RRF, no filter) ==")
    print("  " + "  ".join(f"{k}={v}" for k, v in agg.items()))
    print(f"  candidates -> {CAND_FILE}\n  report -> {RESULT_MD}")
    return {"aggregate": agg, "per_query": per_query}


def _aggregate(rows: List[Dict]) -> Dict[str, float]:
    n = len(rows)
    keys = ["ndcg@10", "p@10>=1", "p@10==2", "hit@1", "hit@5", "hit@10", "rr2"]
    out = {k: round(sum(r[k] for r in rows) / n, 4) for k in keys}
    out["n_queries"] = n
    return out


def _write_md(rows: List[Dict], agg: Dict) -> None:
    lines = [
        "# Gold-set retrieval eval (graded relevance, nDCG, leak-free)",
        "",
        "**Why**: self-retrieval ([eval_retrieval.py](../src/scripts/indexing/eval_retrieval.py)) leaks "
        "(query lấy từ chính text đã index). Ở đây query là NL viết tay, relevance định nghĩa bằng "
        "**predicate thuộc tính** (product_type + colour/occasion/season/fit) độc lập với cách hỏi và "
        "với scoring của retriever → **không leak**. Grade: 2 = type + mọi attr khớp; 1 = chỉ đúng type; 0 = khác.",
        "",
        "**Setup**: pure hybrid (dense text + BM25 sparse, RRF), **không** hard filter từ NLU — đo thẳng "
        "chất lượng ranking (nền mà reranker sẽ cải thiện). 30 query, gold = "
        "[gold_queries.json](../data/eval/gold_queries.json).",
        "",
        "## Aggregate (baseline)",
        "",
        "| nDCG@10 | P@10 (≥1) | P@10 (=2) | hit@1 | hit@5 | hit@10 | MRR2 | n |",
        "|---|---|---|---|---|---|---|---|",
        f"| **{agg['ndcg@10']}** | {agg['p@10>=1']} | {agg['p@10==2']} | {agg['hit@1']} | "
        f"{agg['hit@5']} | {agg['hit@10']} | {agg['rr2']} | {agg['n_queries']} |",
        "",
        "## Per-query",
        "",
        "| id | query | gold2 | nDCG@10 | P@10≥1 | P@10=2 | hit@5 | RR2 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['query']} | {r['gold2']} | {r['ndcg@10']} | "
            f"{r['p@10>=1']} | {r['p@10==2']} | {r['hit@5']} | {r['rr2']} |"
        )
    lines.append("")
    with open(RESULT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
