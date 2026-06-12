"""Is NLU under-extraction actually "fail-safe"? When the gazetteer misses an attribute
(e.g. "oxblood" -> colour dropped), the word still goes into search_query. This measures
whether the retrieval pipeline (SigLIP dense + BM25 sparse, RRF, + cross-encoder rerank)
WITHOUT any hard filter still surfaces items matching the intended attribute.

For every NLU-gold query that has a gold colour_group (and product_type), run filter-free
retrieval on the raw query and report, over the top-10:
  colour@10   mean fraction of results whose colour_group is in the gold set
  type@10     mean fraction whose product_type matches the gold garment
  hit@10      fraction of queries with >=1 result matching BOTH colour and type

If colour@10/hit@10 are high, the missed hard-filter did not cost the user the result
(retrieval recovered it from the query text) -> "fail-safe" is justified. If low, it is not.

Run alone (Qdrant single-process):
  PYTORCH_JIT=0 USE_RERANKER=1 python -m src.scripts.indexing.eval_failsafe
"""
from __future__ import annotations

import json

from src.backend.core.config import BASE_DIR, settings
from src.backend.core.utils import normalize_article_id, normalize_text
from src.backend.retrieval.embeddings import load_sparse_encoder
from src.backend.retrieval.encoders import QueryEncoder, SigLIPEncoder
from src.backend.retrieval.qdrant import QdrantStore
from src.backend.services.catalog import FashionCatalog

GOLD_FILES = ["gold_nlu.json", "gold_nlu_hard.json", "gold_nlu_heldout.json"]
TOPK = 10
POOL = 50


def _type_match(pt: str, gold_words) -> bool:
    m = normalize_text(pt)
    return bool(m) and any(normalize_text(w) in m or m in normalize_text(w) for w in gold_words)


def main() -> None:
    catalog = FashionCatalog(settings.meta_file, settings.graph_file, settings.image_dir)
    store = QdrantStore(settings.db_path, settings.collection_name)
    sparse = load_sparse_encoder(settings.sparse_model_path)
    embedder = QueryEncoder(siglip=SigLIPEncoder(), sparse=sparse)
    reranker = None
    if settings.use_reranker:
        from src.backend.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()

    queries = []
    for fn in GOLD_FILES:
        for q in json.load(open(BASE_DIR / "data" / "eval" / fn, encoding="utf-8"))["queries"]:
            f = q.get("filters", {})
            if f.get("colour_group") and f.get("product_type"):
                queries.append(q)

    col_sum = type_sum = hit = 0
    rows = []
    for n, q in enumerate(queries, 1):
        gold_cols = {normalize_text(c) for c in q["filters"]["colour_group"]}
        gold_types = q["filters"]["product_type"]
        enc = embedder.encode(text=q["query"], image=None, sparse_text=q["query"])
        pts = store.hybrid_search(
            text_dense=enc.get("text_dense"), image_dense=None,
            sparse_indices=enc.get("sparse_indices", []), sparse_values=enc.get("sparse_values", []),
            limit=POOL,
        ) or []
        if reranker and len(pts) > 1:
            docs = []
            for p in pts:
                pl = getattr(p, "payload", {}) or {}
                docs.append(f"{pl.get('colour_group','')} {pl.get('product_type','')}. {pl.get('description','')}".strip())
            order = reranker.rerank(q["query"], docs, list(range(len(pts))))
            pts = [pts[i] for i in order]
        top = pts[:TOPK]

        col_ok = type_ok = 0
        both = False
        for p in top:
            aid = normalize_article_id(str((getattr(p, "payload", {}) or {}).get("article_id", "")))
            m = catalog.get_meta(aid)
            c = normalize_text(m.get("colour_group_name", "")) in gold_cols
            t = _type_match(m.get("product_type_name", ""), gold_types)
            col_ok += int(c); type_ok += int(t); both = both or (c and t)
        denom = max(1, len(top))
        col_sum += col_ok / denom; type_sum += type_ok / denom; hit += int(both)
        rows.append((q["id"], q["query"], round(col_ok / denom, 2), round(type_ok / denom, 2), int(both)))
        print(f"  [{n}/{len(queries)}] {q['id']} colour@10={col_ok}/{len(top)} type@10={type_ok}/{len(top)} both={both}", flush=True)

    nq = len(queries)
    print(f"\n== fail-safe eval (filter-free retrieval{' + rerank' if reranker else ''}, n={nq}) ==")
    print(f"  colour@10={col_sum/nq:.3f}  type@10={type_sum/nq:.3f}  hit@10(colour&type)={hit/nq:.3f}")
    worst = sorted(rows, key=lambda r: (r[4], r[2]))[:6]
    print("  weakest:", [(r[0], r[2], r[4]) for r in worst])


if __name__ == "__main__":
    main()
