"""Experiment: fusion-strategy ablation for hybrid retrieval (gold set, graded nDCG, no NLU filter).

The report already justifies the encoder (SigLIP vs CLIP), the sparse model (BM25 vs TF-IDF) and the
reranker. What it does NOT isolate is the FUSION choice itself. Here I compare, on the same leak-free
gold set and grading as eval_goldset:
  - dense-only        (SigLIP text->image cosine)
  - sparse-only       (BM25)
  - RRF               (rank fusion -- what the system serves; Qdrant's reciprocal-rank fusion)
  - score-convex      (min-max normalise each modality's scores, then weighted sum -- score fusion)
Each is scored on the candidate pool (pre-rerank) so the fusion's own contribution is visible, and
then the cross-encoder reranker is applied to each pool to test whether the fusion choice survives it.

Run: python -m src.scripts.indexing.exp_fusion_ablation
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List

from src.backend.core.config import settings
from src.backend.core.utils import normalize_article_id
from src.backend.retrieval.embeddings import load_sparse_encoder
from src.backend.retrieval.encoders import QueryEncoder, SigLIPEncoder
from src.backend.retrieval.qdrant import QdrantStore
from src.backend.services.catalog import FashionCatalog
from src.scripts.indexing.eval_goldset import GOLD_FILE, TOPK, grade_of, ndcg

DEPTH = 100
CONV_W = 0.5  # dense weight in the convex score fusion (sparse gets 1-w)
METHODS = ["dense", "sparse", "rrf", "score-convex"]


def _aid(p) -> str:
    return normalize_article_id(str((getattr(p, "payload", {}) or {}).get("article_id", "")))


def _scored(points) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in points:
        a = _aid(p)
        if a and a not in out:
            out[a] = float(getattr(p, "score", 0.0) or 0.0)
    return out


def _minmax(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return {a: (s - lo) / rng for a, s in d.items()}


def _rrf(rank_lists: List[List[str]], k: int = 60) -> List[str]:
    score: Dict[str, float] = defaultdict(float)
    for rl in rank_lists:
        for r, a in enumerate(rl):
            score[a] += 1.0 / (k + r + 1)
    return sorted(score, key=lambda a: -score[a])


def main() -> None:
    gold = json.load(open(GOLD_FILE, encoding="utf-8"))["queries"]
    catalog = FashionCatalog(settings.meta_file, settings.graph_file, settings.image_dir)
    store = QdrantStore(settings.db_path, settings.collection_name)
    sparse = load_sparse_encoder(settings.sparse_model_path) if os.path.exists(settings.sparse_model_path) else None
    embedder = QueryEncoder(siglip=SigLIPEncoder(), sparse=sparse)
    reranker = None
    try:
        from src.backend.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
    except Exception as exc:  # noqa: BLE001
        print(f"[fusion] reranker unavailable ({exc}); pre-rerank only")

    items = list(catalog.meta_by_article.items())
    agg = {m: defaultdict(float) for m in METHODS}
    agg_rr = {m: defaultdict(float) for m in METHODS}

    for n, q in enumerate(gold, 1):
        corpus_grades = [grade_of(m, q) for _, m in items]
        grade_by_aid = {aid: g for (aid, _), g in zip(items, corpus_grades) if g > 0}
        enc = embedder.encode(text=q["query"], image=None, sparse_text=q["query"])

        dense_pts = store.hybrid_search(text_dense=enc.get("text_dense"), image_dense=None,
                                        sparse_indices=[], sparse_values=[], limit=DEPTH) or []
        sparse_pts = store.hybrid_search(text_dense=None, image_dense=None,
                                         sparse_indices=enc.get("sparse_indices", []),
                                         sparse_values=enc.get("sparse_values", []), limit=DEPTH) or []
        rrf_pts = store.hybrid_search(text_dense=enc.get("text_dense"), image_dense=None,
                                      sparse_indices=enc.get("sparse_indices", []),
                                      sparse_values=enc.get("sparse_values", []), limit=DEPTH) or []

        dense_rank = [a for a in (_aid(p) for p in dense_pts) if a]
        sparse_rank = [a for a in (_aid(p) for p in sparse_pts) if a]
        rrf_rank = [a for a in (_aid(p) for p in rrf_pts) if a]
        dn, sn = _minmax(_scored(dense_pts)), _minmax(_scored(sparse_pts))
        conv = {a: CONV_W * dn.get(a, 0.0) + (1 - CONV_W) * sn.get(a, 0.0) for a in set(dn) | set(sn)}
        conv_rank = sorted(conv, key=lambda a: -conv[a])

        ranked = {"dense": dense_rank, "sparse": sparse_rank, "rrf": rrf_rank, "score-convex": conv_rank}
        for m, rl in ranked.items():
            rg = [grade_by_aid.get(a, 0) for a in rl]
            _accumulate(agg[m], rg, corpus_grades)
            if reranker is not None:
                rr = _rerank(reranker, q["query"], rl[:DEPTH], catalog)
                rrg = [grade_by_aid.get(a, 0) for a in rr]
                _accumulate(agg_rr[m], rrg, corpus_grades)
        print(f"  [{n}/{len(gold)}] {q['id']}", flush=True)

    print("\n== fusion ablation (gold set, graded nDCG, no NLU filter) ==")
    _print("PRE-RERANK (candidate pool quality)", agg, len(gold))
    if reranker is not None:
        _print("POST-RERANK (cross-encoder on each pool)", agg_rr, len(gold))
    out = {"pre_rerank": {m: _final(agg[m], len(gold)) for m in METHODS},
           "post_rerank": {m: _final(agg_rr[m], len(gold)) for m in METHODS} if reranker else {}}
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(settings.meta_file)))
    with open(os.path.join(repo_root, "md", "exp_fusion_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _rerank(reranker, query: str, aids: List[str], catalog) -> List[str]:
    docs = []
    for a in aids:
        m = catalog.get_meta(a) or {}
        head = " ".join(str(m.get(k, "") or "") for k in
                        ("colour_group_name", "graphical_appearance_name", "dominant_material",
                         "product_type_name", "fit", "occasion", "seasonality") if m.get(k))
        docs.append(f"{head}. {m.get('refined_description', '') or m.get('detail_desc', '')}".strip())
    scores = reranker.score(query, docs)
    return [a for _, a in sorted(zip(scores, aids), key=lambda t: -t[0])]


def _accumulate(acc, ranked_grades, corpus_grades) -> None:
    acc["n"] += 1
    acc["ndcg@10"] += ndcg(ranked_grades, corpus_grades, TOPK)
    acc["p@10>=1"] += sum(1 for g in ranked_grades[:TOPK] if g >= 1) / TOPK
    acc["hit@5"] += 1.0 if any(g == 2 for g in ranked_grades[:5]) else 0.0


def _final(acc, n) -> Dict[str, float]:
    n = max(1, n)
    return {k: round(acc[k] / n, 4) for k in ("ndcg@10", "p@10>=1", "hit@5")}


def _print(title, agg, n) -> None:
    print(f"\n-- {title} --")
    print(f"  {'method':12} {'nDCG@10':>8} {'P@10>=1':>8} {'hit@5':>7}")
    for m in METHODS:
        f = _final(agg[m], n)
        print(f"  {m:12} {f['ndcg@10']:8.4f} {f['p@10>=1']:8.4f} {f['hit@5']:7.4f}")


if __name__ == "__main__":
    main()
