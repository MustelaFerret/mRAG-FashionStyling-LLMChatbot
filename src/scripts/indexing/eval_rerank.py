"""Measure a cross-encoder reranker on the gold-set, re-scoring the fused candidate pool
cached by eval_goldset.py ([data/eval/candidates.json], top-50 per query). Offline: reads
files only, never opens Qdrant, so it is safe to run alongside other work.

Compares three orderings on the SAME candidate pool:
  fused   - the hybrid RRF order (baseline, what eval_goldset measured)
  rerank  - pure cross-encoder relevance order
  blend   - RRF of (fused rank, rerank rank), keeps both signals

Doc text fed to the reranker = "<colour> <product_type>. <refined_description>" so the
cross-encoder sees explicit type/colour anchors plus the rich description.

Run: PYTORCH_JIT=0 python -m src.scripts.indexing.eval_rerank
"""
from __future__ import annotations

import json
import os

from src.backend.core.config import BASE_DIR, settings
from src.backend.retrieval.reranker import CrossEncoderReranker
from src.backend.services.catalog import FashionCatalog
from src.scripts.indexing.eval_goldset import CAND_FILE, GOLD_FILE, TOPK, grade_of, ndcg

RESULT_MD = str(BASE_DIR / "md" / "exp_rerank.md")
DEPTH = settings.rerank_candidate_depth
METRIC_KEYS = ["ndcg@10", "p@10>=1", "p@10==2", "hit@1", "hit@5", "hit@10", "rr2"]


def _doc_text(meta: dict) -> str:
    # structured head: +4.7pp rerank nDCG vs colour+type alone (md/refine_2.MD);
    # mirrors rag_service._rerank_blend doc construction (payload key names differ)
    head = " ".join(x for x in [
        meta.get("colour_group_name", ""), meta.get("graphical_appearance_name", ""),
        meta.get("dominant_material", ""), meta.get("product_type_name", ""),
        meta.get("fit", ""), meta.get("occasion", ""), meta.get("seasonality", ""),
    ] if x)
    desc = str(meta.get("refined_description", "") or "")
    return f"{head}. {desc}".strip()


def _metrics(ranked_grades, corpus_grades):
    first2 = next((i + 1 for i, g in enumerate(ranked_grades) if g == 2), None)
    return {
        "ndcg@10": ndcg(ranked_grades, corpus_grades, TOPK),
        "p@10>=1": sum(1 for g in ranked_grades[:TOPK] if g >= 1) / TOPK,
        "p@10==2": sum(1 for g in ranked_grades[:TOPK] if g == 2) / TOPK,
        "hit@1": float(any(g == 2 for g in ranked_grades[:1])),
        "hit@5": float(any(g == 2 for g in ranked_grades[:5])),
        "hit@10": float(any(g == 2 for g in ranked_grades[:10])),
        "rr2": 1.0 / first2 if first2 else 0.0,
    }


def _rrf(rank_a, rank_b, k=60):
    return 1.0 / (k + rank_a) + 1.0 / (k + rank_b)


def main() -> dict:
    gold = json.load(open(GOLD_FILE, encoding="utf-8"))["queries"]
    candidates = json.load(open(CAND_FILE, encoding="utf-8"))
    catalog = FashionCatalog(settings.meta_file, settings.graph_file, settings.image_dir)
    reranker = CrossEncoderReranker()

    items = list(catalog.meta_by_article.items())
    agg = {cfg: {k: 0.0 for k in METRIC_KEYS} for cfg in ("fused", "rerank", "blend")}
    rows = []

    for n, q in enumerate(gold, 1):
        corpus_grades = [grade_of(m, q) for _, m in items]
        grade_by_aid = {aid: g for (aid, _), g in zip(items, corpus_grades) if g > 0}

        cand = candidates[q["id"]][:DEPTH]
        docs = [_doc_text(catalog.get_meta(a)) for a in cand]
        scores = reranker.score(q["query"], docs)

        fused_order = cand  # candidates.json is already in fused rank order
        rerank_order = [cand[i] for i in sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)]
        fused_rank = {a: i for i, a in enumerate(fused_order)}
        rerank_rank = {a: i for i, a in enumerate(rerank_order)}
        blend_order = sorted(cand, key=lambda a: _rrf(fused_rank[a], rerank_rank[a]), reverse=True)

        per = {}
        for cfg, order in (("fused", fused_order), ("rerank", rerank_order), ("blend", blend_order)):
            m = _metrics([grade_by_aid.get(a, 0) for a in order], corpus_grades)
            per[cfg] = m
            for k in METRIC_KEYS:
                agg[cfg][k] += m[k]
        rows.append((q["id"], q["query"], per))
        print(f"  [{n}/{len(gold)}] {q['id']} fused={per['fused']['ndcg@10']:.3f} "
              f"rerank={per['rerank']['ndcg@10']:.3f} blend={per['blend']['ndcg@10']:.3f}", flush=True)

    nq = len(gold)
    for cfg in agg:
        agg[cfg] = {k: round(v / nq, 4) for k, v in agg[cfg].items()}

    _write_md(rows, agg)
    print("\n== reranker eval (pool=fused top-{}, n={}) ==".format(DEPTH, nq))
    for cfg in ("fused", "rerank", "blend"):
        print(f"  {cfg:7s} " + "  ".join(f"{k}={agg[cfg][k]}" for k in METRIC_KEYS))
    print(f"  report -> {RESULT_MD}")
    return agg


def _write_md(rows, agg) -> None:
    lines = [
        "# Reranker eval (cross-encoder, gold-set)",
        "",
        f"Cross-encoder `{settings.reranker_model_id}` re-scoring fused top-{DEPTH} "
        "([candidates.json](../data/eval/candidates.json)). Offline, cùng gold-set "
        "([gold_queries.json](../data/eval/gold_queries.json)). Doc = `colour product_type. refined_description`.",
        "",
        "| config | nDCG@10 | P@10≥1 | P@10=2 | hit@1 | hit@5 | hit@10 | MRR2 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cfg in ("fused", "rerank", "blend"):
        a = agg[cfg]
        lines.append(f"| {cfg} | {a['ndcg@10']} | {a['p@10>=1']} | {a['p@10==2']} | "
                     f"{a['hit@1']} | {a['hit@5']} | {a['hit@10']} | {a['rr2']} |")
    lines += ["", "## Per-query nDCG@10", "", "| id | query | fused | rerank | blend |", "|---|---|---|---|---|"]
    for qid, qtext, per in rows:
        lines.append(f"| {qid} | {qtext} | {per['fused']['ndcg@10']:.3f} | "
                     f"{per['rerank']['ndcg@10']:.3f} | {per['blend']['ndcg@10']:.3f} |")
    lines.append("")
    with open(RESULT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
