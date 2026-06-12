"""Gold-set eval for the NLU stage (DeBERTa intent + Qwen constrained filter extraction).

So far the NLU was only fixed reactively from logs; this measures it systematically. For
each hand-written query the gold says the intent and the must_filters that SHOULD be
extracted. A field absent from gold must NOT be extracted -- that catches the recurring
over-extraction bug (model inventing colour/occasion/product_type the user never said).

Metrics:
  intent_acc           - fraction of queries with correct intent_hint
  filter P / R / F1     - over (product_type lenient, enum fields exact)
  over-extraction       - spurious fields emitted where gold had none (count + per-query rate)
  under-extraction      - gold fields the model missed
  exact_match           - queries with correct intent AND every gold filter right AND nothing spurious

No Qdrant; loads catalog (for the filter vocab) + the real NLU models.
Run: PYTORCH_JIT=0 python -m src.scripts.indexing.eval_nlu
"""
from __future__ import annotations

import argparse
import json

from src.backend.core.config import BASE_DIR, settings
from src.backend.core.utils import normalize_text
from src.backend.retrieval.llm import QwenMultimodalService
from src.backend.services.catalog import FashionCatalog

GOLD_FILE = str(BASE_DIR / "data" / "eval" / "gold_nlu.json")
RESULT_MD = str(BASE_DIR / "md" / "exp_eval_nlu.md")
ENUM_FIELDS = ["colour_group", "fit", "occasion", "seasonality"]


def _pt_match(model_val: str, gold_words) -> bool:
    m = normalize_text(model_val)
    if not m:
        return False
    return any(normalize_text(w) in m or m in normalize_text(w) for w in gold_words)


def _enum_match(model_val: str, gold_vals) -> bool:
    m = normalize_text(model_val)
    return any(m == normalize_text(v) for v in gold_vals)


def main(gold_file: str = GOLD_FILE, result_md: str = RESULT_MD) -> dict:
    gold = json.load(open(gold_file, encoding="utf-8"))["queries"]
    catalog = FashionCatalog(settings.meta_file, settings.graph_file, settings.image_dir)
    vocab = {
        "product_type": list(getattr(catalog, "valid_product_types", []) or []),
        "colour_group": list(getattr(catalog, "valid_colors", []) or []),
        "fit": list(getattr(catalog, "valid_fits", []) or []),
        "occasion": list(getattr(catalog, "valid_occasions", []) or []),
        "seasonality": list(getattr(catalog, "valid_seasonalities", []) or []),
    }
    llm = QwenMultimodalService()

    tp = fp = fn = 0
    over_fields = 0
    intent_ok = 0
    exact = 0
    rows = []

    for i, g in enumerate(gold, 1):
        res = llm.analyze_user_query(g["query"], vocab=vocab)
        pred_intent = res.get("intent_hint", "")
        pred = {k: str(v).strip() for k, v in (res.get("must_filters") or {}).items() if str(v).strip()}
        goldf = g["filters"]

        i_ok = pred_intent == g["intent"]
        intent_ok += int(i_ok)

        q_tp = q_fp = q_fn = q_over = 0
        all_fields = set(goldf) | set(pred)
        for field in all_fields:
            in_gold = field in goldf
            in_pred = field in pred
            if in_pred and in_gold:
                ok = _pt_match(pred[field], goldf[field]) if field == "product_type" else _enum_match(pred[field], goldf[field])
                if ok:
                    q_tp += 1
                else:
                    q_fp += 1
                    q_fn += 1
            elif in_pred and not in_gold:
                q_fp += 1
                q_over += 1
            elif in_gold and not in_pred:
                q_fn += 1

        tp += q_tp; fp += q_fp; fn += q_fn; over_fields += q_over
        q_exact = i_ok and q_fp == 0 and q_fn == 0
        exact += int(q_exact)
        rows.append({
            "id": g["id"], "query": g["query"], "gold_intent": g["intent"], "pred_intent": pred_intent,
            "intent_ok": i_ok, "pred_filters": pred, "tp": q_tp, "fp": q_fp, "fn": q_fn, "over": q_over,
            "exact": q_exact,
        })
        flag = "OK " if q_exact else "BAD"
        print(f"  [{i}/{len(gold)}] {flag} {g['id']} intent={pred_intent}({'ok' if i_ok else 'X'}) "
              f"tp={q_tp} over={q_over} miss={q_fn} pred={pred}", flush=True)

    n = len(gold)
    prec = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    rec = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0
    agg = {
        "intent_acc": round(intent_ok / n, 4),
        "filter_precision": prec,
        "filter_recall": rec,
        "filter_f1": f1,
        "over_extracted_fields": over_fields,
        "under_extracted_fields": fn,
        "exact_match": round(exact / n, 4),
        "n": n,
    }
    _write_md(rows, agg, result_md)
    print("\n== NLU eval ==")
    print("  " + "  ".join(f"{k}={v}" for k, v in agg.items()))
    print(f"  report -> {result_md}")
    return agg


def _write_md(rows, agg, result_md: str = RESULT_MD) -> None:
    lines = [
        "# NLU gold-set eval (intent + filter extraction)",
        "",
        "Đo stage NLU (DeBERTa intent + Qwen-1.5B constrained extract) trên "
        "[gold_nlu.json](../data/eval/gold_nlu.json). Field absent trong gold = KHÔNG được extract "
        "(bắt over-extraction). product_type khớp lenient, enum khớp exact.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| intent_acc | {agg['intent_acc']} |",
        f"| filter precision | {agg['filter_precision']} |",
        f"| filter recall | {agg['filter_recall']} |",
        f"| filter F1 | {agg['filter_f1']} |",
        f"| over-extracted fields | {agg['over_extracted_fields']} |",
        f"| under-extracted fields | {agg['under_extracted_fields']} |",
        f"| **exact_match** | **{agg['exact_match']}** |",
        f"| n | {agg['n']} |",
        "",
        "## Per-query (BAD = sai intent / có over / có miss)",
        "",
        "| id | query | gold→pred intent | over | miss | pred filters |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        status = "" if r["exact"] else " ⚠"
        intent = f"{r['gold_intent']}→{r['pred_intent']}" + ("" if r["intent_ok"] else " ❌")
        lines.append(f"| {r['id']}{status} | {r['query']} | {intent} | {r['over']} | {r['fn']} | {r['pred_filters']} |")
    lines.append("")
    with open(result_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=GOLD_FILE)
    ap.add_argument("--out", default=RESULT_MD)
    a = ap.parse_args()
    main(a.gold, a.out)
