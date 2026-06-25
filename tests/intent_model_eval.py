"""Test the intent classifier itself: raw finetuned DeBERTa vs DeBERTa + the hand-written rule
layer (_apply_intent_rules). Answers the question "do the override rules help or hurt the model,
and is the model good enough to trust on its own?" before any redesign of the rule layer.

This isolates the LLM-layer intent decision (text-only); it does NOT touch retrieval, and it does
NOT exercise the rag_service context-injection layer (image/anchor/session), which is a separate,
legitimate concern.

    D:/miniconda/envs/mRAG/python.exe tests/intent_model_eval.py

Inputs : tests/intent_labelset.json
Outputs: tests/intent_scores.json, tests/intent_model_report.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = REPO_ROOT / "tests"
LABEL_FILE = TESTS_DIR / "intent_labelset.json"
SCORES_JSON = TESTS_DIR / "intent_scores.json"
REPORT_MD = TESTS_DIR / "intent_model_report.md"

CLASSES = ["similar_items", "graph_pairing", "color_variant", "composite_intent", "chit_chat"]


@dataclass
class Row:
    query: str
    gold: str
    hard: bool
    model: str
    confidence: float
    rule_final: str
    rule_applied: str

    @property
    def model_ok(self) -> bool:
        return self.model == self.gold

    @property
    def rule_ok(self) -> bool:
        return self.rule_final == self.gold

    @property
    def disagree(self) -> bool:
        return self.model != self.rule_final


class IntentModelEval:
    def __init__(self) -> None:
        from src.backend.core.config import setup_environment
        setup_environment()
        from src.backend.retrieval.llm import QwenMultimodalService
        self.llm = QwenMultimodalService()
        if self.llm.intent_model is None:
            raise RuntimeError("intent classifier did not load (settings.intent_classifier_dir)")

    def _classify(self, query: str) -> Row:
        res = self.llm._classify_intent_local(query, [], anchor_item=None)
        model_intent = str(res.get("intent", ""))
        conf = float(res.get("confidence", 0.0))
        final, dbg = self.llm._apply_intent_rules(query, model_intent)
        return Row(query=query, gold="", hard=False, model=model_intent, confidence=conf,
                   rule_final=str(final), rule_applied=str(dbg.get("rule_applied", "")))

    def run(self) -> Dict[str, Any]:
        spec = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
        rows: List[Row] = []
        for item in spec["labels"]:
            r = self._classify(item["query"])
            r.gold = item["gold"]
            r.hard = bool(item.get("hard", False))
            rows.append(r)
            sys.stdout.write(f"{'OK ' if r.model_ok else 'ERR'} model={r.model:16} "
                             f"rule={r.rule_final:16} gold={r.gold:16} c={r.confidence:.2f}  {r.query}\n")
        summary = self._summarize(rows)
        SCORES_JSON.write_text(json.dumps({"summary": summary, "rows": [vars(r) for r in rows]},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT_MD.write_text(self._report(summary, rows), encoding="utf-8")
        m, ru, n = summary["model_acc"], summary["rule_acc"], summary["n"]
        sys.stdout.write(f"\nModel-alone {summary['model_correct']}/{n} ({m:.1%}) | "
                         f"Model+rules {summary['rule_correct']}/{n} ({ru:.1%})\n")
        sys.stdout.write(f"Rules HURT {summary['rules_hurt']} | Rules HELP {summary['rules_help']} | "
                         f"disagreements {summary['disagreements']}/{n} ({summary['disagree_rate']:.1%})\n")
        sys.stdout.write(f"Wrote {REPORT_MD.name} and {SCORES_JSON.name}\n")
        return summary

    def _summarize(self, rows: List[Row]) -> Dict[str, Any]:
        n = len(rows)
        model_correct = sum(r.model_ok for r in rows)
        rule_correct = sum(r.rule_ok for r in rows)
        rules_hurt = sum(1 for r in rows if r.model_ok and not r.rule_ok)
        rules_help = sum(1 for r in rows if not r.model_ok and r.rule_ok)
        disagree = [r for r in rows if r.disagree]
        per_class: Dict[str, Dict[str, Any]] = {}
        for c in CLASSES:
            rs = [r for r in rows if r.gold == c]
            if not rs:
                continue
            per_class[c] = {
                "n": len(rs),
                "model_correct": sum(r.model_ok for r in rs),
                "rule_correct": sum(r.rule_ok for r in rs),
            }
        conf_ok = [r.confidence for r in rows if r.model_ok]
        conf_bad = [r.confidence for r in rows if not r.model_ok]
        return {
            "n": n,
            "model_correct": model_correct, "model_acc": model_correct / n if n else 0.0,
            "rule_correct": rule_correct, "rule_acc": rule_correct / n if n else 0.0,
            "rules_hurt": rules_hurt, "rules_help": rules_help,
            "disagreements": len(disagree), "disagree_rate": len(disagree) / n if n else 0.0,
            "per_class": per_class,
            "conf_mean_model_correct": sum(conf_ok) / len(conf_ok) if conf_ok else 0.0,
            "conf_min_model_correct": min(conf_ok) if conf_ok else 0.0,
            "conf_mean_model_wrong": sum(conf_bad) / len(conf_bad) if conf_bad else 0.0,
            "conf_max_model_wrong": max(conf_bad) if conf_bad else 0.0,
        }

    def _report(self, s: Dict[str, Any], rows: List[Row]) -> str:
        L: List[str] = []
        L.append("# Kiểm định tầng intent: DeBERTa trần vs DeBERTa + rules")
        L.append("")
        L.append("Cô lập quyết định intent ở tầng LLM (text-only): so `_classify_intent_local` (model trần) "
                 "với `_apply_intent_rules` (model + hand-rules). Không đụng retrieval, không đụng tầng "
                 "bơm-ngữ-cảnh ở rag_service (ảnh/anchor/session).")
        L.append("")
        L.append(f"- **Model trần: {s['model_correct']}/{s['n']} ({s['model_acc']:.1%})**")
        L.append(f"- **Model + rules: {s['rule_correct']}/{s['n']} ({s['rule_acc']:.1%})**")
        delta = s["rule_acc"] - s["model_acc"]
        L.append(f"- Hiệu ứng rules trên net: **{'+' if delta >= 0 else ''}{delta:.1%}** "
                 f"(rules CỨU {s['rules_help']} câu, rules PHÁ {s['rules_hurt']} câu)")
        L.append(f"- Bất đồng model≠rules: **{s['disagreements']}/{s['n']} ({s['disagree_rate']:.1%})** "
                 f"→ tỉ lệ phải gọi LLM-trọng-tài nếu theo phương án đó")
        L.append(f"- Confidence khi model ĐÚNG: mean {s['conf_mean_model_correct']:.2f}, "
                 f"min {s['conf_min_model_correct']:.2f}")
        L.append(f"- Confidence khi model SAI: mean {s['conf_mean_model_wrong']:.2f}, "
                 f"max {s['conf_max_model_wrong']:.2f}  "
                 f"(nếu max thấp → có thể gate theo confidence)")
        L.append("")
        L.append("## Theo lớp")
        L.append("")
        L.append("| Lớp (gold) | n | Model đúng | Model+rules đúng |")
        L.append("|---|---|---|---|")
        for c, v in s["per_class"].items():
            L.append(f"| {c} | {v['n']} | {v['model_correct']}/{v['n']} | {v['rule_correct']}/{v['n']} |")
        L.append("")
        L.append("## Các câu rules PHÁ (model đúng → rule sai)")
        L.append("")
        hurt = [r for r in rows if r.model_ok and not r.rule_ok]
        if hurt:
            L.append("| Query | model (đúng) | rule_applied | rule_final (sai) | conf |")
            L.append("|---|---|---|---|---|")
            for r in hurt:
                L.append(f"| {r.query} | {r.model} | `{r.rule_applied}` | {r.rule_final} | {r.confidence:.2f} |")
        else:
            L.append("(không có)")
        L.append("")
        L.append("## Các câu rules CỨU (model sai → rule đúng)")
        L.append("")
        help_rows = [r for r in rows if not r.model_ok and r.rule_ok]
        if help_rows:
            L.append("| Query | model (sai) | rule_applied | rule_final (đúng) | conf |")
            L.append("|---|---|---|---|---|")
            for r in help_rows:
                L.append(f"| {r.query} | {r.model} | `{r.rule_applied}` | {r.rule_final} | {r.confidence:.2f} |")
        else:
            L.append("(không có)")
        L.append("")
        L.append("## Câu cả hai cùng sai")
        L.append("")
        both = [r for r in rows if not r.model_ok and not r.rule_ok]
        if both:
            L.append("| Query | gold | model | rule_final | conf |")
            L.append("|---|---|---|---|---|")
            for r in both:
                L.append(f"| {r.query} | {r.gold} | {r.model} | {r.rule_final} | {r.confidence:.2f} |")
        else:
            L.append("(không có)")
        return "\n".join(L)


if __name__ == "__main__":
    IntentModelEval().run()
