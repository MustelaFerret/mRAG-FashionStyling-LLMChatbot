"""NLU-inclusive end-to-end exam runner (the "scoring" stage), multi-turn.

Drives the full production pipeline (DeBERTa intent classifier -> deterministic intent rules ->
gazetteer extraction -> Qwen rewrite -> filtered hybrid retrieval -> rerank) via
FashionRAGService.prepare_chat, which yields both the result cards (items) and the analysis payload
(intent, must/must_not filters, retrieval_path, result ids) without running answer generation.

The exam is organised as CONVERSATIONS: each conversation is a list of turns that share one session
(so refinement, sticky anchors and cross-turn references behave as in production). Every turn is
scored against its own gold; a conversation passes iff all its turns pass. A turn context may carry
"anchor_from_prev": <i> to simulate the user clicking result #i of the previous turn.

Inputs  : tests/exam_cases.json   (de thi  -- conversations + turns)
          tests/answer_key.json   (dap an  -- gold predicates per turn id)
Outputs : tests/scores.json       (machine-readable per-turn observation + verdicts)
          tests/scores_report.md  (diem thi -- human report, Vietnamese)

Run with the project env (single-process embedded Qdrant -- no server may be running):
    D:/miniconda/envs/mRAG/python.exe tests/run_nlu_eval.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = REPO_ROOT / "tests"
EXAM_FILE = TESTS_DIR / "exam_cases.json"
KEY_FILE = TESTS_DIR / "answer_key.json"
SCORES_JSON = TESTS_DIR / "scores.json"
SCORES_MD = TESTS_DIR / "scores_report.md"

TIER_ORDER = ["easy", "medium", "hard"]


@dataclass
class Observation:
    """Everything the grader is allowed to inspect, captured from one pipeline turn."""

    intent: str = ""
    classifier_intent: str = ""
    direct_type: str = ""
    must: Dict[str, Any] = field(default_factory=dict)
    must_not: Dict[str, Any] = field(default_factory=dict)
    soft_product_type: str = ""
    retrieval_path: List[str] = field(default_factory=list)
    anchor_id: str = ""
    graph_candidate_count: int = 0
    n_suggestions: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def card_values(self, fieldname: str) -> List[str]:
        return [str(c.get(fieldname, "")).strip() for c in self.results if str(c.get(fieldname, "")).strip()]


class PipelineHarness:
    """Builds the production components once and replays exam conversations through prepare_chat."""

    def __init__(self) -> None:
        from src.backend.core.config import settings, setup_environment
        setup_environment()
        from src.backend.retrieval.embeddings import load_sparse_encoder
        from src.backend.retrieval.encoders import QueryEncoder, SigLIPEncoder
        from src.backend.retrieval.llm import QwenMultimodalService
        from src.backend.retrieval.qdrant import QdrantStore
        from src.backend.services.catalog import FashionCatalog
        from src.backend.services.personalization import PersonalizationStore
        from src.backend.services.rag_service import FashionRAGService
        from src.backend.services.session_manager import SessionStore

        self.settings = settings
        catalog = FashionCatalog(settings.meta_file, settings.graph_file, settings.image_dir,
                                 aux_graph_file=settings.aux_graph_file)
        store = QdrantStore(settings.db_path, settings.collection_name)
        sessions = SessionStore(settings.session_ttl_seconds, settings.max_session_context)
        sparse = load_sparse_encoder(settings.sparse_model_path) if os.path.exists(settings.sparse_model_path) else None
        embedding = QueryEncoder(siglip=SigLIPEncoder(), sparse=sparse)
        llm = QwenMultimodalService()
        personalization = PersonalizationStore(settings.personalization_dir, settings.meta_file)
        self.rag = FashionRAGService(embedding, store, llm, catalog, limit=5,
                                     personalization=personalization, sessions=sessions)

    def _load_image(self, rel_path: Optional[str]):
        if not rel_path:
            return None
        from PIL import Image
        abs_path = REPO_ROOT / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"image not found: {rel_path}")
        return Image.open(abs_path).convert("RGB")

    def _observe(self, items, log_payload, direct) -> Observation:
        dbg = log_payload.get("analysis_debug", {}) or {}
        classifier = (dbg.get("intent_classifier") or {}) if isinstance(dbg, dict) else {}
        anchor = str(log_payload.get("graph_anchor_id") or log_payload.get("active_anchor_id") or "")
        return Observation(
            intent=str(log_payload.get("intent_hint", "")),
            classifier_intent=str(classifier.get("intent", "")),
            direct_type=str((direct or {}).get("type", "")),
            must=dict(log_payload.get("must_filters", {}) or {}),
            must_not=dict(log_payload.get("must_not_filters", {}) or {}),
            soft_product_type=str(log_payload.get("soft_product_type", "") or ""),
            retrieval_path=list(log_payload.get("retrieval_path", []) or []),
            anchor_id=anchor,
            graph_candidate_count=int(log_payload.get("graph_candidate_count", 0) or 0),
            n_suggestions=len(log_payload.get("suggestions", []) or []),
            results=list(items or []),
        )

    def run_conversation(self, conv: Dict[str, Any]) -> List[tuple]:
        session_id = f"exam-{conv['id']}"
        out: List[tuple] = []
        prev_results: List[str] = []
        for ti, turn in enumerate(conv.get("turns", [])):
            ctx = dict(turn.get("context") or {})
            if "anchor_from_prev" in ctx:  # simulate clicking result #i of the previous turn
                idx = int(ctx.pop("anchor_from_prev"))
                if 0 <= idx < len(prev_results):
                    ctx["selected_anchor_id"] = prev_results[idx]
            try:
                image = self._load_image(ctx.get("image"))
                items, _p, log_payload, _hc, direct = self.rag.prepare_chat(
                    query=turn["query"], image=image, session_id=session_id,
                    request_id=turn["id"], started_at=time.perf_counter(),
                    confirmed_intent=ctx.get("confirmed_intent"), customer_id=ctx.get("customer_id"),
                    selected_anchor_id=ctx.get("selected_anchor_id"), no_anchor=bool(ctx.get("no_anchor", False)),
                )
                obs = self._observe(items, log_payload, direct)
            except Exception as exc:  # noqa: BLE001 -- record and continue the exam
                obs = Observation(error=f"{type(exc).__name__}: {exc}")
            out.append((turn, obs))
            prev_results = [c.get("article_id", "") for c in obs.results if c.get("article_id")]
        return out


class Grader:
    """Evaluates one predicate from the answer key against an Observation.

    Predicate schema (each is a dict with a ``check`` key):
      {"check": "intent", "equals": "color_variant"}            final routed intent_hint
      {"check": "intent", "in": ["similar_items", "color_variant"]}
      {"check": "classifier_intent", "equals": "graph_pairing"}  raw DeBERTa label (pre-rules)
      {"check": "direct", "equals": "chit_chat"}                 direct-response short circuit
      {"check": "must", "field": "product_type", "equals": "Dress"}      extracted hard/soft filter
      {"check": "must", "field": "occasion", "contains": "Lounge"}
      {"check": "must", "field": "colour_group", "present": true}
      {"check": "must", "field": "colour_group", "absent": true}
      {"check": "must_not", "field": "colour_group", "contains": "Black"}
      {"check": "must_not", "field": "colour_group", "present": true}
      {"check": "path", "contains": "cold_twin_compat_rrf"}      retrieval_path tag present
      {"check": "path", "absent": "relax_no_filters"}            tag NOT present
      {"check": "anchor", "resolved": true}                      a pairing/variant anchor was bound
      {"check": "graph_candidates", "min": 1}                    warm co-buy neighbours found
      {"check": "graph_candidates", "equals": 0}
      {"check": "results_min", "n": 1}
      {"check": "results_all_in", "field": "product_type", "values": ["Dress"]}
      {"check": "results_none_in", "field": "product_type", "values": ["Trousers"]}
      {"check": "results_frac_in", "field": "colour_group", "values": ["Red"], "min_frac": 0.6}
    """

    @staticmethod
    def _ci(s: str) -> str:
        return str(s).strip().lower()

    @classmethod
    def _values(cls, pred: Dict[str, Any]) -> List[str]:
        return [cls._ci(v) for v in pred.get("values", [])]

    def evaluate(self, pred: Dict[str, Any], obs: Observation) -> Dict[str, Any]:
        check = pred.get("check", "")
        method = getattr(self, f"_check_{check}", None)
        if method is None:
            return {"passed": False, "detail": f"unknown check '{check}'"}
        try:
            return method(pred, obs)
        except Exception as exc:  # noqa: BLE001
            return {"passed": False, "detail": f"grader error: {exc}"}

    # -- intent / routing -------------------------------------------------
    def _check_intent(self, pred, obs):
        if "equals" in pred:
            ok = self._ci(obs.intent) == self._ci(pred["equals"])
            return {"passed": ok, "detail": f"intent={obs.intent!r} expected={pred['equals']!r}"}
        wanted = [self._ci(v) for v in pred.get("in", [])]
        ok = self._ci(obs.intent) in wanted
        return {"passed": ok, "detail": f"intent={obs.intent!r} expected_in={pred.get('in')}"}

    def _check_classifier_intent(self, pred, obs):
        if "equals" in pred:
            ok = self._ci(obs.classifier_intent) == self._ci(pred["equals"])
            return {"passed": ok, "detail": f"classifier={obs.classifier_intent!r} expected={pred['equals']!r}"}
        wanted = [self._ci(v) for v in pred.get("in", [])]
        ok = self._ci(obs.classifier_intent) in wanted
        return {"passed": ok, "detail": f"classifier={obs.classifier_intent!r} expected_in={pred.get('in')}"}

    def _check_direct(self, pred, obs):
        ok = self._ci(obs.direct_type) == self._ci(pred["equals"])
        return {"passed": ok, "detail": f"direct={obs.direct_type!r} expected={pred['equals']!r}"}

    # -- extracted filters ------------------------------------------------
    def _check_must(self, pred, obs):
        return self._filter_check(pred, obs.must, "must")

    def _check_must_not(self, pred, obs):
        return self._filter_check(pred, obs.must_not, "must_not")

    def _filter_check(self, pred, store, label):
        fld = pred["field"]
        present_vals = self._normalize_filter_value(store.get(fld))
        joined = ", ".join(present_vals) if present_vals else "<none>"
        if pred.get("present"):
            return {"passed": bool(present_vals), "detail": f"{label}.{fld}=[{joined}]"}
        if pred.get("absent"):
            return {"passed": not present_vals, "detail": f"{label}.{fld}=[{joined}]"}
        if "equals" in pred:
            ok = any(v == self._ci(pred["equals"]) for v in present_vals)
            return {"passed": ok, "detail": f"{label}.{fld}=[{joined}] expected={pred['equals']!r}"}
        if "contains" in pred:
            needle = self._ci(pred["contains"])
            ok = any(needle in v for v in present_vals)
            return {"passed": ok, "detail": f"{label}.{fld}=[{joined}] contains={pred['contains']!r}"}
        return {"passed": False, "detail": f"{label}.{fld}: malformed predicate"}

    @staticmethod
    def _normalize_filter_value(raw) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            return [str(v).strip().lower() for v in raw if str(v).strip()]
        s = str(raw).strip().lower()
        return [s] if s else []

    # -- retrieval path / graph ------------------------------------------
    def _check_path(self, pred, obs):
        path_ci = [self._ci(p) for p in obs.retrieval_path]
        if "contains" in pred:
            needle = self._ci(pred["contains"])
            ok = any(needle in p for p in path_ci)
            return {"passed": ok, "detail": f"path={obs.retrieval_path} contains={pred['contains']!r}"}
        needle = self._ci(pred["absent"])
        ok = not any(needle in p for p in path_ci)
        return {"passed": ok, "detail": f"path={obs.retrieval_path} absent={pred['absent']!r}"}

    def _check_anchor(self, pred, obs):
        want = bool(pred.get("resolved", True))
        ok = bool(obs.anchor_id) == want
        return {"passed": ok, "detail": f"anchor_id={obs.anchor_id!r} resolved_expected={want}"}

    def _check_suggestions(self, pred, obs):
        ok = obs.n_suggestions >= int(pred.get("min", 1))
        return {"passed": ok, "detail": f"n_suggestions={obs.n_suggestions} min={pred.get('min', 1)}"}

    def _check_graph_candidates(self, pred, obs):
        if "min" in pred:
            ok = obs.graph_candidate_count >= int(pred["min"])
            return {"passed": ok, "detail": f"graph_candidates={obs.graph_candidate_count} min={pred['min']}"}
        ok = obs.graph_candidate_count == int(pred["equals"])
        return {"passed": ok, "detail": f"graph_candidates={obs.graph_candidate_count} expected={pred['equals']}"}

    # -- result-set properties -------------------------------------------
    def _check_results_min(self, pred, obs):
        ok = len(obs.results) >= int(pred["n"])
        return {"passed": ok, "detail": f"n_results={len(obs.results)} min={pred['n']}"}

    def _check_results_all_in(self, pred, obs):
        vals = self._values(pred)
        got = [self._ci(v) for v in obs.card_values(pred["field"])]
        if not got:
            return {"passed": False, "detail": f"{pred['field']}: no results"}
        bad = sorted({v for v in got if v not in vals})
        return {"passed": not bad, "detail": f"{pred['field']} all in {pred['values']}? offenders={bad or 'none'}"}

    def _check_results_none_in(self, pred, obs):
        vals = self._values(pred)
        got = [self._ci(v) for v in obs.card_values(pred["field"])]
        hit = sorted({v for v in got if v in vals})
        return {"passed": not hit, "detail": f"{pred['field']} none in {pred['values']}? offenders={hit or 'none'}"}

    def _check_results_frac_in(self, pred, obs):
        vals = self._values(pred)
        got = [self._ci(v) for v in obs.card_values(pred["field"])]
        if not got:
            return {"passed": False, "detail": f"{pred['field']}: no results"}
        frac = sum(1 for v in got if v in vals) / len(got)
        ok = frac >= float(pred["min_frac"])
        return {"passed": ok, "detail": f"{pred['field']} frac_in {pred['values']}={frac:.2f} min={pred['min_frac']}"}


class ExamRunner:
    def __init__(self) -> None:
        self.exam = json.loads(EXAM_FILE.read_text(encoding="utf-8"))
        self.key = json.loads(KEY_FILE.read_text(encoding="utf-8"))
        self.answers: Dict[str, List[dict]] = self.key.get("answers", {})
        self.grader = Grader()

    def run(self) -> Dict[str, Any]:
        harness = PipelineHarness()
        conversations = self.exam.get("conversations", [])
        conv_records: List[Dict[str, Any]] = []
        for ci, conv in enumerate(conversations, 1):
            sys.stdout.write(f"[{ci}/{len(conversations)}] {conv['id']} ({conv.get('tier')}): {conv.get('title','')}\n")
            sys.stdout.flush()
            turn_results = harness.run_conversation(conv)
            turn_records = []
            for turn, obs in turn_results:
                preds = self.answers.get(turn["id"], [])
                verdicts = [{"pred": p, **self.grader.evaluate(p, obs)} for p in preds]
                passed = bool(verdicts) and all(v["passed"] for v in verdicts) and not obs.error
                turn_records.append(self._turn_record(turn, obs, verdicts, passed))
            conv_records.append({
                "id": conv["id"], "tier": conv.get("tier", "?"), "title": conv.get("title", ""),
                "passed": all(t["passed"] for t in turn_records) and bool(turn_records),
                "turns": turn_records,
            })
        summary = self._summarize(conv_records)
        SCORES_JSON.write_text(json.dumps({"summary": summary, "conversations": conv_records},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
        SCORES_MD.write_text(self._report_md(summary, conv_records), encoding="utf-8")
        o = summary["overall"]
        sys.stdout.write(f"\nConversations {o['conv_passed']}/{o['conv_total']} ({o['conv_pass_rate']:.1%}) | "
                         f"turns {o['turn_passed']}/{o['turn_total']} ({o['turn_pass_rate']:.1%})\n")
        sys.stdout.write(f"Wrote {SCORES_JSON.name} and {SCORES_MD.name}\n")
        return summary

    @staticmethod
    def _turn_record(turn, obs: Observation, verdicts, passed) -> Dict[str, Any]:
        return {
            "id": turn["id"], "query": turn["query"], "passed": passed,
            "n_pred": len(verdicts), "n_pred_pass": sum(1 for v in verdicts if v["passed"]),
            "error": obs.error,
            "observed": {
                "intent": obs.intent, "classifier_intent": obs.classifier_intent,
                "direct_type": obs.direct_type, "must": obs.must, "must_not": obs.must_not,
                "retrieval_path": obs.retrieval_path, "anchor_id": obs.anchor_id,
                "graph_candidate_count": obs.graph_candidate_count, "n_results": len(obs.results),
                "result_types": obs.card_values("product_type"),
                "result_colours": obs.card_values("colour_group"),
                "result_occasions": obs.card_values("occasion"),
            },
            "verdicts": verdicts,
        }

    def _summarize(self, convs) -> Dict[str, Any]:
        out: Dict[str, Any] = {"by_tier": {}}
        for tier in TIER_ORDER:
            cs = [c for c in convs if c["tier"] == tier]
            if not cs:
                continue
            turns = [t for c in cs for t in c["turns"]]
            out["by_tier"][tier] = {
                "conv_total": len(cs), "conv_passed": sum(1 for c in cs if c["passed"]),
                "turn_total": len(turns), "turn_passed": sum(1 for t in turns if t["passed"]),
            }
        all_turns = [t for c in convs for t in c["turns"]]
        out["overall"] = {
            "conv_total": len(convs), "conv_passed": sum(1 for c in convs if c["passed"]),
            "conv_pass_rate": (sum(1 for c in convs if c["passed"]) / len(convs)) if convs else 0.0,
            "turn_total": len(all_turns), "turn_passed": sum(1 for t in all_turns if t["passed"]),
            "turn_pass_rate": (sum(1 for t in all_turns if t["passed"]) / len(all_turns)) if all_turns else 0.0,
        }
        return out

    def _report_md(self, summary, convs) -> str:
        L: List[str] = []
        L.append("# Điểm thi NLU end-to-end (multi-turn)")
        L.append("")
        L.append("Mỗi case là một **hội thoại nhiều lượt** chung session, chạy qua toàn bộ pipeline "
                 "production (`FashionRAGService.prepare_chat`): intent classifier → intent rules → "
                 "gazetteer → Qwen rewrite → filtered hybrid retrieval → rerank. Chấm **từng lượt**; "
                 "một hội thoại đạt chỉ khi **mọi lượt** của nó đạt.")
        L.append("")
        o = summary["overall"]
        L.append(f"- Đề thi: `tests/exam_cases.json` · Đáp án: `tests/answer_key.json`")
        L.append(f"- **Hội thoại: {o['conv_passed']}/{o['conv_total']} ({o['conv_pass_rate']:.1%})** · "
                 f"**Lượt: {o['turn_passed']}/{o['turn_total']} ({o['turn_pass_rate']:.1%})**")
        L.append("")
        L.append("## Tổng hợp theo độ khó")
        L.append("")
        L.append("| Tier | Hội thoại đạt | Lượt đạt |")
        L.append("|---|---|---|")
        for tier in TIER_ORDER:
            t = summary["by_tier"].get(tier)
            if not t:
                continue
            L.append(f"| {tier} | {t['conv_passed']}/{t['conv_total']} | {t['turn_passed']}/{t['turn_total']} |")
        L.append("")
        L.append("## Chi tiết từng hội thoại")
        L.append("")
        for tier in TIER_ORDER:
            cs = [c for c in convs if c["tier"] == tier]
            if not cs:
                continue
            L.append(f"### Tier: {tier}")
            L.append("")
            for c in cs:
                mark = "PASS" if c["passed"] else "FAIL"
                L.append(f"#### [{mark}] {c['id']} — {c['title']}")
                for t in c["turns"]:
                    tmark = "PASS" if t["passed"] else "FAIL"
                    ob = t["observed"]
                    L.append(f"- **[{tmark}] {t['id']}** `{t['query']}`")
                    if t["error"]:
                        L.append(f"  - LỖI: {t['error']}")
                    L.append(f"  - intent=`{ob['intent']}` (cls=`{ob['classifier_intent']}`"
                             + (f", direct=`{ob['direct_type']}`" if ob['direct_type'] else "")
                             + (f", anchor={ob['anchor_id']}" if ob['anchor_id'] else "") + ")"
                             + f" · must={ob['must']} · must_not={ob['must_not']}")
                    L.append(f"  - path={ob['retrieval_path']} · {ob['n_results']} kết quả · "
                             f"types={ob['result_types'][:6]}")
                    fails = [v for v in t["verdicts"] if not v["passed"]]
                    if fails:
                        for v in fails:
                            L.append(f"  - ✗ `{v['pred'].get('check')}`: {v['detail']}")
                L.append("")
        return "\n".join(L)


if __name__ == "__main__":
    ExamRunner().run()
