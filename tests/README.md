# NLU end-to-end exam

A reproducible, NLU-inclusive exam that drives the **full production pipeline** (intent classifier →
intent rules → gazetteer extraction → Qwen rewrite → filtered hybrid retrieval → rerank), unlike
`src/backend/scripts/indexing/eval_goldset.py` which queries Qdrant directly and bypasses NLU.

The exam is organised as **multi-turn conversations**: each case is a list of turns sharing one
session, so refinement, sticky anchors and cross-turn references behave as in production. Every
turn is scored against its own gold; a conversation passes iff all its turns pass.

| File | Role |
|---|---|
| `exam_cases.json` | **Questions** — 20 conversations / 31 turns, easy → hard. A turn context may carry `anchor_from_prev: i` to simulate clicking result #i of the previous turn |
| `answer_key.json` | **Gold** — predicates keyed by **turn id** (expected intent, extracted must/must_not filters, result properties, retrieval path, anchor) |
| `run_nlu_eval.py` | **Runner/grader** — `PipelineHarness` (conversation + anchor-from-prev) + `Grader` + `ExamRunner` |
| `scores_report.md` | **Scores** — human-readable report (Vietnamese), per turn + conversation + tier |
| `scores.json` | machine-readable per-turn observation + verdicts |

Sibling eval `intent_model_eval.py` + `intent_labelset.json` isolates the intent classifier itself
(DeBERTa-alone vs DeBERTa+rules) — see `md/audit_intent_model_vs_rules.md`.

## Run

```
D:/miniconda/envs/mRAG/python.exe tests/run_nlu_eval.py
```

Requirements: project env `mRAG` (loads SigLIP + Qwen + reranker on GPU), `PYTORCH_JIT=0`.
Qdrant runs single-process embedded, so **no API server may be running at the same time**.

## Design notes
- The runner calls `FashionRAGService.prepare_chat`, which returns both result cards and the analysis
  `log_payload` (intent, filters, retrieval_path, result ids) **without** running answer generation.
- The answer key encodes **correct expected behaviour** (the spec), not the current output — a failing
  case is a genuine finding. See `md/test_nlu_eval_harness.md` for the latest run and open findings.
- Predicate schema is documented in the `Grader` docstring in `run_nlu_eval.py`.
