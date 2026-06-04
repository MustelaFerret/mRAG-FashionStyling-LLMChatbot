"""Đánh giá intent model trên test set đã validate — confusion + per-length + so sánh model.

Chạy local (sau khi tải model DeBERTa mới về model_cache/). So baseline RoBERTa cũ vs DeBERTa mới
trên CÙNG test set (đã qua LLM-judge) → công bằng. Đặc biệt soi accuracy câu DÀI per intent
để xác nhận hết length bias.

Run:
    conda activate mRAG
    python -m intent_analysis_model.evaluate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from intent_analysis_model import config as C


class IntentModelEvaluator:
    def __init__(self, model_dir: str, max_length: int = 128):
        self.model_dir = model_dir
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        lm_path = Path(model_dir) / "label_map.json"
        if lm_path.exists():
            label_map = json.loads(lm_path.read_text())
            self.id2label = {v: k for k, v in label_map.items()}
        else:
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def predict(self, queries):
        preds = []
        for i in range(0, len(queries), 64):
            batch = list(queries[i : i + 64])
            inp = self.tok(batch, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True)
            logits = self.model(**inp).logits
            for idx in logits.argmax(dim=-1).tolist():
                preds.append(self.id2label[idx])
        return preds


def evaluate(model_dir: str, max_length: int, test_csv: Path, label: str):
    df = pd.read_csv(test_csv)
    ev = IntentModelEvaluator(model_dir, max_length=max_length)
    df = df.copy()
    df["pred"] = ev.predict(df["query"].astype(str).tolist())

    print(f"\n{'='*64}\n{label}  (model={model_dir}, max_length={max_length})\n{'='*64}")
    labels = C.INTENT_LABELS
    print(classification_report(df["intent"], df["pred"], labels=labels, digits=4, zero_division=0))

    cm = confusion_matrix(df["intent"], df["pred"], labels=labels)
    print("confusion (row=true, col=pred):")
    print("        " + " ".join(f"{l[:6]:>7}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l[:7]:>7} " + " ".join(f"{cm[i][j]:>7}" for j in range(len(labels))))

    df["n_words"] = df["query"].astype(str).str.split().str.len()
    df["len_bucket"] = pd.cut(df["n_words"], [0, 4, 12, 100], labels=["short", "medium", "long"])
    print("\naccuracy theo độ dài:")
    for b in ["short", "medium", "long"]:
        sub = df[df["len_bucket"] == b]
        if len(sub):
            print(f"  {b:7s} n={len(sub):4d} acc={(sub['pred']==sub['intent']).mean():.4f}")
    print("accuracy LONG-query per intent (length-bias check):")
    longsub = df[df["len_bucket"] == "long"]
    for intent in labels:
        s = longsub[longsub["intent"] == intent]
        if len(s):
            print(f"  {intent:18s} n={len(s):3d} acc={(s['pred']==s['intent']).mean():.3f}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(C.DATA_DIR / "test.csv"))
    ap.add_argument("--old_model", default=str(C.REPO_ROOT / "model_cache" / "intent_classifier_roberta"))
    ap.add_argument("--new_model", default=str(C.OUTPUT_MODEL_DIR))
    args = ap.parse_args()

    test_csv = Path(args.test)
    if not test_csv.exists():
        raise SystemExit(f"test set chưa có: {test_csv} — chạy gen_data + validate_data trước")

    # model cũ mô phỏng đúng backend (max_length=32) để thấy length bias
    if Path(args.old_model).exists():
        evaluate(args.old_model, 32, test_csv, "OLD RoBERTa (max_length=32, như backend cũ)")
    # model mới
    if Path(args.new_model).exists():
        evaluate(args.new_model, C.MAX_LENGTH, test_csv, "NEW DeBERTa-v3 (max_length=128)")
    else:
        print(f"\n(model mới chưa có ở {args.new_model} — train trên Colab rồi tải về)")


if __name__ == "__main__":
    main()
