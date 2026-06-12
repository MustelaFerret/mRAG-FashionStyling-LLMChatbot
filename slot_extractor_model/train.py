"""Fine-tune DeBERTa-v3-base for slot BIO token classification. Runs locally or on Colab
(`!python -m slot_extractor_model.train` after uploading the folder + data/bio_*.jsonl).

Mirrors the intent-classifier recipe gotchas (md/intent_classifier_rebuild.md):
  - fp32 only — DeBERTa disentangled attention overflows in bf16/fp16 (val loss -> nan).
  - pin `transformers==4.46.3` — 5.x dropped the legacy gamma/beta rename, breaking backbone load.
  - max_length 64, cosine schedule, warmup, label smoothing, early stopping on val seqeval-F1.
Label alignment: only the FIRST sub-token of each word gets the tag; continuations + specials = -100.
"""
from __future__ import annotations

import json

import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from slot_extractor_model import config as C
from slot_extractor_model.build_bio import LABELS

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def _load(path):
    rows = [json.loads(l) for l in open(C.DATA_DIR / path, encoding="utf-8")]
    return Dataset.from_list([{"tokens": r["tokens"], "tags": [LABEL2ID[t] for t in r["tags"]]} for r in rows])


def main() -> None:
    tok = AutoTokenizer.from_pretrained(C.BACKBONE)

    def align(batch):
        enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=C.MAX_LENGTH)
        labels = []
        for i, tags in enumerate(batch["tags"]):
            word_ids = enc.word_ids(i)
            prev, seq = None, []
            for wid in word_ids:
                if wid is None:
                    seq.append(-100)
                elif wid != prev:
                    seq.append(tags[wid])
                else:
                    seq.append(-100)
                prev = wid
            labels.append(seq)
        enc["labels"] = labels
        return enc

    ds = {sp: _load(f"bio_{sp}.jsonl").map(align, batched=True, remove_columns=["tokens", "tags"])
          for sp in ("train", "val", "test")}

    model = AutoModelForTokenClassification.from_pretrained(
        C.BACKBONE, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        true_l, true_p = [], []
        for pred, lab in zip(preds, p.label_ids):
            true_l.append([ID2LABEL[l] for l in lab if l != -100])
            true_p.append([ID2LABEL[pr] for pr, l in zip(pred, lab) if l != -100])
        return {"f1": f1_score(true_l, true_p)}

    args = TrainingArguments(
        output_dir=str(C.OUTPUT_MODEL_DIR) + "_ckpt",
        learning_rate=2e-5, num_train_epochs=12,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        warmup_ratio=0.1, lr_scheduler_type="cosine", weight_decay=0.01, label_smoothing_factor=0.1,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1", greater_is_better=True, fp16=False, bf16=False,
        logging_steps=50, report_to="none", seed=42,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
        data_collator=DataCollatorForTokenClassification(tok), compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    print("\n== TEST ==")
    pred = trainer.predict(ds["test"])
    print("test f1:", pred.metrics.get("test_f1"))
    preds = np.argmax(pred.predictions, axis=2)
    tl, tp = [], []
    for pr, lab in zip(preds, pred.label_ids):
        tl.append([ID2LABEL[l] for l in lab if l != -100])
        tp.append([ID2LABEL[x] for x, l in zip(pr, lab) if l != -100])
    print(classification_report(tl, tp))

    C.OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(C.OUTPUT_MODEL_DIR))
    tok.save_pretrained(str(C.OUTPUT_MODEL_DIR))
    print(f"[train] saved -> {C.OUTPUT_MODEL_DIR}")


if __name__ == "__main__":
    main()
