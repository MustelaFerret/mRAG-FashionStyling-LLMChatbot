"""Evaluate the trained tagger + linker on the FROZEN held-out NLU gold (and easy/hard),
extracting {field: canonical} and scoring vs gold exactly like eval_nlu, head-to-head with
the gazetteer baseline. Run after training (model in OUTPUT_MODEL_DIR).

  PYTORCH_JIT=0 python -m slot_extractor_model.eval_slot --gold data/eval/gold_nlu_heldout.json --embed
"""
from __future__ import annotations

import argparse
import json
import re

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from slot_extractor_model import config as C
from slot_extractor_model.build_bio import AB
from slot_extractor_model.slot_linker import SlotLinker
from src.backend.core.config import BASE_DIR
from src.backend.core.utils import normalize_text
from src.backend.services.attribute_gazetteer import AttributeGazetteer
from src.backend.services.catalog import FashionCatalog
from src.backend.core.config import settings

_TOK = re.compile(r"[A-Za-z0-9/'-]+")
AB2FIELD = {v: k for k, v in AB.items()}


class Tagger:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(str(C.OUTPUT_MODEL_DIR))
        self.model = AutoModelForTokenClassification.from_pretrained(str(C.OUTPUT_MODEL_DIR)).eval().to(self.device)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def spans(self, query: str):
        words = _TOK.findall(query)
        if not words:
            return {}
        enc = self.tok(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=C.MAX_LENGTH).to(self.device)
        pred = self.model(**enc).logits.argmax(-1)[0].cpu().tolist()
        word_ids = enc.word_ids(0)
        wlab, seen = [], set()
        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            wlab.append(self.id2label[pred[pos]])
        # decode BIO -> {field: surface}
        out, cur_ab, cur = {}, None, []
        def flush():
            if cur_ab and cur_ab in AB2FIELD:
                out.setdefault(AB2FIELD[cur_ab], " ".join(cur))
        for w, lab in zip(words, wlab):
            if lab.startswith("B-"):
                flush(); cur_ab, cur = lab[2:], [w]
            elif lab.startswith("I-") and cur_ab == lab[2:]:
                cur.append(w)
            else:
                flush(); cur_ab, cur = None, []
        flush()
        return out


def _pt_ok(v, gold):  # unused for slots; kept parity
    return False


def _enum_ok(v, gold):
    m = normalize_text(v)
    return any(m == normalize_text(g) for g in gold)


def _score(pred, goldf):
    """value-level: field present AND canonical value correct."""
    tp = fp = fn = 0
    for field in C.FIELDS:
        ig, ip = field in goldf, field in pred and pred[field]
        if ip and ig:
            ok = _enum_ok(pred[field], goldf[field]); tp += ok; fp += (not ok); fn += (not ok)
        elif ip and not ig:
            fp += 1
        elif ig and not ip:
            fn += 1
    return tp, fp, fn


def _score_recog(pred_fields, gold_fields):
    """recognition-level: did we detect the right FIELD, ignoring the canonical value."""
    tp = len(pred_fields & gold_fields)
    return tp, len(pred_fields - gold_fields), len(gold_fields - pred_fields)


def _prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


GOLD_SETS = ["data/eval/gold_nlu.json", "data/eval/gold_nlu_hard.json", "data/eval/gold_nlu_heldout.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", nargs="*", default=GOLD_SETS)
    ap.add_argument("--embed", action="store_true", help="MiniLM linker for non-colour OOV spans")
    a = ap.parse_args()

    catalog = FashionCatalog(settings.meta_file, settings.graph_file, settings.image_dir)
    valid = {"colour_group": list(catalog.valid_colors), "fit": list(catalog.valid_fits),
             "occasion": list(catalog.valid_occasions), "seasonality": list(catalog.valid_seasonalities)}
    encoder = None
    if a.embed:
        from slot_extractor_model.minilm_encoder import MiniLMEncoder
        encoder = MiniLMEncoder()
    tagger = Tagger()
    linker = SlotLinker(valid, encoder=encoder)
    gaz = AttributeGazetteer()

    def _load(path):
        p = BASE_DIR / path
        if str(path).endswith(".jsonl"):
            out = []
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                filt = {f: [r[f + "_canon"]] for f in C.FIELDS if r.get(f + "_canon", "none") != "none"}
                out.append({"query": r["query"], "filters": filt})
            return out
        return json.load(open(p, encoding="utf-8"))["queries"]

    for gf_path in a.gold:
        gold = _load(gf_path)
        recog = [0, 0, 0]          # tagger field-detection (ignore value)
        tval = [0, 0, 0]           # tagger value-level
        gval = [0, 0, 0]           # gazetteer value-level
        eval_ = [0, 0, 0]          # ENSEMBLE: gazetteer first + tagger fallback for missing fields
        for g in gold:
            goldf = {f: g["filters"][f] for f in C.FIELDS if f in g["filters"]}
            gold_fields = set(goldf)
            spans = tagger.spans(g["query"])
            pred = {f: v for f, surf in spans.items() if (v := linker.link(f, surf))}
            gz = gaz.extract(g["query"], valid)
            ens = dict(gz)
            for f, v in pred.items():
                ens.setdefault(f, v)   # gazetteer wins; tagger fills only what it missed
            for acc, (tp, fp, fn) in (
                (recog, _score_recog(set(spans), gold_fields)),
                (tval, _score(pred, goldf)),
                (gval, _score(gz, goldf)),
                (eval_, _score(ens, goldf)),
            ):
                acc[0] += tp; acc[1] += fp; acc[2] += fn
        name = gf_path.split("/")[-1]
        print(f"\n== {name} (n={len(gold)}, embed={a.embed}) ==")
        for lbl, acc in (("tagger RECOG ", recog), ("tagger VALUE ", tval),
                         ("gazetteer VAL", gval), ("ENSEMBLE  VAL", eval_)):
            P, R, F = _prf(*acc)
            print(f"  {lbl} P={P:.3f} R={R:.3f} F1={F:.3f} (tp={acc[0]} fp={acc[1]} fn={acc[2]})")


if __name__ == "__main__":
    main()
