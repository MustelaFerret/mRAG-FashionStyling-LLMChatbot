"""Production slot extractor (stage-1 DeBERTa BIO tagger + stage-2 hybrid linker), used as
an ENSEMBLE fallback: the gazetteer runs first (high precision on known surface forms), and
this fills only the enum fields the gazetteer missed — recovering rare colours / paraphrases
the lexicon can't (verified: held-out value-F1 0.68 -> 0.89). See md/slot_extractor_plan.md.

Linker: gazetteer surface map -> canonical-in-span -> colour via RGB nearest-neighbour
(MiniLM lacks colour semantics). No extra embedding model loaded.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.backend.core.config import settings
from src.backend.core.utils import normalize_text
from src.backend.retrieval.colour_rgb import ColourRGBLinker
from src.backend.services.attribute_gazetteer import AttributeGazetteer

_TOK = re.compile(r"[A-Za-z0-9/'-]+")
_AB2FIELD = {"COL": "colour_group", "FIT": "fit", "OCC": "occasion", "SEA": "seasonality"}


class SlotExtractor:
    def __init__(self, model_dir: str | None = None, valid: Dict[str, List[str]] | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_dir = model_dir or settings.slot_extractor_dir
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).eval().to(self.device)
        self.id2label = self.model.config.id2label
        self.maps = AttributeGazetteer.FIELD_MAPS
        self.valid = {f: set(v) for f, v in (valid or {}).items()}
        self.colour_rgb = ColourRGBLinker(list((valid or {}).get("colour_group", [])))

    @torch.no_grad()
    def _spans(self, query: str) -> Dict[str, str]:
        words = _TOK.findall(query)
        if not words:
            return {}
        enc = self.tok(words, is_split_into_words=True, return_tensors="pt",
                       truncation=True, max_length=settings.intent_max_length).to(self.device)
        pred = self.model(**enc).logits.argmax(-1)[0].cpu().tolist()
        wlab, seen = [], set()
        for pos, wid in enumerate(enc.word_ids(0)):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            wlab.append(self.id2label[pred[pos]])
        out, cur_ab, cur = {}, None, []

        def flush():
            if cur_ab in _AB2FIELD and cur:
                out.setdefault(_AB2FIELD[cur_ab], " ".join(cur))

        for w, lab in zip(words, wlab):
            if lab.startswith("B-"):
                flush(); cur_ab, cur = lab[2:], [w]
            elif lab.startswith("I-") and cur_ab == lab[2:]:
                cur.append(w)
            else:
                flush(); cur_ab, cur = None, []
        flush()
        return out

    def _link(self, field: str, surface: str) -> Optional[str]:
        s = normalize_text(surface).replace("'", "")
        if not s:
            return None
        s_pad = f" {s} "
        mp = self.maps.get(field, {})
        for key in sorted(mp, key=len, reverse=True):
            if f" {key} " in s_pad and mp[key] in self.valid.get(field, {mp[key]}):
                return mp[key]
        for v in self.valid.get(field, []):
            if f" {normalize_text(v)} " in s_pad:
                return v
        if field == "colour_group":
            return self.colour_rgb.link(surface)
        return None

    def fill_missing(self, query: str, current: Dict[str, str]) -> Dict[str, str]:
        """Return enum values for fields NOT already in `current` (ensemble fallback)."""
        add: Dict[str, str] = {}
        for field, surface in self._spans(query).items():
            if field in current or field in add:
                continue
            v = self._link(field, surface)
            if v:
                add[field] = v
        return add
