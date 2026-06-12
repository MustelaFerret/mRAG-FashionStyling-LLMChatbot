"""Stage-2 normalization: map a recognised span (the tagger already decided its FIELD) to a
canonical enum value. Gazetteer lookup first (known surface forms); for an OOV span, embed it
and take the nearest canonical value WITHIN THAT FIELD ONLY. This is well-posed precisely
because the field is fixed by the tagger — the earlier whole-query embedding fallback failed
because it had to guess the field too. Embedding (MiniLM) is optional.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.backend.core.utils import normalize_text
from src.backend.services.attribute_gazetteer import AttributeGazetteer
from slot_extractor_model.colour_rgb import ColourRGBLinker

_FIELD_DESC = {  # richer text for embedding the canonical occasion/season/fit values
    "occasion": {
        "Party/Evening/Wedding": "party evening wedding cocktail gala", "Sport/Active/Workout": "sport gym workout exercise",
        "Office/Workwear": "office work business interview meeting", "Lounge/Sleep/Nightwear": "lounge sleep pyjamas home",
        "Beach/Swimwear": "beach swim pool holiday", "Casual/Everyday": "casual everyday daily",
        "Intimate/Underwear": "underwear lingerie intimate", "Outdoor/Adventure": "outdoor hiking adventure skiing",
    },
    "seasonality": {"Spring/Summer": "summer spring hot warm", "Autumn/Winter": "winter autumn cold freezing",
                    "Core/All-year": "all year round any season"},
    "fit": {"Tight/Skinny/Bodycon": "tight skinny bodycon", "Oversized/Loose/Relaxed": "oversized loose baggy",
            "Slim/Tailored": "slim tailored", "Regular/Straight": "regular straight"},
}


class SlotLinker:
    def __init__(self, valid: Dict[str, List[str]], encoder=None, rgb_colour=True):
        self.maps = AttributeGazetteer.FIELD_MAPS  # field -> {surface: canonical}
        self.valid = {f: set(v) for f, v in valid.items()}
        self.encoder = encoder
        # colour normalised by RGB nearest-neighbour (MiniLM lacks colour semantics)
        self.colour_rgb = ColourRGBLinker(list(valid.get("colour_group", []))) if rgb_colour else None
        self._emb_vals: Dict[str, List[str]] = {}
        self._emb_mat: Dict[str, np.ndarray] = {}
        if encoder is not None:
            for field in self.maps:
                vals = [v for v in valid.get(field, []) if v != "Unknown"]
                if not vals:
                    continue
                if field == "colour_group":
                    texts = [v.lower() for v in vals]
                else:
                    texts = [_FIELD_DESC.get(field, {}).get(v, v.lower()) for v in vals]
                self._emb_vals[field] = vals
                self._emb_mat[field] = encoder.encode_texts(texts, show_progress=False)

    def link(self, field: str, surface: str) -> Optional[str]:
        s = normalize_text(surface).replace("'", "")
        if not s:
            return None
        s_pad = f" {s} "
        # 1) gazetteer: longest known surface form contained in the span
        mp = self.maps.get(field, {})
        for key in sorted(mp, key=len, reverse=True):
            if f" {key} " in s_pad and mp[key] in self.valid.get(field, {mp[key]}):
                return mp[key]
        # 2) canonical value itself appearing in the span
        for v in self.valid.get(field, []):
            if f" {normalize_text(v)} " in s_pad:
                return v
        # 2b) colour: RGB nearest-neighbour (handles rare colour words generically)
        if field == "colour_group" and self.colour_rgb is not None:
            return self.colour_rgb.link(surface)
        # 3) embedding nearest canonical within this field (field already fixed by tagger)
        if field in self._emb_mat:
            q = self.encoder.encode_texts([s], show_progress=False)[0]
            sims = self._emb_mat[field] @ q
            return self._emb_vals[field][int(sims.argmax())]
        return None
