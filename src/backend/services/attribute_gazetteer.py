"""Deterministic surface-form -> enum extraction for the closed-vocabulary filter fields
(colour_group, fit, occasion, seasonality).

Audit (md/audit_nlu.md) showed the 1.5B LLM under constrained decoding both *dropped*
these fields and *hallucinated wrong* enum values (winter -> Spring/Summer, wedding ->
Beach). For a fixed vocabulary a keyword gazetteer is far more reliable, so enum
extraction is moved here; the LLM keeps only product_type (open-vocab) + the query rewrite.

A hard gold-set (md/exp_eval_nlu_hard.md) then exposed the exact-match brittleness, so the
lexicon is broadened with common colour synonyms + occasion/season paraphrases, plus a
negation guard ("jeans that aren't too skinny" must NOT yield fit=skinny). Targets are
validated against the live vocab, so an unknown enum is never emitted. Longest surface form
wins within a field. The genuinely-novel surface form (rare coinage, typo) is the residual
gap that an embedding fallback or a learned slot model would close.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.backend.core.utils import normalize_text

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# apostrophes are stripped by normalisation, so contractions arrive as "arent", "dont"...
_NEGATIONS = {"not", "no", "never", "without", "except", "arent", "isnt", "dont",
              "doesnt", "wasnt", "werent", "cant", "avoid", "anti", "nothing", "none", "exclude"}
_NEG_WINDOW = 5  # tokens before a match to scan for a negation cue
# clause boundary: a negation does not cross these into the next clause. "only"/"just" introduce a
# POSITIVE list ("not any boot, only shoe or sneaker" -> exclude boot, KEEP shoe/sneaker).
_NEG_STOP = {"but", "however", "though", "only", "just", "clausebreak"}
# a comma/semicolon/period ends a clause, so a negation does not reach past it: a leading discourse
# "no," ("no, some other shoe") or a scoped "not any boot," must not negate the nouns that follow.
# Replaced with a plain alnum token so the existing _NEG_STOP scan treats it as a boundary.
_CLAUSE_BREAK = re.compile(r"[,;.!?]+")
_BREAK_TOKEN = "clausebreak"

COLOUR: Dict[str, str] = {
    "black": "Black",
    "white": "White", "off white": "Off White", "offwhite": "Off White", "cream": "Off White",
    "ivory": "Off White", "chalk": "Off White",
    "navy": "Dark Blue", "dark blue": "Dark Blue", "cobalt": "Dark Blue", "indigo": "Dark Blue",
    "denim blue": "Dark Blue", "midnight blue": "Dark Blue",
    "light blue": "Light Blue", "sky blue": "Light Blue", "baby blue": "Light Blue", "powder blue": "Light Blue",
    "blue": "Blue", "royal blue": "Blue",
    "red": "Red", "scarlet": "Red", "cherry": "Red",
    "dark red": "Dark Red", "burgundy": "Dark Red", "maroon": "Dark Red", "wine": "Dark Red", "crimson": "Dark Red",
    "pink": "Pink", "salmon": "Pink", "coral": "Pink", "rose": "Pink",
    "light pink": "Light Pink", "blush": "Light Pink", "baby pink": "Light Pink", "pastel pink": "Light Pink",
    "dark pink": "Dark Pink", "hot pink": "Dark Pink", "fuchsia": "Dark Pink", "magenta": "Dark Pink",
    "grey": "Grey", "gray": "Grey",
    "dark grey": "Dark Grey", "dark gray": "Dark Grey", "charcoal": "Dark Grey", "slate": "Dark Grey",
    "light grey": "Light Grey", "light gray": "Light Grey",
    "beige": "Beige", "tan": "Beige", "nude": "Beige", "camel": "Beige", "champagne": "Beige",
    "oatmeal": "Beige", "stone": "Beige",
    "light beige": "Light Beige", "sand": "Light Beige", "ecru": "Light Beige",
    "dark beige": "Dark Beige", "taupe": "Dark Beige", "mushroom": "Dark Beige",
    "green": "Dark Green", "dark green": "Dark Green", "emerald": "Dark Green", "forest green": "Dark Green",
    "jade": "Dark Green", "bottle green": "Dark Green", "hunter green": "Dark Green", "sage": "Dark Green",
    "mint": "Dark Green",
    "khaki": "Greenish Khaki", "olive": "Greenish Khaki", "greenish khaki": "Greenish Khaki", "army green": "Greenish Khaki",
    "yellow": "Yellow", "mustard": "Yellow", "lemon": "Yellow",
    "gold": "Gold", "golden": "Gold", "silver": "Silver", "metallic": "Silver",
    "orange": "Orange", "light orange": "Light Orange", "peach": "Light Orange", "apricot": "Light Orange",
    "dark orange": "Dark Orange", "rust": "Dark Orange", "terracotta": "Dark Orange", "burnt orange": "Dark Orange",
    "brown": "Yellowish Brown", "yellowish brown": "Yellowish Brown", "chocolate": "Yellowish Brown",
    "mocha": "Yellowish Brown", "bronze": "Yellowish Brown", "coffee": "Yellowish Brown", "tobacco": "Yellowish Brown",
    "purple": "Purple", "lilac": "Purple", "violet": "Purple", "lavender": "Purple",
    "plum": "Purple", "mauve": "Purple",
    "turquoise": "Turquoise", "teal": "Turquoise", "aqua": "Turquoise", "cyan": "Turquoise",
}
FIT: Dict[str, str] = {
    "skinny": "Tight/Skinny/Bodycon", "tight": "Tight/Skinny/Bodycon", "bodycon": "Tight/Skinny/Bodycon",
    "fitted": "Tight/Skinny/Bodycon", "figure hugging": "Tight/Skinny/Bodycon",
    "oversized": "Oversized/Loose/Relaxed", "loose": "Oversized/Loose/Relaxed", "relaxed": "Oversized/Loose/Relaxed",
    "baggy": "Oversized/Loose/Relaxed", "loose fit": "Oversized/Loose/Relaxed", "boxy": "Oversized/Loose/Relaxed",
    "slim": "Slim/Tailored", "tailored": "Slim/Tailored", "slim fit": "Slim/Tailored",
    "regular": "Regular/Straight", "straight": "Regular/Straight", "classic fit": "Regular/Straight",
}
OCCASION: Dict[str, str] = {
    "party": "Party/Evening/Wedding", "evening": "Party/Evening/Wedding", "wedding": "Party/Evening/Wedding",
    "weddings": "Party/Evening/Wedding", "cocktail": "Party/Evening/Wedding", "gala": "Party/Evening/Wedding",
    "prom": "Party/Evening/Wedding", "night out": "Party/Evening/Wedding", "date night": "Party/Evening/Wedding",
    "festive": "Party/Evening/Wedding", "clubbing": "Party/Evening/Wedding", "club": "Party/Evening/Wedding",
    "festival": "Party/Evening/Wedding", "concert": "Party/Evening/Wedding", "gig": "Party/Evening/Wedding",
    "graduation": "Party/Evening/Wedding", "going out": "Party/Evening/Wedding", "birthday": "Party/Evening/Wedding",
    "workout": "Sport/Active/Workout", "working out": "Sport/Active/Workout", "gym": "Sport/Active/Workout",
    "sport": "Sport/Active/Workout", "sports": "Sport/Active/Workout", "running": "Sport/Active/Workout",
    "jogging": "Sport/Active/Workout", "training": "Sport/Active/Workout", "athletic": "Sport/Active/Workout",
    "exercise": "Sport/Active/Workout", "yoga": "Sport/Active/Workout",
    "office": "Office/Workwear", "work": "Office/Workwear", "business": "Office/Workwear", "formal": "Office/Workwear",
    "workwear": "Office/Workwear", "interview": "Office/Workwear", "meeting": "Office/Workwear",
    "meetings": "Office/Workwear", "professional": "Office/Workwear",
    "lounge": "Lounge/Sleep/Nightwear", "loungewear": "Lounge/Sleep/Nightwear", "sleep": "Lounge/Sleep/Nightwear",
    "sleeping": "Lounge/Sleep/Nightwear", "sleepwear": "Lounge/Sleep/Nightwear", "pyjama": "Lounge/Sleep/Nightwear",
    "pajama": "Lounge/Sleep/Nightwear", "nightwear": "Lounge/Sleep/Nightwear", "bedtime": "Lounge/Sleep/Nightwear",
    "beach": "Beach/Swimwear", "swim": "Beach/Swimwear", "swimming": "Beach/Swimwear", "pool": "Beach/Swimwear",
    "poolside": "Beach/Swimwear",
    "casual": "Casual/Everyday", "everyday": "Casual/Everyday", "daily": "Casual/Everyday", "errands": "Casual/Everyday",
    "lingerie": "Intimate/Underwear", "intimate": "Intimate/Underwear",
    "outdoor": "Outdoor/Adventure", "hiking": "Outdoor/Adventure", "adventure": "Outdoor/Adventure",
    "camping": "Outdoor/Adventure", "skiing": "Outdoor/Adventure", "trekking": "Outdoor/Adventure",
}
SEASONALITY: Dict[str, str] = {
    "summer": "Spring/Summer", "spring": "Spring/Summer", "summery": "Spring/Summer", "warm weather": "Spring/Summer",
    "hot summer": "Spring/Summer", "in the heat": "Spring/Summer", "heatwave": "Spring/Summer",
    "winter": "Autumn/Winter", "autumn": "Autumn/Winter", "fall": "Autumn/Winter", "wintery": "Autumn/Winter",
    "chilly": "Autumn/Winter", "cold weather": "Autumn/Winter", "snow": "Autumn/Winter", "skiing": "Autumn/Winter",
    "all year": "Core/All-year", "all-year": "Core/All-year", "year round": "Core/All-year",
}
# "no pattern / plain / basic" -> Solid (the majority value). The multi-word negative phrases
# carry their own "no"/"without" INSIDE the surface, so the negation guard (which scans tokens
# BEFORE a match) does not flip them -- they resolve to a POSITIVE Solid filter, which is what
# the user means ("a plain white tee", "trousers without a pattern").
GRAPHICAL: Dict[str, str] = {
    "solid": "Solid", "plain": "Solid", "basic": "Solid", "minimal": "Solid", "minimalist": "Solid",
    "unpatterned": "Solid", "no pattern": "Solid", "no patterns": "Solid", "without pattern": "Solid",
    "without patterns": "Solid", "without any pattern": "Solid", "no print": "Solid",
    "without print": "Solid", "without prints": "Solid", "no prints": "Solid", "plain colour": "Solid",
    "solid colour": "Solid", "single colour": "Solid", "plain color": "Solid", "solid color": "Solid",
}
# Gender intent -> H&M index group. Applied as a post-filter on section_name/index_name (not a
# hard Qdrant field), so the value is the canonical group name.
GENDER: Dict[str, str] = {
    "men": "Men", "mens": "Men", "male": "Men", "man": "Men", "gentleman": "Men",
    "gentlemen": "Men", "masculine": "Men", "boy": "Men", "boys": "Men", "guys": "Men",
    "women": "Women", "womens": "Women", "female": "Women", "woman": "Women", "ladies": "Women",
    "lady": "Women", "feminine": "Women", "girl": "Women", "girls": "Women",
}
# Common garment words -> canonical product_type. The 1.5B query LLM is unreliable at this for
# short queries ("boot to go with this" -> it hallucinated "Dress"; "shoe..." -> nothing), so the
# deterministic map WINS for these common words; the LLM still handles rarer/OOV types. Longest
# surface wins, so the compound guards ("dress shirt", "shirt dress", "boot cut") resolve correctly.
PRODUCT_TYPE: Dict[str, str] = {
    "shoe": "Other shoe", "shoes": "Other shoe", "dress shoe": "Other shoe", "dress shoes": "Other shoe",
    "boot": "Boots", "boots": "Boots", "bootie": "Boots", "booties": "Boots", "ankle boot": "Boots",
    "boot cut": "Trousers", "bootcut": "Trousers",
    "sneaker": "Sneakers", "sneakers": "Sneakers", "trainer": "Sneakers", "trainers": "Sneakers",
    "sandal": "Sandals", "sandals": "Sandals", "flip flop": "Sandals",
    "trouser": "Trousers", "trousers": "Trousers", "pants": "Trousers", "slacks": "Trousers", "chinos": "Trousers",
    "jeans": "Trousers", "jean": "Trousers", "denim trousers": "Trousers",
    "leggings": "Leggings/Tights", "tights": "Leggings/Tights",
    "skirt": "Skirt", "skirts": "Skirt", "shorts": "Shorts",
    "shirt dress": "Dress", "shirtdress": "Dress", "dress": "Dress", "dresses": "Dress", "gown": "Dress",
    "dress shirt": "Shirt", "shirt": "Shirt", "shirts": "Shirt",
    "t shirt": "T-shirt", "tshirt": "T-shirt", "t shirts": "T-shirt", "tee": "T-shirt", "tees": "T-shirt",
    "blouse": "Blouse", "blouses": "Blouse",
    "vest top": "Vest top", "tank top": "Vest top", "crop top": "Vest top",
    "blazer": "Blazer", "blazers": "Blazer", "jacket": "Jacket", "jackets": "Jacket",
    "coat": "Coat", "coats": "Coat", "parka": "Jacket",
    "sweater": "Sweater", "sweaters": "Sweater", "jumper": "Sweater", "pullover": "Sweater", "knit": "Sweater",
    "hoodie": "Hoodie", "hoodies": "Hoodie", "cardigan": "Cardigan", "cardigans": "Cardigan",
    "bag": "Bag", "bags": "Bag", "handbag": "Bag", "purse": "Bag", "belt": "Belt", "belts": "Belt",
}


class AttributeGazetteer:
    FIELD_MAPS = {"colour_group": COLOUR, "fit": FIT, "occasion": OCCASION,
                  "seasonality": SEASONALITY, "graphical_appearance": GRAPHICAL, "gender": GENDER,
                  "product_type": PRODUCT_TYPE}

    def __init__(self) -> None:
        self._ordered = {
            field: sorted(mapping.keys(), key=len, reverse=True)
            for field, mapping in self.FIELD_MAPS.items()
        }

    @staticmethod
    def _span_start(tokens: List[str], surface_tokens: List[str]) -> Optional[int]:
        n = len(surface_tokens)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == surface_tokens:
                return i
        return None

    def _negated(self, tokens: List[str], start: int) -> bool:
        lo = max(0, start - _NEG_WINDOW)
        for t in reversed(tokens[lo:start]):
            if t in _NEG_STOP:  # e.g. "no heels but a red dress" -> "no" must not reach "red"
                break
            if t in _NEGATIONS:
                return True
        return False

    def extract(self, text: str, vocab: Dict[str, List[str]] | None = None) -> Dict[str, str]:
        # drop apostrophes (join) so contractions stay one token: "aren't" -> "arent",
        # otherwise _NON_ALNUM would split it to "aren" + "t" and the negation cue is lost.
        cleaned = normalize_text(text or "").replace("'", "").replace("’", "")
        cleaned = _CLAUSE_BREAK.sub(f" {_BREAK_TOKEN} ", cleaned)
        raw = _NON_ALNUM.sub(" ", cleaned).strip()
        if not raw:
            return {}
        tokens = raw.split()
        norm = f" {raw} "
        out: Dict[str, str] = {}
        for field, surfaces in self._ordered.items():
            valid = set(vocab.get(field, []) or []) if vocab else None
            for surface in surfaces:
                if f" {surface} " not in norm:
                    continue
                start = self._span_start(tokens, surface.split())
                if start is not None and self._negated(tokens, start):
                    continue  # e.g. "aren't too skinny" -> do not extract fit
                target = self.FIELD_MAPS[field][surface]
                if valid is None or target in valid:
                    out[field] = target
                    break
        return out

    def extract_negated(self, text: str, vocab: Dict[str, List[str]] | None = None) -> Dict[str, str]:
        """Mirror of extract() that keeps the NEGATED matches instead of dropping them
        ("a dress but not red", "jeans that aren't too skinny") so the caller can feed them
        to must_not and actually exclude the attribute, rather than silently ignoring it."""
        cleaned = normalize_text(text or "").replace("'", "").replace("’", "")
        cleaned = _CLAUSE_BREAK.sub(f" {_BREAK_TOKEN} ", cleaned)
        raw = _NON_ALNUM.sub(" ", cleaned).strip()
        if not raw:
            return {}
        tokens = raw.split()
        norm = f" {raw} "
        out: Dict[str, str] = {}
        for field, surfaces in self._ordered.items():
            valid = set(vocab.get(field, []) or []) if vocab else None
            for surface in surfaces:
                if f" {surface} " not in norm:
                    continue
                start = self._span_start(tokens, surface.split())
                if start is None or not self._negated(tokens, start):
                    continue  # keep ONLY negated matches here
                target = self.FIELD_MAPS[field][surface]
                if valid is None or target in valid:
                    out[field] = target
                    break
        return out
