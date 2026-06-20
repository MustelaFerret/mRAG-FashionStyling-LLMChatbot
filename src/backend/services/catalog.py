from __future__ import annotations

import csv
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from src.backend.core.utils import normalize_article_id, normalize_text
from src.scripts.graph.outfit_slots import get_slot as _get_outfit_slot


PT_SYNONYMS = {
    "pants": "Trousers", "pant": "Trousers", "slacks": "Trousers", "chinos": "Trousers",
    "jeans": "Trousers", "jean": "Trousers", "denim": "Trousers", "trouser": "Trousers",
    "tee": "T-shirt", "tees": "T-shirt", "tshirt": "T-shirt", "t shirt": "T-shirt", "t-shirts": "T-shirt",
    "jumper": "Sweater", "pullover": "Sweater", "knit": "Sweater", "sweaters": "Sweater",
    "hoody": "Hoodie", "hoodies": "Hoodie",
    "trainers": "Sneakers", "trainer": "Sneakers", "kicks": "Sneakers", "sneaker": "Sneakers",
    "tank": "Vest top", "tank top": "Vest top", "singlet": "Vest top",
    "jackets": "Jacket", "coats": "Coat", "blazers": "Blazer", "shirts": "Shirt",
    "dresses": "Dress", "skirts": "Skirt", "shorts": "Shorts",
}


class FashionCatalog:
    def __init__(self, meta_file: str, graph_file: str, image_dir: str, aux_graph_file: str = ""):
        self.meta_file = meta_file
        self.graph_file = graph_file
        self.image_dir = image_dir

        self.meta_by_article: Dict[str, Dict] = {}
        self.valid_product_types: List[str] = []
        self.valid_colors: List[str] = []
        self.valid_occasions: List[str] = []
        self.valid_fits: List[str] = []
        self.valid_seasonalities: List[str] = []
        self.valid_graphical_appearances: List[str] = []
        self.graph_adj: Dict[str, List[Tuple[str, float]]] = {}
        # auxiliary transaction-derived edges (P3alpha) covering items the co-buy
        # graph misses; cold-tier #1 in pairing (md/refine_5.MD)
        self.aux_adj: Dict[str, List[Tuple[str, float]]] = {}

        self._load_meta()
        self.graph_adj = self._build_graph_adjacency()
        if aux_graph_file and os.path.exists(aux_graph_file):
            self.aux_adj = self._build_graph_adjacency(aux_graph_file)


    @staticmethod
    def _unique_values(values: Iterable[str]) -> List[str]:
        cleaned = {str(v).strip() for v in values if str(v).strip()}
        return sorted(cleaned)

    def _load_meta(self) -> None:
        product_type_values: set[str] = set()
        color_values: set[str] = set()
        occasion_values: set[str] = set()
        fit_values: set[str] = set()
        seasonality_values: set[str] = set()
        graphical_values: set[str] = set()
        term_pt_counts: Dict[str, Counter] = defaultdict(Counter)

        with open(self.meta_file, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                article_id = normalize_article_id(row.get("article_id", ""))
                if not article_id:
                    continue

                product_type = str(row.get("product_type_name", "") or "").strip()
                colour_group = str(row.get("colour_group_name", "") or "").strip()
                fit = str(row.get("fit", "") or "").strip()
                occasion = str(row.get("occasion", "") or "").strip()
                seasonality = str(row.get("seasonality", "") or "").strip()
                description = str(row.get("refined_description", "") or "").strip()

                self.meta_by_article[article_id] = {
                    "article_id": article_id,
                    "product_type_name": product_type,
                    "colour_group_name": colour_group,
                    "fit": fit,
                    "occasion": occasion,
                    "seasonality": seasonality,
                    # slot 4-tier fallback inputs (infer_article_slot reads these): without them
                    # get_slot only had Tier-1 (product_type), so ~111 Unknown-PT items resolved
                    # to no slot at runtime and were dropped from slot-filtered pairing.
                    "product_group_name": str(row.get("product_group_name", "") or "").strip(),
                    "garment_group_name": str(row.get("garment_group_name", "") or "").strip(),
                    "prod_name": str(row.get("prod_name", "") or "").strip(),
                    "section_name": str(row.get("section_name", "") or "").strip(),
                    "department_name": str(row.get("department_name", "") or "").strip(),
                    # gap C (audit_metadata): pattern / material / colour shade+family —
                    # usable attributes that were dropped from runtime meta, kept for
                    # grading and (future) hard/soft filtering.
                    "graphical_appearance_name": str(row.get("graphical_appearance_name", "") or "").strip(),
                    "dominant_material": str(row.get("dominant_material", "") or "").strip(),
                    "perceived_colour_value_name": str(row.get("perceived_colour_value_name", "") or "").strip(),
                    "perceived_colour_master_name": str(row.get("perceived_colour_master_name", "") or "").strip(),
                    "refined_description": description,
                }

                if product_type:
                    product_type_values.add(product_type)
                    text = f"{row.get('prod_name', '')} {row.get('detail_desc', '')} {description}".lower()
                    for token in set(re.findall(r"[a-z]{3,}", text)):
                        term_pt_counts[token][product_type] += 1
                if colour_group:
                    color_values.add(colour_group)
                if occasion:
                    occasion_values.add(occasion)
                if fit:
                    fit_values.add(fit)
                if seasonality:
                    seasonality_values.add(seasonality)
                graphical = str(row.get("graphical_appearance_name", "") or "").strip()
                if graphical:
                    graphical_values.add(graphical)

        self.term_pt_counts = {t: c for t, c in term_pt_counts.items() if sum(c.values()) >= 5}
        self.valid_product_types = self._unique_values(product_type_values)
        self.valid_colors = self._unique_values(color_values)
        self.valid_occasions = self._unique_values(occasion_values)
        self.valid_fits = self._unique_values(fit_values)
        self.valid_seasonalities = self._unique_values(seasonality_values)
        self.valid_graphical_appearances = self._unique_values(graphical_values)

    def _build_graph_adjacency(self, graph_file: str = "") -> Dict[str, List[Tuple[str, float]]]:
        adjacency: Dict[str, Dict[str, float]] = {}
        with open(graph_file or self.graph_file, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                a = normalize_article_id(row.get("item_a", ""))
                b = normalize_article_id(row.get("item_b", ""))
                try:
                    w = float(row.get("weight", 0.0) or 0.0)
                except (TypeError, ValueError):
                    w = 0.0
                if not a or not b or a == b:
                    continue

                adjacency.setdefault(a, {})
                adjacency.setdefault(b, {})
                adjacency[a][b] = max(w, adjacency[a].get(b, 0.0))
                adjacency[b][a] = max(w, adjacency[b].get(a, 0.0))

        out: Dict[str, List[Tuple[str, float]]] = {}
        for aid, neighbors in adjacency.items():
            out[aid] = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        return out

    _SLOT_KEYWORDS = [
        ("shoe", ["sneaker", "shoe", "boot", "sandal", "loafer", "heel", "clog", "slipper", "giay"]),
        ("bottom", ["pants", "trouser", "jean", "skirt", "short", "legging", "jogger", "chinos", "quan", "chan vay"]),
        ("outerwear", ["jacket", "coat", "blazer", "cardigan", "outerwear", "windbreaker", "parka"]),
        ("accessory", ["bag", "belt", "hat", "cap", "scarf", "watch", "jewelry", "accessory", "phu kien"]),
        ("top", ["shirt", "top", "tee", "t-shirt", "blouse", "hoodie", "sweater", "tank", "polo", "ao"]),
    ]

    @staticmethod
    def infer_slot_from_text(text: str, exclude_slot: str = "") -> str:
        value = normalize_text(text)
        if not value:
            return ""
        for slot, keywords in FashionCatalog._SLOT_KEYWORDS:
            if slot == exclude_slot:
                continue
            if any(k in value for k in keywords):
                return slot
        return ""

    def infer_article_slot(self, article_id: str) -> str:
        aid = normalize_article_id(article_id)
        if not aid:
            return ""
        meta = self.get_meta(aid)
        # use the validated 4-tier PT->slot mapping (outfit_slots), not the query-keyword
        # heuristic: keyword matching on PT names leaks (same-slot pairing stayed at 7.1%)
        slot = _get_outfit_slot(
            meta.get("product_type_name", ""),
            product_group=meta.get("product_group_name", ""),
            garment_group=meta.get("garment_group_name", ""),
            prod_name=meta.get("prod_name", ""),
            section_name=meta.get("section_name", ""),
            department_name=meta.get("department_name", ""),
        )
        return "" if slot == "other" else slot

    def get_meta(self, article_id: str) -> Dict:
        return self.meta_by_article.get(normalize_article_id(article_id), {})

    def canonical_product_type(self, value: str) -> str:
        v = str(value or "").strip()
        if not v:
            return ""
        low = v.lower()
        for pt in self.valid_product_types:
            if pt.lower() == low:
                return pt
        syn = PT_SYNONYMS.get(low)
        if syn and syn in self.valid_product_types:
            return syn
        return ""

    def corpus_product_type(self, term: str, min_support: int = 15, min_majority: float = 0.5) -> str:
        """Canonicalise an out-of-vocabulary garment term via corpus statistics.

        Looks at which product_type the catalogue items whose description contains the
        term actually carry, and returns the dominant one. This follows the catalogue's
        real labelling convention (e.g. parka -> Jacket) rather than generic word
        similarity (which mis-maps parka -> Coat). Empty if support/majority too weak.
        """
        counts = getattr(self, "term_pt_counts", None)
        if not term or not counts:
            return ""
        combined: Counter = Counter()
        for token in re.findall(r"[a-z]{3,}", term.lower()):
            bucket = counts.get(token)
            if bucket:
                combined.update(bucket)
        total = sum(combined.values())
        if total < min_support:
            return ""
        pt, cnt = combined.most_common(1)[0]
        if cnt / total >= min_majority and pt in self.valid_product_types:
            return pt
        return ""

    def article_image_path(self, article_id: str) -> str:
        aid = normalize_article_id(article_id)
        if not aid:
            return ""
        return os.path.join(self.image_dir, aid[:3], f"{aid}.jpg")

    def get_graph_diverse_neighbors(
        self,
        anchor_id: str,
        limit: int,
        max_per_pt: int,
        preferred_min_weight: int,
        hard_min_weight: int,
        use_aux: bool = False,
    ) -> List[str]:
        """Diverse top neighbors from the co-buy graph (or, with use_aux=True, from the
        P3alpha transaction edges — cold-tier #1; those weights are walk scores, so no
        hard_min_weight cut is applied to them)."""
        aid = normalize_article_id(anchor_id)
        source = self.aux_adj if use_aux else self.graph_adj
        min_w = 0.0 if use_aux else hard_min_weight
        all_neighbors = [(n, w) for n, w in source.get(aid, []) if w >= min_w]
        if not all_neighbors:
            return []
        anchor_season = self.get_meta(aid).get("seasonality", "")
        pt_count: Dict[str, int] = {}
        selected: List[str] = []
        for nid, _ in all_neighbors:
            cid = normalize_article_id(nid)
            if not cid:
                continue
            meta = self.get_meta(cid)
            pt = meta.get("product_type_name", "")
            # NOTE: no slot-pair guard here — measured 0 whitelist-violating edges in the
            # whole production graph (built with the redesigned slot filter), a runtime
            # check would be pure overhead. Re-add if the graph is ever swapped for one
            # built without slot_pair_allowed.
            # hard season clash guard (Spring/Summer x Autumn/Winter): the graph still
            # carries 522 such edges (0.17%); measured 0.21% -> 0.00% of returned pairs.
            if {anchor_season, meta.get("seasonality", "")} == {"Spring/Summer", "Autumn/Winter"}:
                continue
            if pt and pt_count.get(pt, 0) >= max_per_pt:
                continue
            selected.append(cid)
            if pt:
                pt_count[pt] = pt_count.get(pt, 0) + 1
            if len(selected) >= limit:
                break
        return selected
