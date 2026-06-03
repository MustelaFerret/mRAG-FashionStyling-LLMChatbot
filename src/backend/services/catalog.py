from __future__ import annotations

import csv
import os
import re
from typing import Dict, Iterable, List, Tuple

from src.backend.core.utils import normalize_article_id, normalize_text


class FashionCatalog:
    def __init__(self, meta_file: str, graph_file: str, image_dir: str):
        self.meta_file = meta_file
        self.graph_file = graph_file
        self.image_dir = image_dir

        self.meta_by_article: Dict[str, Dict] = {}
        self.product_code_to_articles: Dict[str, List[str]] = {}
        self.product_type_values: List[Tuple[str, str]] = []
        self.color_values: List[Tuple[str, str]] = []
        self.occasion_values: List[Tuple[str, str]] = []
        self.fit_values: List[Tuple[str, str]] = []
        self.valid_product_types: List[str] = []
        self.valid_colors: List[str] = []
        self.valid_occasions: List[str] = []
        self.valid_fits: List[str] = []
        self.valid_seasonalities: List[str] = []
        self.graph_adj: Dict[str, List[Tuple[str, float]]] = {}

        self._load_meta()
        self.graph_adj = self._build_graph_adjacency()

    @staticmethod
    def _build_search_values(values: Iterable[str]) -> List[tuple[str, str]]:
        unique_values: List[str] = []
        seen: set[str] = set()
        for value in values:
            raw = str(value).strip()
            if not raw or raw in seen:
                continue
            seen.add(raw)
            unique_values.append(raw)
        unique_values.sort(key=lambda v: len(v), reverse=True)
        return [(normalize_text(v), v) for v in unique_values]

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

        with open(self.meta_file, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                article_id = normalize_article_id(row.get("article_id", ""))
                if not article_id:
                    continue

                product_code = str(row.get("product_code", "") or "").strip()
                if product_code:
                    product_code = product_code.zfill(6)

                product_type = str(row.get("product_type_name", "") or "").strip()
                colour_group = str(row.get("colour_group_name", "") or "").strip()
                fit = str(row.get("fit", "") or "").strip()
                occasion = str(row.get("occasion", "") or "").strip()
                seasonality = str(row.get("seasonality", "") or "").strip()
                description = str(row.get("refined_description", "") or "").strip()

                self.meta_by_article[article_id] = {
                    "article_id": article_id,
                    "product_code": product_code,
                    "product_type_name": product_type,
                    "colour_group_name": colour_group,
                    "fit": fit,
                    "occasion": occasion,
                    "seasonality": seasonality,
                    "refined_description": description,
                }

                if product_code:
                    self.product_code_to_articles.setdefault(product_code, []).append(article_id)

                if product_type:
                    product_type_values.add(product_type)
                if colour_group:
                    color_values.add(colour_group)
                if occasion:
                    occasion_values.add(occasion)
                if fit:
                    fit_values.add(fit)
                if seasonality:
                    seasonality_values.add(seasonality)

        self.product_type_values = self._build_search_values(product_type_values)
        self.color_values = self._build_search_values(color_values)
        self.occasion_values = self._build_search_values(occasion_values)
        self.fit_values = self._build_search_values(fit_values)
        self.valid_product_types = self._unique_values(product_type_values)
        self.valid_colors = self._unique_values(color_values)
        self.valid_occasions = self._unique_values(occasion_values)
        self.valid_fits = self._unique_values(fit_values)
        self.valid_seasonalities = self._unique_values(seasonality_values)

    def _build_graph_adjacency(self) -> Dict[str, List[Tuple[str, float]]]:
        adjacency: Dict[str, Dict[str, float]] = {}
        with open(self.graph_file, newline="", encoding="utf-8") as handle:
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

    @staticmethod
    def _contains_phrase(query: str, phrase: str) -> bool:
        if not query or not phrase:
            return False
        # hyphen/underscore/slash → space để 'off-white' khớp color value 'off white',
        # 'v-neck'/'wide-leg' khớp tương tự. Áp cả 2 vế để giữ tính đối xứng.
        q = query.replace("-", " ").replace("_", " ").replace("/", " ")
        p = phrase.replace("-", " ").replace("_", " ").replace("/", " ")
        return re.search(rf"(?<!\\w){re.escape(p)}(?!\\w)", q) is not None

    @staticmethod
    def _match_first_value(query: str, values: List[Tuple[str, str]]) -> str:
        for normalized, original in values:
            if normalized and FashionCatalog._contains_phrase(query, normalized):
                return original
        return ""

    @staticmethod
    def _match_value_by_alias(query: str, values: List[Tuple[str, str]], aliases: List[str]) -> str:
        if not aliases:
            return ""
        if not any(FashionCatalog._contains_phrase(query, alias) for alias in aliases):
            return ""
        for normalized, original in values:
            if any(FashionCatalog._contains_phrase(normalized, alias) for alias in aliases):
                return original
        return ""

    @staticmethod
    def _infer_slot_from_text(text: str) -> str:
        value = normalize_text(text)
        if not value:
            return ""

        if any(k in value for k in ["sneaker", "shoe", "boot", "sandal", "loafer", "heel", "clog", "slipper", "giay"]):
            return "shoe"
        if any(k in value for k in ["pants", "trouser", "jean", "skirt", "short", "legging", "jogger", "chinos", "quan", "chan vay"]):
            return "bottom"
        if any(k in value for k in ["jacket", "coat", "blazer", "cardigan", "outerwear", "windbreaker", "parka"]):
            return "outerwear"
        if any(k in value for k in ["bag", "belt", "hat", "cap", "scarf", "watch", "jewelry", "accessory", "phu kien"]):
            return "accessory"
        if any(k in value for k in ["shirt", "top", "tee", "t-shirt", "blouse", "hoodie", "sweater", "tank", "polo", "ao"]):
            return "top"
        return ""

    def infer_article_slot(self, article_id: str) -> str:
        aid = normalize_article_id(article_id)
        if not aid:
            return ""
        meta = self.get_meta(aid)
        return self._infer_slot_from_text(meta.get("product_type_name", ""))

    def infer_item_slot(self, item: Dict) -> str:
        if not item:
            return ""
        slot = self._infer_slot_from_text(item.get("product_type", ""))
        if slot:
            return slot
        return self._infer_slot_from_text(item.get("description", ""))

    def infer_target_slots(self, user_query: str, anchor_id: str = "") -> List[str]:
        query = normalize_text(user_query)
        requested_slot = self._infer_slot_from_text(query)
        anchor_slot = self.infer_article_slot(anchor_id)

        if requested_slot == "top":
            return ["top", "bottom", "shoe"]
        if requested_slot == "bottom":
            return ["bottom", "shoe", "outerwear"]
        if requested_slot == "shoe":
            return ["shoe", "bottom", "top"]
        if requested_slot == "outerwear":
            return ["outerwear", "top", "bottom"]
        if requested_slot == "accessory":
            return ["accessory", "top", "bottom"]

        if anchor_slot == "top":
            return ["bottom", "shoe", "accessory"]
        if anchor_slot == "bottom":
            return ["top", "shoe", "accessory"]
        if anchor_slot == "shoe":
            return ["bottom", "top", "accessory"]
        if anchor_slot == "outerwear":
            return ["top", "bottom", "shoe"]

        return ["top", "bottom", "shoe"]

    def _weighted_neighbors(
        self,
        article_id: str,
        preferred_min_weight: int,
        hard_min_weight: int,
    ) -> List[Tuple[str, float]]:
        aid = normalize_article_id(article_id)
        neighbors = self.graph_adj.get(aid, [])
        if not neighbors:
            return []

        preferred = [(nid, w) for nid, w in neighbors if w >= preferred_min_weight]
        if preferred:
            return preferred
        return [(nid, w) for nid, w in neighbors if w >= hard_min_weight]

    def get_meta(self, article_id: str) -> Dict:
        return self.meta_by_article.get(normalize_article_id(article_id), {})

    def build_item(self, article_id: str, payload: Dict | None = None) -> Dict:
        payload = payload or {}
        aid = normalize_article_id(payload.get("article_id", article_id))
        meta = self.get_meta(aid)
        return {
            "article_id": aid,
            "product_type": str(payload.get("product_type") or meta.get("product_type_name", "")),
            "colour_group": str(payload.get("colour_group") or meta.get("colour_group_name", "")),
            "fit": str(payload.get("fit") or meta.get("fit", "")),
            "occasion": str(payload.get("occasion") or meta.get("occasion", "")),
            "seasonality": str(payload.get("seasonality") or meta.get("seasonality", "")),
            "description": str(payload.get("description") or meta.get("refined_description", "")),
        }

    def article_image_path(self, article_id: str) -> str:
        aid = normalize_article_id(article_id)
        if not aid:
            return ""
        return os.path.join(self.image_dir, aid[:3], f"{aid}.jpg")

    def parse_query_filters(self, user_query: str) -> Dict[str, str]:
        query = normalize_text(user_query)
        if not query:
            return {}

        filters: Dict[str, str] = {}

        product_type = self._match_first_value(query, self.product_type_values)
        if not product_type:
            product_type_aliases = {
                "shirt": ["shirt", "so mi", "ao so mi", "blouse", "top", "tee", "t-shirt", "ao"],
                "pants": ["pants", "trouser", "jean", "jogger", "legging", "quan", "chan vay", "skirt", "short"],
                "shoe": ["shoe", "sneaker", "boot", "sandal", "loafer", "giay"],
                "jacket": ["jacket", "coat", "blazer", "cardigan", "outerwear", "khoac"],
                "dress": ["dress", "vay dam", "dam", "vay"],
                "hoodie": ["hoodie", "sweater", "len", "pullover"],
            }
            for aliases in product_type_aliases.values():
                product_type = self._match_value_by_alias(query, self.product_type_values, aliases)
                if product_type:
                    break
        if product_type:
            filters["product_type"] = product_type

        color = self._match_first_value(query, self.color_values)
        if not color:
            color_aliases = {
                "black": ["black", "den"],
                "white": ["white", "trang"],
                "blue": ["blue", "xanh duong", "xanh navy", "xanh"],
                "red": ["red", "do"],
                "green": ["green", "xanh la"],
                "brown": ["brown", "nau"],
                "beige": ["beige", "kem", "nude"],
                "grey": ["grey", "gray", "xam", "ghi"],
                "pink": ["pink", "hong"],
                "yellow": ["yellow", "vang"],
                "purple": ["purple", "tim"],
            }
            for aliases in color_aliases.values():
                color = self._match_value_by_alias(query, self.color_values, aliases)
                if color:
                    break
        if color:
            filters["colour_group"] = color

        season_map = {
            "xuan he": "Spring/Summer",
            "mua he": "Spring/Summer",
            "summer": "Spring/Summer",
            "spring summer": "Spring/Summer",
            "autumn winter": "Autumn/Winter",
            "fall winter": "Autumn/Winter",
            "mua dong": "Autumn/Winter",
            "winter": "Autumn/Winter",
            "dong": "Autumn/Winter",
            "he": "Spring/Summer",
            "all season": "All-Season",
            "all-season": "All-Season",
        }
        for keyword, season in season_map.items():
            if self._contains_phrase(query, keyword):
                filters["seasonality"] = season
                break

        occasion = self._match_first_value(query, self.occasion_values)
        if not occasion:
            occasion_aliases = {
                "casual": ["casual", "daily", "hang ngay", "di choi", "street"],
                "formal": ["formal", "office", "cong so", "smart", "business"],
                "sport": ["sport", "gym", "training", "workout", "the thao"],
                "party": ["party", "date", "event", "tiec"],
                "outdoor": ["outdoor", "travel", "du lich"],
            }
            for aliases in occasion_aliases.values():
                occasion = self._match_value_by_alias(query, self.occasion_values, aliases)
                if occasion:
                    break
        if occasion:
            filters["occasion"] = occasion

        fit = self._match_first_value(query, self.fit_values)
        if not fit:
            fit_aliases = {
                "regular": ["regular", "vua van", "classic"],
                "slim": ["slim", "om", "fitted"],
                "oversized": ["oversized", "rong", "loose", "relaxed", "wide"],
            }
            for aliases in fit_aliases.values():
                fit = self._match_value_by_alias(query, self.fit_values, aliases)
                if fit:
                    break
        if fit:
            filters["fit"] = fit

        return filters

    def infer_target_filter_key(self, user_query: str) -> str:
        query = normalize_text(user_query)
        if not query:
            return ""

        if any(k in query for k in ["quan", "pants", "trouser", "short", "jean", "jogger", "legging", "skirt", "chan vay"]):
            return "bottom"
        if any(k in query for k in ["giay", "shoe", "sneaker", "boot", "sandal"]):
            return "shoe"
        if any(k in query for k in ["ao", "shirt", "blouse", "top", "tee", "sweater", "hoodie", "jacket", "cardigan"]):
            return "top"
        return ""

    def item_matches_target(self, item: Dict, target_key: str) -> bool:
        if not target_key:
            return True

        text = normalize_text(
            f"{item.get('product_type', '')} {item.get('occasion', '')} {item.get('description', '')}"
        )
        if target_key == "bottom":
            return any(k in text for k in ["trouser", "pants", "short", "jean", "legging", "skirt", "chan vay", "jogger"])
        if target_key == "shoe":
            return any(k in text for k in ["shoe", "sneaker", "boot", "sandal", "loafer", "heel"])
        if target_key == "top":
            return any(k in text for k in ["shirt", "blouse", "top", "tee", "sweater", "hoodie", "cardigan", "jacket"])
        return True

    def filter_items_by_target(self, items: List[Dict], user_query: str) -> List[Dict]:
        target_key = self.infer_target_filter_key(user_query)
        if not target_key:
            return items
        filtered = [item for item in items if self.item_matches_target(item, target_key)]
        return filtered if filtered else items

    def get_graph_neighbor_ids(self, anchor_id: str, limit: int, preferred_min_weight: int, hard_min_weight: int) -> List[str]:
        neighbors = self._weighted_neighbors(anchor_id, preferred_min_weight, hard_min_weight)
        return [normalize_article_id(i) for i, _ in neighbors[:limit]]

    def get_graph_diverse_neighbors(
        self,
        anchor_id: str,
        limit: int,
        max_per_pt: int,
        preferred_min_weight: int,
        hard_min_weight: int,
    ) -> List[str]:
        aid = normalize_article_id(anchor_id)
        all_neighbors = [(n, w) for n, w in self.graph_adj.get(aid, []) if w >= hard_min_weight]
        if not all_neighbors:
            return []
        pt_count: Dict[str, int] = {}
        selected: List[str] = []
        for nid, _ in all_neighbors:
            cid = normalize_article_id(nid)
            if not cid:
                continue
            pt = self.get_meta(cid).get("product_type_name", "")
            if pt and pt_count.get(pt, 0) >= max_per_pt:
                continue
            selected.append(cid)
            if pt:
                pt_count[pt] = pt_count.get(pt, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    def get_graph_multihop_outfit_ids(
        self,
        anchor_id: str,
        max_hops: int,
        branch_per_hop: int,
        preferred_min_weight: int,
        hard_min_weight: int,
        limit: int,
        target_slots: List[str] | None = None,
    ) -> List[str]:
        aid = normalize_article_id(anchor_id)
        if not aid:
            return []

        hops = max(1, int(max_hops))
        branch = max(1, int(branch_per_hop))
        target_slots = target_slots or self.infer_target_slots("", anchor_id=aid)

        frontier: List[Tuple[str, float, int]] = [(aid, 1.0, 0)]
        best_score_by_node: Dict[str, float] = {aid: 1.0}
        ranked_candidates: List[Tuple[float, str, str, int]] = []
        best_by_slot: Dict[str, Tuple[float, str, int]] = {}

        for hop in range(1, hops + 1):
            next_frontier: List[Tuple[str, float, int]] = []
            for node, node_score, _ in frontier:
                neighbors = self._weighted_neighbors(node, preferred_min_weight, hard_min_weight)[:branch]
                for neighbor_id, weight in neighbors:
                    nid = normalize_article_id(neighbor_id)
                    if not nid or nid == aid:
                        continue

                    score = float(node_score) * float(weight)
                    prev_score = best_score_by_node.get(nid, -1.0)
                    if score <= prev_score:
                        continue

                    best_score_by_node[nid] = score
                    next_frontier.append((nid, score, hop))

                    slot = self.infer_article_slot(nid)
                    ranked_candidates.append((score, nid, slot, hop))
                    if slot:
                        current = best_by_slot.get(slot)
                        if current is None or score > current[0]:
                            best_by_slot[slot] = (score, nid, hop)

            if not next_frontier:
                break

            next_frontier.sort(key=lambda x: x[1], reverse=True)
            frontier = next_frontier[: branch * 2]

        selected: List[str] = []
        used = {aid}

        for slot in target_slots:
            slot_pick = best_by_slot.get(slot)
            if slot_pick is None:
                continue
            nid = slot_pick[1]
            if nid in used:
                continue
            selected.append(nid)
            used.add(nid)
            if len(selected) >= limit:
                return selected

        ranked_candidates.sort(key=lambda x: x[0], reverse=True)
        pt_count: Dict[str, int] = {}
        anchor_pt = self.get_meta(aid).get("product_type_name", "")
        max_per_pt = 2
        for _, nid, _, _ in ranked_candidates:
            if nid in used:
                continue
            cand_pt = self.get_meta(nid).get("product_type_name", "")
            if cand_pt and cand_pt == anchor_pt:
                continue
            if cand_pt and pt_count.get(cand_pt, 0) >= max_per_pt:
                continue
            selected.append(nid)
            used.add(nid)
            if cand_pt:
                pt_count[cand_pt] = pt_count.get(cand_pt, 0) + 1
            if len(selected) >= limit:
                break

        return selected

    def get_color_variant_ids(self, anchor_id: str, limit: int) -> List[str]:
        aid = normalize_article_id(anchor_id)
        anchor_meta = self.get_meta(aid)
        product_code = str(anchor_meta.get("product_code", "")).zfill(6)
        if not product_code and len(aid) >= 7 and aid[1:7].isdigit():
            product_code = aid[1:7]
        if not product_code:
            return []

        candidates = self.product_code_to_articles.get(product_code, [])
        if not candidates:
            return []

        anchor_color = str(anchor_meta.get("colour_group_name", "")).strip().lower()
        diff_color = []
        same_color = []
        for raw_id in candidates:
            cid = normalize_article_id(raw_id)
            if not cid or cid == aid:
                continue
            color = str(self.get_meta(cid).get("colour_group_name", "")).strip().lower()
            if color and color != anchor_color:
                diff_color.append(cid)
            else:
                same_color.append(cid)

        return (diff_color + same_color)[:limit]
