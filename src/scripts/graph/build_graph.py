from __future__ import annotations

import gc
import itertools
import logging
import math
import os
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from src.scripts.graph.outfit_slots import get_slot, slot_pair_allowed, same_pt_family

log = logging.getLogger(__name__)


@dataclass
class BuilderConfig:
    transactions_file: str
    meta_file: str
    output_file: str
    method: str = "npmi"
    min_cooc: int = 3
    npmi_threshold: float = 0.0
    weight_scale: int = 1000
    basket_min_size: int = 2
    basket_max_size: int = 10
    train_cutoff_date: str | None = None
    content_filter: str = "minimal"
    basket_cache_pickle: str | None = None


class OutfitGraphBuilder:
    UNWANTED_GROUPS = {"Under-, Nightwear"}
    ALLOWED_SAME_GROUPS = {"Accessories", "Swimwear"}

    NEUTRALS = {"Black", "White", "Off White", "Grey", "Light Grey", "Dark Grey", "Beige", "Light Beige", "Greyish Beige", "Silver", "Gold", "Transparent"}
    NAVY_BLUES = {"Dark Blue", "Navy"}
    BLUES = {"Blue", "Light Blue", "Other Blue", "Turquoise", "Light Turquoise", "Dark Turquoise"}
    GREENS = {"Green", "Light Green", "Dark Green", "Greenish Khaki", "Other Green", "Olive"}
    PINKS_REDS = {"Red", "Light Red", "Dark Red", "Pink", "Light Pink", "Dark Pink", "Other Red", "Other Pink", "Burgundy"}
    YELLOWS_ORANGES = {"Orange", "Light Orange", "Dark Orange", "Yellow", "Light Yellow", "Dark Yellow", "Other Yellow", "Other Orange", "Bronze/Copper"}
    BROWNS = {"Brown", "Dark Brown", "Yellowish Brown"}

    SPORT_SECTIONS = {"Ladies H&M Sport", "Men H&M Sport", "Kids Sports"}
    FORMAL_SECTIONS = {"Womens Tailoring", "Men Suits & Tailoring", "Contemporary Smart"}

    WINTER_KEYWORDS = ("winter", "fall", "snow", "cold", "chill")
    SUMMER_KEYWORDS = ("summer", "beach", "heat", "hot")
    ALL_SEASON_KEYWORDS = ("all-season", "all-year", "any season", "transition")

    def __init__(self, config: BuilderConfig):
        self.config = config
        self.meta_df: pd.DataFrame | None = None
        self.baskets: List[List[str]] | None = None
        self.pair_counts: Counter[Tuple[str, str]] | None = None
        self.item_counts: Counter[str] | None = None
        self.total_baskets_observed: int = 0

    def load_meta(self) -> pd.DataFrame:
        cols = [
            "article_id",
            "product_type_name",
            "product_group_name",
            "garment_group_name",
            "index_name",
            "section_name",
            "department_name",
            "prod_name",
            "colour_group_name",
            "seasonality",
        ]
        df = pd.read_csv(self.config.meta_file, usecols=cols, dtype={"article_id": str})
        df["article_id"] = df["article_id"].str.zfill(10)
        self.meta_df = df.set_index("article_id")
        log.info("loaded meta: %d articles", len(self.meta_df))
        return df

    def build_baskets(self) -> List[List[str]]:
        if self.meta_df is None:
            self.load_meta()
        if self.config.basket_cache_pickle and os.path.exists(self.config.basket_cache_pickle):
            with open(self.config.basket_cache_pickle, "rb") as handle:
                payload = pickle.load(handle)
            baskets = payload.get("train_baskets")
            if baskets:
                log.info("loaded %d train baskets from cache: %s", len(baskets), self.config.basket_cache_pickle)
                self.baskets = baskets
                return baskets
        valid_ids = set(self.meta_df.index)

        df = pd.read_csv(
            self.config.transactions_file,
            usecols=["t_dat", "customer_id", "article_id"],
            dtype={"article_id": str, "customer_id": "category", "t_dat": "string"},
        )
        df["article_id"] = df["article_id"].str.zfill(10)
        df = df[df["article_id"].isin(valid_ids)]

        if self.config.train_cutoff_date:
            df = df[df["t_dat"] <= self.config.train_cutoff_date]
            log.info("applied train cutoff %s, rows=%d", self.config.train_cutoff_date, len(df))

        df = df.drop_duplicates(subset=["customer_id", "t_dat", "article_id"])
        grouped = df.groupby(["customer_id", "t_dat"], observed=True)["article_id"].apply(list)
        baskets: List[List[str]] = []
        for items in grouped:
            unique_items = list(dict.fromkeys(items))
            size = len(unique_items)
            if size < self.config.basket_min_size or size > self.config.basket_max_size:
                continue
            baskets.append(unique_items)
        log.info("built %d baskets", len(baskets))
        self.baskets = baskets
        return baskets

    def compute_pair_counts(self) -> Counter[Tuple[str, str]]:
        if self.baskets is None:
            self.build_baskets()
        pair_counter: Counter[Tuple[str, str]] = Counter()
        item_counter: Counter[str] = Counter()
        observed = 0
        for items in self.baskets:
            observed += 1
            sorted_items = sorted(items)
            for item in sorted_items:
                item_counter[item] += 1
            for pair in itertools.combinations(sorted_items, 2):
                pair_counter[pair] += 1
        self.pair_counts = pair_counter
        self.item_counts = item_counter
        self.total_baskets_observed = observed
        log.info(
            "pair counts: total_pairs=%d unique_items=%d total_baskets=%d",
            len(pair_counter),
            len(item_counter),
            observed,
        )
        return pair_counter

    def _passes_content_filter(self, a: str, b: str, mode: str) -> bool:
        if mode == "none" or self.meta_df is None:
            return True
        if a not in self.meta_df.index or b not in self.meta_df.index:
            return False
        meta_a = self.meta_df.loc[a]
        meta_b = self.meta_df.loc[b]

        garment_a = str(meta_a.get("garment_group_name", "") or "")
        garment_b = str(meta_b.get("garment_group_name", "") or "")
        if garment_a in self.UNWANTED_GROUPS or garment_b in self.UNWANTED_GROUPS:
            return False

        index_a = str(meta_a.get("index_name", "") or "")
        index_b = str(meta_b.get("index_name", "") or "")
        if index_a and index_b and index_a != index_b:
            return False

        if mode == "minimal":
            return True

        product_group_a = str(meta_a.get("product_group_name", "") or "")
        product_group_b = str(meta_b.get("product_group_name", "") or "")
        garment_group_a = str(meta_a.get("garment_group_name", "") or "")
        garment_group_b = str(meta_b.get("garment_group_name", "") or "")
        product_type_a = str(meta_a.get("product_type_name", "") or "")
        product_type_b = str(meta_b.get("product_type_name", "") or "")

        if mode == "redesigned":
            prod_name_a = str(meta_a.get("prod_name", "") or "")
            prod_name_b = str(meta_b.get("prod_name", "") or "")
            section_a = str(meta_a.get("section_name", "") or "")
            section_b = str(meta_b.get("section_name", "") or "")
            dept_a = str(meta_a.get("department_name", "") or "")
            dept_b = str(meta_b.get("department_name", "") or "")
            slot_a = get_slot(product_type_a, product_group_a, garment_group_a, prod_name_a, section_a, dept_a)
            slot_b = get_slot(product_type_b, product_group_b, garment_group_b, prod_name_b, section_b, dept_b)
            if slot_a in {"other", "nightwear", "inner"} or slot_b in {"other", "nightwear", "inner"}:
                return False
            if not slot_pair_allowed(slot_a, slot_b):
                return False
            if product_type_a and product_type_a == product_type_b:
                return False
            if same_pt_family(product_type_a, product_type_b):
                return False
            if a[:6] == b[:6]:
                return False
            colour_a = str(meta_a.get("colour_group_name", "") or "")
            colour_b = str(meta_b.get("colour_group_name", "") or "")
            if not self._colours_compatible(colour_a, colour_b, garment_group_a, garment_group_b):
                return False
            return True

        allowed_same = product_group_a in self.ALLOWED_SAME_GROUPS

        if product_type_a and product_type_a == product_type_b and not allowed_same:
            return False
        if (product_group_a == product_group_b) and (garment_group_a == garment_group_b) and not allowed_same:
            return False
        if a[:6] == b[:6]:
            return False

        if mode == "standard":
            return True

        if mode == "legacy":
            colour_a = str(meta_a.get("colour_group_name", "") or "")
            colour_b = str(meta_b.get("colour_group_name", "") or "")
            if not self._colours_compatible(colour_a, colour_b, garment_group_a, garment_group_b):
                return False
            season_a = str(meta_a.get("seasonality", "") or "")
            season_b = str(meta_b.get("seasonality", "") or "")
            if not self._seasons_compatible(season_a, season_b):
                return False
            section_a = str(meta_a.get("section_name", "") or "")
            section_b = str(meta_b.get("section_name", "") or "")
            if self._sections_conflict(section_a, section_b):
                return False
            return True

        return True

    def _colours_compatible(self, color_a: str, color_b: str, garment_a: str, garment_b: str) -> bool:
        if not color_a or not color_b:
            return True
        if color_a == color_b:
            return True
        if color_a in self.NEUTRALS or color_b in self.NEUTRALS:
            return True
        if color_a in self.NAVY_BLUES or color_b in self.NAVY_BLUES:
            return True
        if garment_a == "Trousers Denim" or garment_b == "Trousers Denim":
            return True
        for palette in (self.BLUES, self.GREENS, self.PINKS_REDS, self.YELLOWS_ORANGES, self.BROWNS):
            if color_a in palette and color_b in palette:
                return True
        earth_neighbours = self.GREENS | self.YELLOWS_ORANGES
        if (color_a in self.BROWNS and color_b in earth_neighbours) or (color_b in self.BROWNS and color_a in earth_neighbours):
            return True
        if (color_a in self.BLUES and color_b in self.PINKS_REDS) or (color_b in self.BLUES and color_a in self.PINKS_REDS):
            return True
        return False

    def _seasons_compatible(self, season_a: str, season_b: str) -> bool:
        lower_a = season_a.lower()
        lower_b = season_b.lower()
        a_winter = any(k in lower_a for k in self.WINTER_KEYWORDS)
        a_summer = any(k in lower_a for k in self.SUMMER_KEYWORDS)
        b_winter = any(k in lower_b for k in self.WINTER_KEYWORDS)
        b_summer = any(k in lower_b for k in self.SUMMER_KEYWORDS)
        a_all = any(k in lower_a for k in self.ALL_SEASON_KEYWORDS)
        b_all = any(k in lower_b for k in self.ALL_SEASON_KEYWORDS)
        if a_all or b_all:
            return True
        if a_winter and b_summer:
            return False
        if a_summer and b_winter:
            return False
        return True

    def _sections_conflict(self, section_a: str, section_b: str) -> bool:
        a_sport = section_a in self.SPORT_SECTIONS
        b_sport = section_b in self.SPORT_SECTIONS
        a_formal = section_a in self.FORMAL_SECTIONS
        b_formal = section_b in self.FORMAL_SECTIONS
        return (a_sport and b_formal) or (a_formal and b_sport)

    def _npmi(self, count_ab: int, count_a: int, count_b: int, n_baskets: int) -> float:
        if count_ab <= 0 or count_a <= 0 or count_b <= 0 or n_baskets <= 0:
            return -1.0
        p_ab = count_ab / n_baskets
        p_a = count_a / n_baskets
        p_b = count_b / n_baskets
        pmi = math.log(p_ab / (p_a * p_b))
        return pmi / -math.log(p_ab)

    def select_edges(self) -> List[Tuple[str, str, float]]:
        if self.pair_counts is None or self.item_counts is None:
            self.compute_pair_counts()

        edges: List[Tuple[str, str, float]] = []
        method = self.config.method.lower()
        n = self.total_baskets_observed

        for (a, b), count_ab in self.pair_counts.items():
            if count_ab < self.config.min_cooc:
                continue
            if not self._passes_content_filter(a, b, self.config.content_filter):
                continue
            if method == "cobuy":
                weight = float(count_ab)
            elif method == "npmi":
                score = self._npmi(count_ab, self.item_counts[a], self.item_counts[b], n)
                if score < self.config.npmi_threshold:
                    continue
                weight = float(score)
            else:
                raise ValueError(f"unknown method: {self.config.method}")
            edges.append((a, b, weight))

        log.info("selected %d edges (method=%s, min_cooc=%d, threshold=%.4f)",
                 len(edges), method, self.config.min_cooc, self.config.npmi_threshold)
        return edges

    def write(self, edges: List[Tuple[str, str, float]]) -> None:
        bidirectional: Dict[Tuple[str, str], float] = {}
        for a, b, weight in edges:
            scaled = weight * self.config.weight_scale if self.config.method == "npmi" else weight
            bidirectional[(a, b)] = scaled
            bidirectional[(b, a)] = scaled

        df = pd.DataFrame(
            [(a, b, w) for (a, b), w in bidirectional.items()],
            columns=["item_a", "item_b", "weight"],
        )
        df = df.sort_values(by=["item_a", "weight"], ascending=[True, False])

        out_path = Path(self.config.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info("wrote graph: %s rows=%d", out_path, len(df))

    def run(self) -> None:
        start = time.time()
        self.load_meta()
        self.build_baskets()
        del_baskets = self.baskets
        self.compute_pair_counts()
        self.baskets = None
        del del_baskets
        gc.collect()

        edges = self.select_edges()
        self.write(edges)
        log.info("done in %.1fs", time.time() - start)


def _config_from_env() -> BuilderConfig:
    base = Path(__file__).resolve().parent.parent.parent
    return BuilderConfig(
        transactions_file=str(base / os.getenv("TRANS_FILE", "data/raw/transactions_train.csv")),
        meta_file=str(base / os.getenv("META_FILE", "data/processed/dataset_final_qwen_filled.csv")),
        output_file=str(base / os.getenv("OUTPUT_FILE", "data/processed/final_outfit_graph.csv")),
        method=os.getenv("GRAPH_METHOD", "npmi"),
        min_cooc=int(os.getenv("GRAPH_MIN_COOC", "3")),
        npmi_threshold=float(os.getenv("GRAPH_NPMI_THRESHOLD", "0.0")),
        weight_scale=int(os.getenv("GRAPH_WEIGHT_SCALE", "1000")),
        train_cutoff_date=os.getenv("GRAPH_TRAIN_CUTOFF", "") or None,
        content_filter=os.getenv("GRAPH_CONTENT_FILTER", "minimal"),
        basket_cache_pickle=os.getenv("GRAPH_BASKET_CACHE", "") or None,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    config = _config_from_env()
    builder = OutfitGraphBuilder(config)
    builder.run()


if __name__ == "__main__":
    main()
