from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import pickle
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

log = logging.getLogger(__name__)

AdjacencyMap = Dict[str, List[Tuple[str, float]]]


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    recall_at_k: Dict[int, float]
    map_at_k: Dict[int, float]
    hit_at_k: Dict[int, float]
    coverage: float
    anchors_with_neighbors: int
    anchors_total: int
    num_test_baskets: int


@dataclass
class EvalConfig:
    transactions_file: str
    meta_file: str | None = None
    k_values: Tuple[int, ...] = (5, 10, 20)
    test_ratio: float = 0.2
    basket_min_size: int = 2
    basket_max_size: int = 10
    max_test_baskets: int = 30_000
    seed: int = 42
    cache_dir: str | None = None
    ground_truth_filter: str = "all"

    def cache_signature(self) -> str:
        payload = {
            "transactions_file": self.transactions_file,
            "meta_file": self.meta_file,
            "test_ratio": self.test_ratio,
            "basket_min_size": self.basket_min_size,
            "basket_max_size": self.basket_max_size,
            "max_test_baskets": self.max_test_baskets,
            "seed": self.seed,
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]


class OutfitGraphLoader:
    @staticmethod
    def from_csv(path: str) -> AdjacencyMap:
        adjacency: Dict[str, Dict[str, float]] = defaultdict(dict)
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                a = str(row.get("item_a", "") or "").strip().zfill(10)
                b = str(row.get("item_b", "") or "").strip().zfill(10)
                if not a or not b or a == b:
                    continue
                try:
                    w = float(row.get("weight", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if w > adjacency[a].get(b, 0.0):
                    adjacency[a][b] = w
                if w > adjacency[b].get(a, 0.0):
                    adjacency[b][a] = w
        return {
            node: sorted(neighbors.items(), key=lambda kv: kv[1], reverse=True)
            for node, neighbors in adjacency.items()
        }


class OutfitGraphEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self._valid_ids: set[str] | None = None
        self._train_baskets: List[List[str]] | None = None
        self._test_baskets: List[List[str]] | None = None
        self._cutoff_date: str | None = None
        self._product_type_by_article: Dict[str, str] = {}

    def load_product_types(self, meta_file: str) -> None:
        df = pd.read_csv(meta_file, usecols=["article_id", "product_type_name"], dtype=str)
        df["article_id"] = df["article_id"].str.zfill(10)
        self._product_type_by_article = dict(zip(df["article_id"], df["product_type_name"].fillna("")))
        log.info("loaded product_type for %d articles", len(self._product_type_by_article))

    def _cache_path(self) -> Path | None:
        if not self.config.cache_dir:
            return None
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"eval_baskets_{self.config.cache_signature()}.pkl"

    def _try_load_cache(self) -> bool:
        cache_path = self._cache_path()
        if cache_path is None or not cache_path.exists():
            return False
        try:
            with open(cache_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            return False
        self._train_baskets = payload["train_baskets"]
        self._test_baskets = payload["test_baskets"]
        self._cutoff_date = payload["cutoff_date"]
        self._valid_ids = payload.get("valid_ids")
        log.info(
            "loaded baskets from cache: train=%d test=%d cutoff=%s",
            len(self._train_baskets or []),
            len(self._test_baskets or []),
            self._cutoff_date,
        )
        return True

    def _write_cache(self) -> None:
        cache_path = self._cache_path()
        if cache_path is None:
            return
        payload = {
            "train_baskets": self._train_baskets,
            "test_baskets": self._test_baskets,
            "cutoff_date": self._cutoff_date,
            "valid_ids": self._valid_ids,
        }
        with open(cache_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("wrote baskets cache: %s", cache_path)

    def prepare(self, use_cache: bool = True) -> None:
        if use_cache and self._try_load_cache():
            return

        if self.config.meta_file and os.path.exists(self.config.meta_file):
            meta_df = pd.read_csv(self.config.meta_file, usecols=["article_id"], dtype={"article_id": str})
            self._valid_ids = set(meta_df["article_id"].str.zfill(10))
            log.info("loaded %d valid article ids from meta", len(self._valid_ids))

        df = pd.read_csv(
            self.config.transactions_file,
            usecols=["t_dat", "customer_id", "article_id"],
            dtype={"article_id": str, "customer_id": "category", "t_dat": "string"},
        )
        df["article_id"] = df["article_id"].str.zfill(10)

        if self._valid_ids is not None:
            df = df[df["article_id"].isin(self._valid_ids)]

        df = df.drop_duplicates(subset=["customer_id", "t_dat", "article_id"])

        unique_dates = sorted(df["t_dat"].unique())
        if not unique_dates:
            raise ValueError("no transactions found")

        split_idx = max(1, int(len(unique_dates) * (1.0 - self.config.test_ratio)))
        self._cutoff_date = str(unique_dates[split_idx - 1])
        train_mask = df["t_dat"] <= self._cutoff_date
        log.info(
            "temporal split: cutoff=%s, train_rows=%d, test_rows=%d",
            self._cutoff_date,
            int(train_mask.sum()),
            int((~train_mask).sum()),
        )

        train_baskets = self._build_baskets(df[train_mask])
        test_baskets = self._build_baskets(df[~train_mask])

        rng = random.Random(self.config.seed)
        rng.shuffle(test_baskets)
        if len(test_baskets) > self.config.max_test_baskets:
            test_baskets = test_baskets[: self.config.max_test_baskets]

        self._train_baskets = train_baskets
        self._test_baskets = test_baskets
        log.info("train_baskets=%d test_baskets=%d", len(train_baskets), len(test_baskets))
        self._write_cache()

    def _build_baskets(self, df: pd.DataFrame) -> List[List[str]]:
        grouped = df.groupby(["customer_id", "t_dat"], observed=True)["article_id"].apply(list)
        baskets: List[List[str]] = []
        for items in grouped:
            unique_items = list(dict.fromkeys(items))
            size = len(unique_items)
            if size < self.config.basket_min_size or size > self.config.basket_max_size:
                continue
            baskets.append(unique_items)
        return baskets

    @property
    def train_baskets(self) -> List[List[str]]:
        if self._train_baskets is None:
            raise RuntimeError("prepare() must be called first")
        return self._train_baskets

    @property
    def test_baskets(self) -> List[List[str]]:
        if self._test_baskets is None:
            raise RuntimeError("prepare() must be called first")
        return self._test_baskets

    @property
    def cutoff_date(self) -> str:
        if self._cutoff_date is None:
            raise RuntimeError("prepare() must be called first")
        return self._cutoff_date

    def evaluate(self, adjacency: AdjacencyMap, name: str = "graph") -> EvaluationResult:
        if self._test_baskets is None:
            raise RuntimeError("prepare() must be called first")
        gt_mode = self.config.ground_truth_filter
        if gt_mode == "cross_product_type" and not self._product_type_by_article:
            if self.config.meta_file:
                self.load_product_types(self.config.meta_file)
            else:
                raise RuntimeError("ground_truth_filter=cross_product_type requires meta_file")

        k_values = sorted(self.config.k_values)
        max_k = max(k_values)
        recall_acc = {k: 0.0 for k in k_values}
        map_acc = {k: 0.0 for k in k_values}
        hit_acc = {k: 0 for k in k_values}
        anchors_with_neighbors = 0
        anchors_total = 0

        for basket in self._test_baskets:
            basket_set = set(basket)
            for anchor in basket:
                ground_truth = basket_set - {anchor}
                if gt_mode == "cross_product_type":
                    ground_truth = self._filter_cross_product_type(anchor, ground_truth)
                if not ground_truth:
                    continue
                anchors_total += 1
                neighbors = adjacency.get(anchor)
                if not neighbors:
                    continue
                anchors_with_neighbors += 1
                top_neighbors = [nid for nid, _ in neighbors[:max_k]]
                self._accumulate(top_neighbors, ground_truth, k_values, recall_acc, map_acc, hit_acc)

        denom = max(1, anchors_with_neighbors)
        recall_avg = {k: recall_acc[k] / denom for k in k_values}
        map_avg = {k: map_acc[k] / denom for k in k_values}
        hit_avg = {k: hit_acc[k] / denom for k in k_values}
        coverage = anchors_with_neighbors / max(1, anchors_total)

        return EvaluationResult(
            name=name,
            recall_at_k=recall_avg,
            map_at_k=map_avg,
            hit_at_k=hit_avg,
            coverage=coverage,
            anchors_with_neighbors=anchors_with_neighbors,
            anchors_total=anchors_total,
            num_test_baskets=len(self._test_baskets),
        )

    def _filter_cross_product_type(self, anchor: str, candidates: set[str]) -> set[str]:
        anchor_type = self._product_type_by_article.get(anchor, "")
        if not anchor_type:
            return set()
        return {item for item in candidates if self._product_type_by_article.get(item, "") != anchor_type}

    @staticmethod
    def _accumulate(
        ranked: Sequence[str],
        ground_truth: set[str],
        k_values: Sequence[int],
        recall_acc: Dict[int, float],
        map_acc: Dict[int, float],
        hit_acc: Dict[int, int],
    ) -> None:
        gt_size = len(ground_truth)
        precision_running_sum = 0.0
        relevant_so_far = 0
        for rank, nid in enumerate(ranked, start=1):
            if nid in ground_truth:
                relevant_so_far += 1
                precision_running_sum += relevant_so_far / rank
            for k in k_values:
                if rank == k:
                    recall_acc[k] += relevant_so_far / gt_size
                    map_acc[k] += precision_running_sum / min(k, gt_size)
                    if relevant_so_far > 0:
                        hit_acc[k] += 1
        max_rank = len(ranked)
        for k in k_values:
            if max_rank >= k:
                continue
            recall_acc[k] += relevant_so_far / gt_size
            map_acc[k] += precision_running_sum / min(k, gt_size)
            if relevant_so_far > 0:
                hit_acc[k] += 1

    @staticmethod
    def format_report(results: Sequence[EvaluationResult]) -> str:
        if not results:
            return ""
        k_values = sorted(results[0].recall_at_k.keys())
        header_metrics = ["coverage"] + [f"recall@{k}" for k in k_values] + [f"map@{k}" for k in k_values] + [f"hit@{k}" for k in k_values]
        header = "| graph | " + " | ".join(header_metrics) + " |"
        sep = "|" + "|".join(["---"] * (len(header_metrics) + 1)) + "|"
        rows = [header, sep]
        for r in results:
            cells = [f"{r.coverage:.4f}"]
            cells += [f"{r.recall_at_k[k]:.4f}" for k in k_values]
            cells += [f"{r.map_at_k[k]:.4f}" for k in k_values]
            cells += [f"{r.hit_at_k[k]:.4f}" for k in k_values]
            rows.append(f"| {r.name} | " + " | ".join(cells) + " |")
        return "\n".join(rows)


def _config_from_env() -> EvalConfig:
    base = Path(__file__).resolve().parent.parent.parent
    return EvalConfig(
        transactions_file=str(base / os.getenv("TRANS_FILE", "data/raw/transactions_train.csv")),
        meta_file=str(base / os.getenv("META_FILE", "data/processed/dataset_final_qwen_filled.csv")),
        k_values=tuple(int(k) for k in os.getenv("EVAL_K_VALUES", "5,10,20").split(",")),
        test_ratio=float(os.getenv("EVAL_TEST_RATIO", "0.2")),
        basket_min_size=int(os.getenv("EVAL_BASKET_MIN", "2")),
        basket_max_size=int(os.getenv("EVAL_BASKET_MAX", "10")),
        max_test_baskets=int(os.getenv("EVAL_MAX_TEST_BASKETS", "30000")),
        seed=int(os.getenv("EVAL_SEED", "42")),
        cache_dir=str(base / os.getenv("EVAL_CACHE_DIR", "data/processed/eval_cache")),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    config = _config_from_env()
    evaluator = OutfitGraphEvaluator(config)
    evaluator.prepare()

    base = Path(__file__).resolve().parent.parent.parent
    graph_file = str(base / os.getenv("GRAPH_FILE", "data/processed/final_outfit_graph.csv"))
    adjacency = OutfitGraphLoader.from_csv(graph_file)
    log.info("graph loaded: nodes=%d", len(adjacency))

    result = evaluator.evaluate(adjacency, name="baseline_cobuy")
    log.info("evaluation complete: %s", result)
    print(OutfitGraphEvaluator.format_report([result]))


if __name__ == "__main__":
    main()
