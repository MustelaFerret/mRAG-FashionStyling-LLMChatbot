"""Milestone 0 — data gate for the compatibility-embedding task.

Exports four artifacts shared by every downstream model (linear metric / MLP head /
GraphSAGE):
  - node_features.npy : SigLIP image_emb [N, 768] (frozen) for every indexed item
  - node_ids.json     : article_id per row + article_id -> row index map
  - node_slots.json   : 4-tier outfit slot per row (for hard-negative mining + filter)
  - edges.npz         : undirected co-buy edge_index [2, E] + weight [E] + train/val/test split

Run: python -m src.scripts.compat.build_compat_dataset
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from src.backend.core.config import settings
from src.backend.core.utils import normalize_article_id
from src.backend.retrieval.qdrant import QdrantStore
from src.scripts.graph.outfit_slots import get_slot

OUTPUT_DIR = os.path.join(os.path.dirname(settings.meta_file), "compat")
SCROLL_BATCH = 4096
SPLIT_SEED = 42
VAL_RATIO = 0.075
TEST_RATIO = 0.075


class CompatDatasetBuilder:
    def __init__(self, vector_name: str = ""):
        self.vector_name = vector_name or settings.vector_name_image
        self.store = QdrantStore(settings.db_path, settings.collection_name)
        self.article_ids: List[str] = []
        self.aid_to_idx: Dict[str, int] = {}
        self.features: np.ndarray | None = None
        self.slots: List[str] = []

    def _scroll_nodes(self) -> None:
        ids: List[str] = []
        vectors: List[List[float]] = []
        offset = None
        while True:
            points, offset = self.store.client.scroll(
                collection_name=settings.collection_name,
                limit=SCROLL_BATCH,
                with_payload=True,
                with_vectors=[self.vector_name],
                offset=offset,
            )
            for point in points:
                vec = point.vector.get(self.vector_name) if isinstance(point.vector, dict) else None
                if vec is None:
                    continue
                aid = normalize_article_id(str((point.payload or {}).get("article_id", "")))
                if not aid:
                    continue
                ids.append(aid)
                vectors.append(list(vec))
            if offset is None:
                break
        self.article_ids = ids
        self.aid_to_idx = {aid: i for i, aid in enumerate(ids)}
        self.features = np.asarray(vectors, dtype=np.float32)

    def _load_slots(self) -> None:
        slot_by_aid: Dict[str, str] = {}
        with open(settings.meta_file, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                aid = normalize_article_id(str(row.get("article_id", "")))
                if not aid:
                    continue
                slot_by_aid[aid] = get_slot(
                    product_type=str(row.get("product_type_name", "") or ""),
                    product_group=str(row.get("product_group_name", "") or ""),
                    garment_group=str(row.get("garment_group_name", "") or ""),
                    prod_name=str(row.get("prod_name", "") or ""),
                    section_name=str(row.get("section_name", "") or ""),
                    department_name=str(row.get("department_name", "") or ""),
                )
        self.slots = [slot_by_aid.get(aid, "other") for aid in self.article_ids]

    def _load_edges(self) -> Tuple[np.ndarray, np.ndarray, int]:
        seen: Dict[Tuple[int, int], float] = {}
        dropped = 0
        with open(settings.graph_file, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                a = normalize_article_id(str(row.get("item_a", "")))
                b = normalize_article_id(str(row.get("item_b", "")))
                ia = self.aid_to_idx.get(a)
                ib = self.aid_to_idx.get(b)
                if ia is None or ib is None or ia == ib:
                    dropped += 1
                    continue
                try:
                    w = float(row.get("weight", 0.0) or 0.0)
                except (TypeError, ValueError):
                    w = 0.0
                key = (ia, ib) if ia < ib else (ib, ia)
                if w > seen.get(key, 0.0):
                    seen[key] = w
        if not seen:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32), dropped
        pairs = np.asarray(list(seen.keys()), dtype=np.int64)
        weights = np.asarray(list(seen.values()), dtype=np.float32)
        edge_index = pairs.T
        return edge_index, weights, dropped

    @staticmethod
    def _split_edges(num_edges: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(SPLIT_SEED)
        perm = rng.permutation(num_edges)
        n_val = int(num_edges * VAL_RATIO)
        n_test = int(num_edges * TEST_RATIO)
        test_idx = np.sort(perm[:n_test])
        val_idx = np.sort(perm[n_test:n_test + n_val])
        train_idx = np.sort(perm[n_test + n_val:])
        return train_idx, val_idx, test_idx

    def build(self) -> Dict:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self._scroll_nodes()
        self._load_slots()
        edge_index, edge_weight, dropped = self._load_edges()
        train_idx, val_idx, test_idx = self._split_edges(edge_index.shape[1])

        np.save(os.path.join(OUTPUT_DIR, "node_features.npy"), self.features)
        with open(os.path.join(OUTPUT_DIR, "node_ids.json"), "w", encoding="utf-8") as f:
            json.dump({"article_ids": self.article_ids, "aid_to_idx": self.aid_to_idx}, f)
        with open(os.path.join(OUTPUT_DIR, "node_slots.json"), "w", encoding="utf-8") as f:
            json.dump(self.slots, f)
        np.savez(
            os.path.join(OUTPUT_DIR, "edges.npz"),
            edge_index=edge_index,
            edge_weight=edge_weight,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

        node_set = set(range(len(self.article_ids)))
        nodes_with_edges = set(edge_index.flatten().tolist()) if edge_index.shape[1] else set()
        slot_counts: Dict[str, int] = {}
        for s in self.slots:
            slot_counts[s] = slot_counts.get(s, 0) + 1
        meta = {
            "num_nodes": len(self.article_ids),
            "feature_dim": int(self.features.shape[1]) if self.features is not None else 0,
            "num_edges": int(edge_index.shape[1]),
            "edges_dropped_unmapped": dropped,
            "nodes_in_graph": len(nodes_with_edges),
            "nodes_cold": len(node_set - nodes_with_edges),
            "graph_coverage_pct": round(100.0 * len(nodes_with_edges) / max(1, len(node_set)), 2),
            "edge_weight_min": float(edge_weight.min()) if edge_weight.size else 0.0,
            "edge_weight_median": float(np.median(edge_weight)) if edge_weight.size else 0.0,
            "edge_weight_max": float(edge_weight.max()) if edge_weight.size else 0.0,
            "split": {"train": int(train_idx.size), "val": int(val_idx.size), "test": int(test_idx.size)},
            "slot_distribution": dict(sorted(slot_counts.items(), key=lambda x: -x[1])),
            "split_seed": SPLIT_SEED,
            "vector_name": self.vector_name,
        }
        with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return meta


def main() -> None:
    meta = CompatDatasetBuilder().build()
    print(f"[compat] artifacts -> {OUTPUT_DIR}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
