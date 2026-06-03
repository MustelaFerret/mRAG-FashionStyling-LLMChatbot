from __future__ import annotations

import csv
import logging
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.scripts.graph.eval_graph import AdjacencyMap, OutfitGraphLoader

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphProfile:
    num_nodes: int
    num_edges_directed: int
    num_unique_undirected_edges: int
    nodes_in_meta: int
    meta_coverage: float
    degree_quantiles: Dict[str, int]
    weight_quantiles: Dict[str, float]
    top_degree_nodes: List[Tuple[str, int, str]]
    intra_product_type_ratio: float
    intra_product_group_ratio: float
    intra_color_ratio: float


class GraphProfiler:
    def __init__(self, meta_file: str):
        self.meta_file = meta_file
        self._meta_by_article: Dict[str, Dict[str, str]] = {}
        self._load_meta()

    def _load_meta(self) -> None:
        df = pd.read_csv(
            self.meta_file,
            usecols=["article_id", "product_type_name", "product_group_name", "colour_group_name"],
            dtype=str,
        )
        df["article_id"] = df["article_id"].str.zfill(10)
        for _, row in df.iterrows():
            self._meta_by_article[row["article_id"]] = {
                "product_type": str(row.get("product_type_name", "") or ""),
                "product_group": str(row.get("product_group_name", "") or ""),
                "colour_group": str(row.get("colour_group_name", "") or ""),
            }
        log.info("meta loaded: %d articles", len(self._meta_by_article))

    @staticmethod
    def _quantiles(values: List[float], qs: Tuple[float, ...] = (0.5, 0.9, 0.99)) -> Dict[str, float]:
        if not values:
            return {f"p{int(q * 100)}": 0.0 for q in qs}
        sorted_values = sorted(values)
        out: Dict[str, float] = {}
        for q in qs:
            idx = min(len(sorted_values) - 1, int(q * len(sorted_values)))
            out[f"p{int(q * 100)}"] = float(sorted_values[idx])
        out["min"] = float(sorted_values[0])
        out["max"] = float(sorted_values[-1])
        out["mean"] = float(statistics.fmean(sorted_values))
        return out

    def profile(self, adjacency: AdjacencyMap) -> GraphProfile:
        nodes = list(adjacency.keys())
        degrees = [len(adjacency[n]) for n in nodes]
        all_weights: List[float] = []
        intra_pt = 0
        intra_pg = 0
        intra_color = 0
        undirected_pairs: set[Tuple[str, str]] = set()
        directed_edges = 0
        for node, neighbors in adjacency.items():
            node_meta = self._meta_by_article.get(node, {})
            node_pt = node_meta.get("product_type", "")
            node_pg = node_meta.get("product_group", "")
            node_color = node_meta.get("colour_group", "")
            for neighbor, weight in neighbors:
                directed_edges += 1
                all_weights.append(weight)
                pair = (node, neighbor) if node < neighbor else (neighbor, node)
                undirected_pairs.add(pair)
                neighbor_meta = self._meta_by_article.get(neighbor, {})
                if node_pt and node_pt == neighbor_meta.get("product_type", ""):
                    intra_pt += 1
                if node_pg and node_pg == neighbor_meta.get("product_group", ""):
                    intra_pg += 1
                if node_color and node_color == neighbor_meta.get("colour_group", ""):
                    intra_color += 1

        total_edges = max(1, directed_edges)
        degree_q = self._quantiles([float(d) for d in degrees])
        weight_q = self._quantiles(all_weights)
        degree_quantiles_int = {k: int(v) for k, v in degree_q.items()}

        ranked_by_degree = sorted(adjacency.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
        top_degree_nodes = [
            (node, len(neighbors), self._meta_by_article.get(node, {}).get("product_type", ""))
            for node, neighbors in ranked_by_degree
        ]

        nodes_in_meta = sum(1 for node in nodes if node in self._meta_by_article)

        return GraphProfile(
            num_nodes=len(nodes),
            num_edges_directed=directed_edges,
            num_unique_undirected_edges=len(undirected_pairs),
            nodes_in_meta=nodes_in_meta,
            meta_coverage=nodes_in_meta / max(1, len(nodes)),
            degree_quantiles=degree_quantiles_int,
            weight_quantiles=weight_q,
            top_degree_nodes=top_degree_nodes,
            intra_product_type_ratio=intra_pt / total_edges,
            intra_product_group_ratio=intra_pg / total_edges,
            intra_color_ratio=intra_color / total_edges,
        )

    @staticmethod
    def format_report(profile: GraphProfile) -> str:
        lines: List[str] = []
        lines.append("## Graph profile")
        lines.append("")
        lines.append("### Size")
        lines.append(f"- nodes: {profile.num_nodes}")
        lines.append(f"- directed edges: {profile.num_edges_directed}")
        lines.append(f"- undirected unique edges: {profile.num_unique_undirected_edges}")
        lines.append(f"- nodes with meta: {profile.nodes_in_meta} ({profile.meta_coverage:.4f})")
        lines.append("")
        lines.append("### Degree")
        lines.append(
            f"- mean={profile.degree_quantiles.get('mean', 0)} "
            f"p50={profile.degree_quantiles.get('p50', 0)} "
            f"p90={profile.degree_quantiles.get('p90', 0)} "
            f"p99={profile.degree_quantiles.get('p99', 0)} "
            f"max={profile.degree_quantiles.get('max', 0)}"
        )
        lines.append("")
        lines.append("### Weight")
        lines.append(
            f"- mean={profile.weight_quantiles.get('mean', 0.0):.4f} "
            f"p50={profile.weight_quantiles.get('p50', 0.0):.4f} "
            f"p90={profile.weight_quantiles.get('p90', 0.0):.4f} "
            f"p99={profile.weight_quantiles.get('p99', 0.0):.4f} "
            f"max={profile.weight_quantiles.get('max', 0.0):.4f}"
        )
        lines.append("")
        lines.append("### Top-10 highest-degree nodes")
        for node, deg, pt in profile.top_degree_nodes:
            lines.append(f"- {node} degree={deg} product_type={pt}")
        lines.append("")
        lines.append("### Pairing nature (lower = more diverse pairings)")
        lines.append(f"- intra_product_type_ratio: {profile.intra_product_type_ratio:.4f}")
        lines.append(f"- intra_product_group_ratio: {profile.intra_product_group_ratio:.4f}")
        lines.append(f"- intra_color_ratio: {profile.intra_color_ratio:.4f}")
        return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    base = Path(__file__).resolve().parent.parent.parent
    graph_file = str(base / os.getenv("GRAPH_FILE", "data/processed/final_outfit_graph.csv"))
    meta_file = str(base / os.getenv("META_FILE", "data/processed/dataset_final_qwen_filled.csv"))

    adjacency = OutfitGraphLoader.from_csv(graph_file)
    log.info("graph loaded: nodes=%d", len(adjacency))

    profiler = GraphProfiler(meta_file)
    profile = profiler.profile(adjacency)
    print(GraphProfiler.format_report(profile))


if __name__ == "__main__":
    main()
