"""Milestone 1 — evaluation harness + baselines for compatibility pairing.

Link-prediction on held-out co-buy edges (test split). For each test edge it builds
two eval instances (i->j and j->i) so both warm and cold anchors are covered. The
target is ranked against a pool of {target} + sampled negatives, excluding the
anchor's known partners.

Negative modes:
  - hard_random : 50% share target's slot (hard) + 50% random   [default]
  - pop_matched : negatives drawn from the target's degree band  -> neutralises
                  popularity bias (Krichene & Rendle 2020), measures true compatibility

Baselines (no training): cobuy (train-graph weight), siglip (frozen image_emb cosine),
popularity (train degree). Extra learned embeddings can be passed in to score as cosine.

Metrics: HitRate@{1,5,10}, MRR, AUC (tie-aware). Reported overall + warm/cold anchor.

Run: python -m src.scripts.compat.eval_pairing
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from src.backend.core.config import settings

DATA_DIR = os.path.join(os.path.dirname(settings.meta_file), "compat")
NUM_NEG = 100
HARD_FRACTION = 0.5
SEED = 42
K_VALUES = (1, 5, 10)


class PairingEvaluator:
    def __init__(self, data_dir: str = DATA_DIR, extra_methods: Dict[str, np.ndarray] | None = None,
                 neg_mode: str = "hard_random"):
        self.X = np.load(os.path.join(data_dir, "node_features.npy"))
        self.slots = json.load(open(os.path.join(data_dir, "node_slots.json"), encoding="utf-8"))
        edges = np.load(os.path.join(data_dir, "edges.npz"))
        self.edge_index = edges["edge_index"]
        self.train_idx = edges["train_idx"]
        self.test_idx = edges["test_idx"]
        self.num_nodes = self.X.shape[0]
        self.extra_methods = extra_methods or {}
        self.neg_mode = neg_mode
        self.method_names = ["cobuy", "siglip", "popularity"] + list(self.extra_methods.keys())
        self.rng = np.random.default_rng(SEED)

        self._build_train_graph()
        self._build_full_partners()
        self._build_slot_index()
        self._build_degree_bins()

    def _build_train_graph(self) -> None:
        self.train_adj: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.degree = np.zeros(self.num_nodes, dtype=np.float32)
        ei = self.edge_index[:, self.train_idx]
        for a, b in ei.T.tolist():
            self.train_adj[a][b] = 1.0
            self.train_adj[b][a] = 1.0
            self.degree[a] += 1.0
            self.degree[b] += 1.0

    def _build_full_partners(self) -> None:
        self.full_partners: Dict[int, set] = defaultdict(set)
        for a, b in self.edge_index.T.tolist():
            self.full_partners[a].add(b)
            self.full_partners[b].add(a)

    def _build_slot_index(self) -> None:
        self.slot_to_nodes: Dict[str, np.ndarray] = {}
        buckets: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(self.slots):
            buckets[s].append(i)
        for s, nodes in buckets.items():
            self.slot_to_nodes[s] = np.asarray(nodes, dtype=np.int64)

    def _build_degree_bins(self) -> None:
        # log2 degree bins for popularity-matched negative sampling.
        self.node_bin = np.floor(np.log2(self.degree + 1.0)).astype(np.int64)
        self.bin_to_nodes: Dict[int, np.ndarray] = {}
        buckets: Dict[int, List[int]] = defaultdict(list)
        for i, b in enumerate(self.node_bin.tolist()):
            buckets[b].append(i)
        for b, nodes in buckets.items():
            self.bin_to_nodes[b] = np.asarray(nodes, dtype=np.int64)

    def _draw(self, pool: np.ndarray, exclude: set, taken: set, want: int, out: List[int]) -> None:
        if pool is None or pool.size == 0:
            return
        picks = self.rng.choice(pool, size=min(want * 4, pool.size), replace=False)
        for c in picks.tolist():
            if c not in exclude and c not in taken:
                out.append(c)
                taken.add(c)
            if len(out) >= want:
                break

    def _sample_negatives(self, anchor: int, target: int) -> np.ndarray:
        exclude = self.full_partners.get(anchor, set()) | {anchor, target}
        taken: set = set()
        negs: List[int] = []

        if self.neg_mode == "pop_matched":
            self._draw(self.bin_to_nodes.get(int(self.node_bin[target])), exclude, taken, NUM_NEG, negs)
        elif self.neg_mode == "same_slot":
            # All negatives share the target's slot -> removes slot-level signal,
            # isolates pure item-level compatibility within the complement category.
            self._draw(self.slot_to_nodes.get(self.slots[target]), exclude, taken, NUM_NEG, negs)
        else:
            self._draw(self.slot_to_nodes.get(self.slots[target]), exclude, taken, int(NUM_NEG * HARD_FRACTION), negs)

        need = NUM_NEG - len(negs)
        while need > 0:
            picks = self.rng.integers(0, self.num_nodes, size=need * 2)
            for c in picks.tolist():
                if c not in exclude and c not in taken:
                    negs.append(c)
                    taken.add(c)
                if len(negs) >= NUM_NEG:
                    break
            need = NUM_NEG - len(negs)
        return np.asarray(negs[:NUM_NEG], dtype=np.int64)

    def _scores(self, anchor: int, candidates: np.ndarray) -> Dict[str, np.ndarray]:
        out = {
            "cobuy": np.array([self.train_adj.get(anchor, {}).get(int(c), 0.0) for c in candidates], dtype=np.float32),
            "siglip": self.X[candidates] @ self.X[anchor],
            "popularity": self.degree[candidates],
        }
        for name, method in self.extra_methods.items():
            # an extra method is either a per-node embedding (scored by dot product) or a callable
            # scorer fn(anchor_idx, candidate_idxs) -> scores (for pairwise / type-conditioned models).
            if callable(method):
                out[name] = np.asarray(method(anchor, candidates), dtype=np.float32)
            else:
                out[name] = method[candidates] @ method[anchor]
        return out

    @staticmethod
    def _rank_of_target(scores: np.ndarray) -> float:
        target = scores[0]
        greater = int(np.sum(scores[1:] > target))
        ties = int(np.sum(scores[1:] == target))
        return greater + 1 + ties / 2.0

    @staticmethod
    def _auc(scores: np.ndarray) -> float:
        target = scores[0]
        negs = scores[1:]
        wins = np.sum(negs < target) + 0.5 * np.sum(negs == target)
        return float(wins / max(1, negs.size))

    def evaluate(self) -> Dict:
        agg = {g: self._new_acc() for g in ("overall", "warm", "cold")}
        for i, j in self.edge_index[:, self.test_idx].T.tolist():
            for anchor, target in ((i, j), (j, i)):
                candidates = np.concatenate(([target], self._sample_negatives(anchor, target)))
                bucket = "warm" if self.degree[anchor] > 0 else "cold"
                for method, sc in self._scores(anchor, candidates).items():
                    rank = self._rank_of_target(sc)
                    auc = self._auc(sc)
                    for group in ("overall", bucket):
                        self._record(agg[group][method], rank, auc)
        return {g: {m: self._finalize(acc) for m, acc in methods.items()} for g, methods in agg.items()}

    def _new_acc(self) -> Dict[str, Dict[str, float]]:
        return {m: {"n": 0, "mrr": 0.0, "auc": 0.0, **{f"hit@{k}": 0.0 for k in K_VALUES}}
                for m in self.method_names}

    @staticmethod
    def _record(acc: Dict[str, float], rank: float, auc: float) -> None:
        acc["n"] += 1
        acc["mrr"] += 1.0 / rank
        acc["auc"] += auc
        for k in K_VALUES:
            if rank <= k:
                acc[f"hit@{k}"] += 1.0

    @staticmethod
    def _finalize(acc: Dict[str, float]) -> Dict[str, float]:
        n = max(1, acc["n"])
        out = {"n": int(acc["n"]), "mrr": round(acc["mrr"] / n, 4), "auc": round(acc["auc"] / n, 4)}
        for k in K_VALUES:
            out[f"hit@{k}"] = round(acc[f"hit@{k}"] / n, 4)
        return out


def main() -> None:
    compat_path = os.path.join(DATA_DIR, "compat_emb.npy")
    extra = {"compat": np.load(compat_path)} if os.path.exists(compat_path) else None
    for mode in ("hard_random", "pop_matched", "same_slot"):
        report = PairingEvaluator(extra_methods=extra, neg_mode=mode).evaluate()
        with open(os.path.join(DATA_DIR, f"eval_baselines_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump({"neg_mode": mode, "num_neg": NUM_NEG, "report": report}, f, indent=2)
        print(f"\n===== neg_mode={mode} =====")
        for group in ("warm", "cold"):
            print(f"[{group}]")
            for m, v in report[group].items():
                print(f"  {m:11s} AUC={v['auc']:.3f} MRR={v['mrr']:.3f} H@1={v['hit@1']:.3f} H@10={v['hit@10']:.3f} (n={v['n']})")


if __name__ == "__main__":
    main()
