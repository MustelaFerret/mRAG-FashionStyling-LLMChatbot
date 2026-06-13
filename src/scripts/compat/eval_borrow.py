"""Experiment A — cold-start mitigation by NEIGHBOR BORROWING.

A cold anchor has no train edges, but its visually-nearest WARM item of the same slot
(its "twin") does. Hypothesis: the twin's co-buy partners are good pairings for the cold
anchor. Scored on the standard PairingEvaluator protocol:

  borrow1c : nearest twin's train edges (binary) + 0.01 * compat cosine tie-break
  borrow5c : similarity-weighted union of top-5 twins' edges + 0.01 * compat tie-break

Twins are same-slot warm nodes ranked by frozen SigLIP image cosine. Compared against
the shipped compat MLP on warm/cold splits, hard_random + pop_matched negatives.

Run: python -m src.scripts.compat.eval_borrow
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from src.scripts.compat.eval_pairing import DATA_DIR, PairingEvaluator

TOP_TWINS = 5
TIEBREAK = 0.01


class BorrowEvaluator(PairingEvaluator):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.compat = self.extra_methods["compat"]
        self._twin_cache: Dict[int, tuple] = {}
        # warm same-slot pools (twin candidates)
        self._warm_pool: Dict[str, np.ndarray] = {}
        warm = self.degree > 0
        for slot, nodes in self.slot_to_nodes.items():
            pool = nodes[warm[nodes]]
            if pool.size:
                self._warm_pool[slot] = pool

    def _twins(self, anchor: int):
        hit = self._twin_cache.get(anchor)
        if hit is not None:
            return hit
        pool = self._warm_pool.get(self.slots[anchor])
        if pool is None or pool.size == 0:
            out = (np.empty(0, np.int64), np.empty(0, np.float32))
            self._twin_cache[anchor] = out
            return out
        sims = self.X[pool] @ self.X[anchor]
        k = min(TOP_TWINS + 1, sims.size)
        top = pool[np.argpartition(-sims, k - 1)[:k]]
        top = top[top != anchor][:TOP_TWINS]
        tsims = (self.X[top] @ self.X[anchor]).astype(np.float32)
        out = (top, tsims)
        self._twin_cache[anchor] = out
        return out

    def _scores(self, anchor: int, candidates: np.ndarray) -> Dict[str, np.ndarray]:
        out = super()._scores(anchor, candidates)
        twins, tsims = self._twins(anchor)
        tie = TIEBREAK * (self.compat[candidates] @ self.compat[anchor])
        b1 = np.zeros(len(candidates), dtype=np.float32)
        b5 = np.zeros(len(candidates), dtype=np.float32)
        for rank_t, (t, s) in enumerate(zip(twins.tolist(), tsims.tolist())):
            adj = self.train_adj.get(t)
            if not adj:
                continue
            for ci, c in enumerate(candidates.tolist()):
                if c in adj:
                    if rank_t == 0:
                        b1[ci] = 1.0
                    b5[ci] += s
        out["borrow1c"] = b1 + tie
        out["borrow5c"] = b5 + tie
        return out


def main() -> None:
    compat = np.load(os.path.join(DATA_DIR, "compat_emb.npy"))
    for mode in ("hard_random", "pop_matched"):
        ev = BorrowEvaluator(extra_methods={"compat": compat}, neg_mode=mode)
        ev.method_names = ev.method_names + ["borrow1c", "borrow5c"]
        report = ev.evaluate()
        with open(os.path.join(DATA_DIR, f"eval_borrow_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump({"neg_mode": mode, "report": report}, f, indent=2)
        print(f"\n===== neg_mode={mode} =====", flush=True)
        for group in ("warm", "cold"):
            print(f"[{group}]")
            for m in ("compat", "borrow1c", "borrow5c", "cobuy", "siglip"):
                v = report[group][m]
                print(f"  {m:9s} AUC={v['auc']:.4f} MRR={v['mrr']:.4f} H@1={v['hit@1']:.3f} H@10={v['hit@10']:.3f} (n={v['n']})")


if __name__ == "__main__":
    main()
