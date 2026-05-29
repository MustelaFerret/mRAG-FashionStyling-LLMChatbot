from __future__ import annotations

import gc
import logging
import math
import os
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


@dataclass
class Item2VecConfig:
    basket_cache_pickle: str
    meta_file: str
    output_dir: str
    embed_dim: int = 128
    num_negatives: int = 5
    hard_negative_ratio: float = 0.6
    min_item_count: int = 5
    epochs: int = 5
    batch_size: int = 8192
    lr: float = 5e-3
    weight_decay: float = 0.0
    seed: int = 42
    device: str = ""
    knn_k: int = 20
    knn_chunk: int = 1024
    log_every_batches: int = 500

    def resolve_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Item2VecModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, embed_dim)
        self.out_emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.in_emb.weight, std=1.0 / math.sqrt(embed_dim))
        nn.init.zeros_(self.out_emb.weight)

    def positive_score(self, anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        a = self.in_emb(anchors)
        p = self.out_emb(positives)
        return (a * p).sum(dim=-1)

    def negative_scores(self, anchors: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        a = self.in_emb(anchors).unsqueeze(1)
        n = self.out_emb(negatives)
        return (a * n).sum(dim=-1)

    def normalized_input_embedding(self) -> torch.Tensor:
        with torch.no_grad():
            return F.normalize(self.in_emb.weight.detach(), p=2, dim=-1)


def sgns_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    pos_loss = -F.logsigmoid(pos_scores).mean()
    neg_loss = -F.logsigmoid(-neg_scores).mean()
    return pos_loss + neg_loss


class Item2VecTrainer:
    def __init__(self, config: Item2VecConfig):
        self.config = config
        self.device = config.resolve_device()
        self.vocab: Dict[str, int] = {}
        self.id_to_item: List[str] = []
        self.item_counts: np.ndarray | None = None
        self.pair_anchors: np.ndarray | None = None
        self.pair_positives: np.ndarray | None = None
        self.pt_for_item: np.ndarray | None = None
        self.pt_offsets: np.ndarray | None = None
        self.pt_sizes: np.ndarray | None = None
        self.items_by_pt: np.ndarray | None = None
        self.unigram_dist: torch.Tensor | None = None
        self.model: Item2VecModel | None = None
        self._optimizer: torch.optim.Optimizer | None = None

    def prepare(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        with open(self.config.basket_cache_pickle, "rb") as handle:
            payload = pickle.load(handle)
        baskets: List[List[str]] = payload["train_baskets"]
        log.info("loaded %d train baskets from cache", len(baskets))

        item_counter: Counter[str] = Counter()
        for basket in baskets:
            for item in basket:
                item_counter[item] += 1
        kept_items = sorted(item for item, count in item_counter.items() if count >= self.config.min_item_count)
        self.vocab = {item: idx for idx, item in enumerate(kept_items)}
        self.id_to_item = kept_items
        vocab_size = len(self.vocab)
        log.info("vocab size: %d (kept items with count >= %d)", vocab_size, self.config.min_item_count)

        self.item_counts = np.array([item_counter[item] for item in self.id_to_item], dtype=np.int64)

        self._build_pairs(baskets)
        del baskets
        gc.collect()

        self._build_product_type_index()
        self._build_unigram_distribution()

        self.model = Item2VecModel(vocab_size, self.config.embed_dim).to(self.device)
        log.info("model on device: %s", self.device)

    def _build_pairs(self, baskets: List[List[str]]) -> None:
        anchor_chunks: List[np.ndarray] = []
        positive_chunks: List[np.ndarray] = []
        for basket in baskets:
            mapped = np.asarray([self.vocab[item] for item in basket if item in self.vocab], dtype=np.int32)
            n = mapped.size
            if n < 2:
                continue
            grid_a, grid_p = np.meshgrid(mapped, mapped, indexing="ij")
            mask = ~np.eye(n, dtype=bool)
            anchor_chunks.append(grid_a[mask])
            positive_chunks.append(grid_p[mask])
        self.pair_anchors = np.concatenate(anchor_chunks).astype(np.int32, copy=False)
        self.pair_positives = np.concatenate(positive_chunks).astype(np.int32, copy=False)
        log.info("materialized %d training pairs", self.pair_anchors.size)

    def _build_product_type_index(self) -> None:
        meta_df = pd.read_csv(self.config.meta_file, usecols=["article_id", "product_type_name"], dtype=str)
        meta_df["article_id"] = meta_df["article_id"].str.zfill(10)
        meta_df["product_type_name"] = meta_df["product_type_name"].fillna("")
        pt_map = dict(zip(meta_df["article_id"], meta_df["product_type_name"]))

        pt_to_idx: Dict[str, int] = {}
        vocab_size = len(self.vocab)
        pt_for_item = np.full(vocab_size, -1, dtype=np.int32)
        for item, idx in self.vocab.items():
            pt = pt_map.get(item, "")
            if not pt:
                continue
            if pt not in pt_to_idx:
                pt_to_idx[pt] = len(pt_to_idx)
            pt_for_item[idx] = pt_to_idx[pt]
        self.pt_for_item = pt_for_item

        items_by_pt: List[List[int]] = [[] for _ in range(len(pt_to_idx))]
        for idx, pt_idx in enumerate(pt_for_item.tolist()):
            if pt_idx >= 0:
                items_by_pt[pt_idx].append(idx)
        flat_items: List[int] = []
        pt_offsets = np.zeros(len(items_by_pt), dtype=np.int64)
        pt_sizes = np.zeros(len(items_by_pt), dtype=np.int64)
        for pt_idx, items in enumerate(items_by_pt):
            pt_offsets[pt_idx] = len(flat_items)
            pt_sizes[pt_idx] = len(items)
            flat_items.extend(items)
        self.items_by_pt = np.asarray(flat_items, dtype=np.int32)
        self.pt_offsets = pt_offsets
        self.pt_sizes = pt_sizes
        log.info("indexed %d product_types for hard negative sampling", len(pt_to_idx))

    def _build_unigram_distribution(self) -> None:
        counts = self.item_counts.astype(np.float64) ** 0.75
        probs = counts / counts.sum()
        self.unigram_dist = torch.from_numpy(probs).float()

    def _sample_negatives(self, anchor_ids: np.ndarray) -> np.ndarray:
        B = anchor_ids.size
        N = self.config.num_negatives
        n_hard = int(round(N * self.config.hard_negative_ratio))
        n_soft = N - n_hard
        vocab_size = len(self.id_to_item)

        out = np.zeros((B, N), dtype=np.int32)
        if n_hard > 0:
            anchor_pts = self.pt_for_item[anchor_ids]
            no_pt_mask = (anchor_pts < 0) | (self.pt_sizes[np.where(anchor_pts >= 0, anchor_pts, 0)] == 0)
            safe_pts = np.where(anchor_pts >= 0, anchor_pts, 0)
            sizes = self.pt_sizes[safe_pts]
            for k in range(n_hard):
                rand_pos = np.random.randint(0, np.maximum(sizes, 1))
                flat_idx = self.pt_offsets[safe_pts] + rand_pos
                hard_samples = self.items_by_pt[flat_idx]
                fallback = np.random.randint(0, vocab_size, size=B)
                out[:, k] = np.where(no_pt_mask, fallback, hard_samples)
        if n_soft > 0:
            soft = torch.multinomial(self.unigram_dist, B * n_soft, replacement=True).numpy().reshape(B, n_soft)
            out[:, n_hard : n_hard + n_soft] = soft.astype(np.int32)
        return out

    def train_epoch(self, epoch: int) -> float:
        rng = np.random.default_rng(self.config.seed + epoch)
        order = rng.permutation(self.pair_anchors.size)
        bs = self.config.batch_size
        total_loss = 0.0
        n_batches = 0
        self.model.train()

        for start in range(0, order.size, bs):
            idx = order[start : start + bs]
            anchor_np = self.pair_anchors[idx]
            positive_np = self.pair_positives[idx]
            negative_np = self._sample_negatives(anchor_np)

            anchors = torch.from_numpy(anchor_np).long().to(self.device, non_blocking=True)
            positives = torch.from_numpy(positive_np).long().to(self.device, non_blocking=True)
            negatives = torch.from_numpy(negative_np).long().to(self.device, non_blocking=True)

            pos_scores = self.model.positive_score(anchors, positives)
            neg_scores = self.model.negative_scores(anchors, negatives)
            loss = sgns_loss(pos_scores, neg_scores)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            if n_batches % self.config.log_every_batches == 0:
                log.info("epoch %d batch %d running_loss=%.4f", epoch, n_batches, total_loss / n_batches)

        return total_loss / max(1, n_batches)

    def train(self) -> None:
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        for epoch in range(self.config.epochs):
            start = time.time()
            avg_loss = self.train_epoch(epoch)
            log.info("epoch %d done in %.1fs avg_loss=%.4f", epoch, time.time() - start, avg_loss)

    def export_embeddings(self) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model.normalized_input_embedding().cpu().numpy()

    def build_knn_graph(self) -> List[Tuple[str, str, float]]:
        emb = self.model.normalized_input_embedding()
        vocab_size = emb.size(0)
        K = self.config.knn_k
        chunk = self.config.knn_chunk
        edges: List[Tuple[str, str, float]] = []
        for start in range(0, vocab_size, chunk):
            end = min(start + chunk, vocab_size)
            chunk_emb = emb[start:end]
            sims = chunk_emb @ emb.t()
            for local_idx in range(end - start):
                sims[local_idx, start + local_idx] = -2.0
            top_sims, top_idx = sims.topk(K, dim=-1)
            top_sims_cpu = top_sims.cpu().numpy()
            top_idx_cpu = top_idx.cpu().numpy()
            for local_idx in range(end - start):
                anchor_id = self.id_to_item[start + local_idx]
                for rank in range(K):
                    nbr_idx = int(top_idx_cpu[local_idx, rank])
                    sim = float(top_sims_cpu[local_idx, rank])
                    edges.append((anchor_id, self.id_to_item[nbr_idx], sim))
        return edges

    def write_knn_graph_csv(self, path: str) -> None:
        edges = self.build_knn_graph()
        df = pd.DataFrame(edges, columns=["item_a", "item_b", "weight"])
        df["weight"] = (df["weight"].clip(lower=0.0) * 1000.0).astype(int)
        df = df.sort_values(by=["item_a", "weight"], ascending=[True, False])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log.info("wrote knn graph: %s rows=%d", path, len(df))

    def save_embeddings(self, path: str) -> None:
        emb = self.export_embeddings()
        np.savez_compressed(path, embeddings=emb, items=np.array(self.id_to_item))
        log.info("saved embeddings: %s shape=%s", path, emb.shape)


def _config_from_env() -> Item2VecConfig:
    base = Path(__file__).resolve().parent.parent.parent
    return Item2VecConfig(
        basket_cache_pickle=str(base / os.getenv(
            "BASKET_CACHE", "data/processed/eval_cache/eval_baskets_98e4905692c809f6.pkl"
        )),
        meta_file=str(base / os.getenv("META_FILE", "data/processed/dataset_final_qwen_filled.csv")),
        output_dir=str(base / os.getenv("ITEM2VEC_OUTPUT_DIR", "data/processed/item2vec")),
        embed_dim=int(os.getenv("EMBED_DIM", "128")),
        num_negatives=int(os.getenv("NUM_NEGS", "5")),
        hard_negative_ratio=float(os.getenv("HARD_NEG_RATIO", "0.6")),
        min_item_count=int(os.getenv("MIN_ITEM_COUNT", "5")),
        epochs=int(os.getenv("EPOCHS", "5")),
        batch_size=int(os.getenv("BATCH_SIZE", "8192")),
        lr=float(os.getenv("LR", "5e-3")),
        seed=int(os.getenv("SEED", "42")),
        knn_k=int(os.getenv("KNN_K", "20")),
        knn_chunk=int(os.getenv("KNN_CHUNK", "1024")),
        log_every_batches=int(os.getenv("LOG_EVERY", "500")),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    config = _config_from_env()
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    trainer = Item2VecTrainer(config)
    trainer.prepare()
    trainer.train()

    tag = f"d{config.embed_dim}_n{config.num_negatives}_h{int(config.hard_negative_ratio * 100)}_e{config.epochs}"
    knn_path = os.path.join(config.output_dir, f"item2vec_{tag}_k{config.knn_k}.csv")
    trainer.write_knn_graph_csv(knn_path)

    emb_path = os.path.join(config.output_dir, f"item2vec_{tag}_embeddings.npz")
    trainer.save_embeddings(emb_path)


if __name__ == "__main__":
    main()
