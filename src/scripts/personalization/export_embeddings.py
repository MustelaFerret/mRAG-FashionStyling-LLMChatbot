"""Dump image_emb của mọi article từ Qdrant -> npy matrix + article_id index.

Dùng chung cho taste vector (offline) lẫn cosine re-rank (runtime).

    python -m src.scripts.personalization.export_embeddings
"""
from __future__ import annotations

import json

import numpy as np
from tqdm.auto import tqdm

from src.backend.core.config import settings
from src.backend.retrieval.qdrant import QdrantStore
from src.scripts.personalization import config as C


def main():
    store = QdrantStore(settings.db_path, settings.collection_name)
    total = store.client.get_collection(settings.collection_name).points_count

    ids, vecs = [], []
    offset = None
    with tqdm(total=total, desc="export") as bar:
        while True:
            pts, offset = store.client.scroll(
                collection_name=settings.collection_name,
                limit=2000,
                offset=offset,
                with_payload=["article_id"],
                with_vectors=[settings.vector_name_image],
            )
            if not pts:
                break
            for p in pts:
                v = p.vector.get(settings.vector_name_image) if isinstance(p.vector, dict) else None
                if v is None:
                    continue
                ids.append(str(p.payload.get("article_id", "")).zfill(10))
                vecs.append(v)
            bar.update(len(pts))
            if offset is None:
                break

    mat = np.asarray(vecs, dtype=np.float32)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    np.save(C.ARTICLE_EMB_NPY, mat.astype(np.float16))
    C.ARTICLE_INDEX.write_text(json.dumps({aid: i for i, aid in enumerate(ids)}))
    print(f"saved {mat.shape} -> {C.ARTICLE_EMB_NPY.name}, index {len(ids)} articles")


if __name__ == "__main__":
    main()
