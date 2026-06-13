"""Build rich compat node features = L2norm(SigLIP image_emb) ⊕ L2norm(SigLIP text_emb of
refined_description), in the compat node order. Compatibility is semantic (material/occasion/
style) as much as visual, so adding the text channel gives the metric head more to project.
Output: data/processed/compat/node_features_rich.npy (N, 1536)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.backend.core.config import settings
from src.backend.retrieval.encoders import SigLIPEncoder

DATA = settings.compat_dir
OUT = os.path.join(DATA, "node_features_rich.npy")


def main() -> None:
    ids = json.load(open(os.path.join(DATA, "node_ids.json"), encoding="utf-8"))["article_ids"]
    img = np.load(os.path.join(DATA, "node_features.npy")).astype(np.float32)
    assert img.shape[0] == len(ids)

    meta = pd.read_csv(settings.meta_file, usecols=["article_id", "refined_description"])
    meta["article_id"] = meta["article_id"].astype(str).str.zfill(10)
    desc = meta.set_index("article_id")["refined_description"].astype(str).to_dict()
    texts = [desc.get(a, "") or a for a in ids]

    enc = SigLIPEncoder()
    txt = enc.encode_texts(texts, show_progress=True).astype(np.float32)

    def l2(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n

    rich = np.concatenate([l2(img), l2(txt)], axis=1).astype(np.float32)
    np.save(OUT, rich)
    print(f"rich features {rich.shape} -> {OUT}")


if __name__ == "__main__":
    main()
