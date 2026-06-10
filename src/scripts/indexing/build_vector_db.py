"""Build vector DB cho mRAG fashion với SOTA multimodal hybrid retrieval.

Schema (Qdrant collection `fashion_products`):
- text_emb (768d, COSINE)   : SigLIP-text(prod_name + refined_description + structured metadata)
- image_emb (768d, COSINE)  : SigLIP-image(product photo)
- sparse_bm25               : TF-IDF trên rich text (prod_name x2 + refined_description + structured metadata)

Process:
1. Load + clean dataset.
2. Build text strings (dense_text, sparse_text) per row.
3. Fit TF-IDF trên rich sparse texts → vocab ~16k+ (so với 359 cũ).
4. Batched encode text qua SigLIP (batch=64, FP16) → np.array (N, 768).
5. Batched encode image qua SigLIP (batch=32, FP16) → np.array (N, 768).
6. Encode sparse vectors qua SparseTfidfEncoder.
7. Upsert vào Qdrant với HNSW tuned (M=32, ef_construct=256).

Run:
    conda activate mRAG
    python -m src.scripts.build_vector_db
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import List

import pandas as pd
import torch
from qdrant_client.http.models import PointStruct, SparseVector
from tqdm.auto import tqdm

from src.backend.core.config import settings
from src.backend.core.utils import normalize_text
from src.backend.retrieval.embeddings import SparseTfidfEncoder
from src.backend.retrieval.encoders import SigLIPEncoder
from src.backend.retrieval.qdrant import QdrantStore


UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "256"))
RESET_DB = os.getenv("RESET_DB", "1") == "1"


class VectorIndexBuilder:
    INVALID_VALUES = {
        "",
        "unknown",
        "nan",
        "none",
        "null",
        "unspecified",
        "unspecified/other",
        "other",
    }

    STRUCTURED_FIELDS = [
        "product_type_name",
        "product_group_name",
        "garment_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "fit",
        "occasion",
        "seasonality",
        "style_aesthetic",
        "dominant_material",
        "design_detail",
    ]

    PAYLOAD_INDEX_FIELDS = [
        "product_type",
        "colour_group",
        "fit",
        "occasion",
        "seasonality",
        "department",
        "section_name",
        "index_name",
        "product_group",
        "garment_group",
        "graphical_appearance",
        "dominant_material",
        "colour_master",
    ]

    def __init__(self) -> None:
        self.df: pd.DataFrame | None = None
        self.sparse_encoder: SparseTfidfEncoder | None = None
        self.siglip: SigLIPEncoder | None = None
        self.store = QdrantStore(settings.db_path, settings.collection_name)

    def clean_value(self, value: object) -> str:
        if value is None:
            return ""
        raw = str(value).strip()
        if not raw:
            return ""
        norm = normalize_text(raw)
        if not norm or norm in self.INVALID_VALUES:
            return ""
        return raw

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(settings.meta_file, dtype={"article_id": str})
        df["article_id"] = df["article_id"].astype(str).str.zfill(10)
        df = df.fillna("")
        self.df = df.reset_index(drop=True)
        print(f"loaded meta: {len(self.df):,} rows")
        return self.df

    def build_dense_text(self, row: pd.Series) -> str:
        """Text dùng để encode SigLIP-text vector (text_emb).

        Format: '<prod_name>. <refined_description>. <key metadata phrases>'

        NOTE (audit_metadata bug A): SigLIP-base cắt ở 64 token và refined_description
        (~71 token median) đẩy cụm metadata ở đuôi ra ngoài cửa sổ ở ~71% sản phẩm. Đã
        THỬ front-load thuộc tính lên đầu để chống truncation -> đo gold-set **regress**
        (nDCG 0.761 -> 0.65-0.71): caption làm đồng nhất hoá embedding + chiếm chỗ của mô
        tả NL mà dense encoder dựa vào, trong khi tín hiệu cấu trúc đã được BM25 (sparse)
        bao trong RRF. Kết luận: để dense là mô tả thuần. Chi tiết: md/audit_metadata.md.
        """
        parts: List[str] = []
        name = self.clean_value(row.get("prod_name", ""))
        if name:
            parts.append(name)
        desc = self.clean_value(row.get("refined_description", ""))
        if not desc:
            desc = self.clean_value(row.get("detail_desc", ""))
        if desc:
            parts.append(desc)
        meta_phrase = self._structured_phrase(row, limit_fields=6)
        if meta_phrase:
            parts.append(meta_phrase)
        return ". ".join(parts) if parts else "fashion item"

    def _structured_phrase(self, row: pd.Series, limit_fields: int = 6) -> str:
        out: List[str] = []
        priority = [
            "product_type_name",
            "colour_group_name",
            "occasion",
            "style_aesthetic",
            "seasonality",
            "fit",
            "dominant_material",
            "section_name",
        ]
        for field in priority:
            value = self.clean_value(row.get(field, ""))
            if value:
                out.append(value)
            if len(out) >= limit_fields:
                break
        return " ".join(out)

    def build_sparse_text(self, row: pd.Series) -> str:
        """Rich text cho TF-IDF: prod_name (x2 boost) + refined_description + structured metadata.

        So với version cũ (chỉ structured metadata, 359 vocab) → version này có vocab ~16k+,
        cover natural language query như 'elegant flowy summer dress', 'chunky oversized knit'.
        """
        parts: List[str] = []
        name = self.clean_value(row.get("prod_name", ""))
        if name:
            parts.append(name)
            parts.append(name)  # boost x2
        product_type = self.clean_value(row.get("product_type_name", ""))
        if product_type:
            parts.append(product_type)
            parts.append(product_type)
        desc = self.clean_value(row.get("refined_description", ""))
        if not desc:
            desc = self.clean_value(row.get("detail_desc", ""))
        if desc:
            parts.append(desc)
        for field in self.STRUCTURED_FIELDS:
            if field == "product_type_name":
                continue
            value = self.clean_value(row.get(field, ""))
            if value:
                parts.append(value)
        return " ".join(parts)

    def build_payload(self, row: pd.Series, image_path: str) -> dict:
        payload = {
            "article_id": str(row.get("article_id", "")),
            "image_path": image_path,
            "prod_name": str(row.get("prod_name", "")),
        }
        mapping = {
            "product_type": "product_type_name",
            "product_group": "product_group_name",
            "garment_group": "garment_group_name",
            "colour_group": "colour_group_name",
            "index_name": "index_name",
            "section_name": "section_name",
            "department": "department_name",
            "fit": "fit",
            "occasion": "occasion",
            "seasonality": "seasonality",
            "style_aesthetic": "style_aesthetic",
            # gap C (audit_metadata): attributes that were usable but absent from payload,
            # so they could not be hard/soft filtered or graded. pattern + material are
            # indexed for filtering; shade/master broaden colour matching.
            "graphical_appearance": "graphical_appearance_name",
            "dominant_material": "dominant_material",
            "colour_value": "perceived_colour_value_name",
            "colour_master": "perceived_colour_master_name",
            "description": "refined_description",
        }
        for payload_key, row_key in mapping.items():
            payload[payload_key] = self.clean_value(row.get(row_key, ""))
        return payload

    def fit_sparse_encoder(self, df: pd.DataFrame) -> SparseTfidfEncoder:
        print("building sparse texts...")
        sparse_texts = [self.build_sparse_text(row) for _, row in tqdm(df.iterrows(), total=len(df))]
        encoder = SparseTfidfEncoder()
        encoder.fit(sparse_texts)
        out_path = Path(settings.sparse_model_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        encoder.save(str(out_path))
        print(f"sparse encoder: vocab={len(encoder.vocab):,} (vs 359 old) -> {out_path}")
        self.sparse_encoder = encoder
        return encoder

    def _image_paths(self, df: pd.DataFrame) -> List[str]:
        paths: List[str] = []
        for aid in df["article_id"]:
            folder = aid[:3]
            paths.append(os.path.join(settings.image_dir, folder, f"{aid}.jpg"))
        return paths

    def encode_all(self, df: pd.DataFrame):
        assert self.sparse_encoder is not None
        self.siglip = SigLIPEncoder()

        dense_texts = [self.build_dense_text(row) for _, row in df.iterrows()]
        sparse_texts = [self.build_sparse_text(row) for _, row in df.iterrows()]
        image_paths = self._image_paths(df)

        text_embs = self.siglip.encode_texts(
            dense_texts,
            batch_size=settings.encode_batch_text,
            progress_desc="text_emb",
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        image_embs = self.siglip.encode_images(
            image_paths,
            batch_size=settings.encode_batch_image,
            progress_desc="image_emb",
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.siglip.free()
        self.siglip = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return text_embs, image_embs, sparse_texts, image_paths

    def create_collection(self) -> None:
        self.store.ensure_collection(
            text_vector_name=settings.vector_name_text,
            image_vector_name=settings.vector_name_image,
            sparse_vector_name=settings.vector_name_sparse,
            embed_dim=768,
            reset=RESET_DB,
            hnsw_m=settings.hnsw_m,
            hnsw_ef_construct=settings.hnsw_ef_construct,
            payload_index_fields=self.PAYLOAD_INDEX_FIELDS,
        )

    def upsert_all(self, df: pd.DataFrame, text_embs, image_embs, sparse_texts, image_paths) -> None:
        assert self.sparse_encoder is not None
        n = len(df)
        print(f"upsert {n:,} points to '{settings.collection_name}' (batch={UPSERT_BATCH})")
        for start in tqdm(range(0, n, UPSERT_BATCH), desc="upsert"):
            stop = min(start + UPSERT_BATCH, n)
            points: List[PointStruct] = []
            for i in range(start, stop):
                row = df.iloc[i]
                aid = str(row["article_id"]).zfill(10)
                try:
                    point_id = int(aid)
                except ValueError:
                    continue
                sparse_idx, sparse_val = self.sparse_encoder.encode(sparse_texts[i])
                vector = {
                    settings.vector_name_text: text_embs[i].tolist(),
                    settings.vector_name_image: image_embs[i].tolist(),
                }
                if sparse_idx:
                    vector[settings.vector_name_sparse] = SparseVector(
                        indices=sparse_idx, values=sparse_val
                    )
                payload = self.build_payload(row, image_paths[i])
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            if points:
                self.store.client.upsert(collection_name=settings.collection_name, points=points)

    def run(self) -> None:
        df = self.load_data()
        self.fit_sparse_encoder(df)
        self.create_collection()
        text_embs, image_embs, sparse_texts, image_paths = self.encode_all(df)
        self.upsert_all(df, text_embs, image_embs, sparse_texts, image_paths)
        info = self.store.client.get_collection(settings.collection_name)
        print(f"\ndone. collection points={info.points_count}")


def main() -> None:
    VectorIndexBuilder().run()


if __name__ == "__main__":
    main()
