import gc
import os

import pandas as pd
import torch
from PIL import Image
from qdrant_client.http.models import PointStruct, SparseVector
from tqdm.auto import tqdm

from src.backend.core.utils import normalize_text
from src.backend.retrieval.embeddings import HybridEmbeddingService, SparseTfidfEncoder
from src.backend.retrieval.qdrant import QdrantStore

DATA_FILE = os.getenv("DATA_FILE", "data/processed/dataset_qwen_completed.csv")
IMAGE_DIR = os.getenv("IMAGE_DIR", "data/raw/images")
DB_PATH = os.getenv("QDRANT_DB_PATH", "db/qdrant_local_db")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "fashion_products")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
START_INDEX = int(os.getenv("START_INDEX", "0"))
RESET_DB = os.getenv("RESET_DB", "1") == "1"
DENSE_VECTOR_NAME = os.getenv("DENSE_VECTOR_NAME", "dense")
SPARSE_VECTOR_NAME = os.getenv("SPARSE_VECTOR_NAME", "sparse")
SPARSE_MODEL_PATH = os.getenv("SPARSE_MODEL_PATH", "data/processed/sparse_tfidf.json")


class HybridVectorBuilder:
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.sparse_encoder: SparseTfidfEncoder | None = None
        self.embedder: HybridEmbeddingService | None = None
        self.store = QdrantStore(DB_PATH, COLLECTION_NAME)
        self.invalid_values = {
            "",
            "unknown",
            "nan",
            "none",
            "null",
            "unspecified",
            "unspecified/other",
            "other",
        }
        self.sparse_fields = [
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
        ]
        self.name_fields = ["prod_name"]
        self.payload_index_fields = [
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
        ]

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(DATA_FILE, dtype={"article_id": str})
        df["article_id"] = df["article_id"].astype(str).str.zfill(10)
        self.df = df
        return df

    def clean_value(self, value: object) -> str:
        if value is None:
            return ""
        raw = str(value).strip()
        if not raw:
            return ""
        norm = normalize_text(raw)
        if not norm or norm in self.invalid_values:
            return ""
        return raw

    def build_sparse_text(self, row: pd.Series) -> str:
        parts: list[str] = []
        product_type = self.clean_value(row.get("product_type_name", ""))
        if product_type:
            parts.append(product_type)
            parts.append(product_type)
        for field in self.sparse_fields:
            if field == "product_type_name":
                continue
            value = self.clean_value(row.get(field, ""))
            if value:
                parts.append(value)
        return " ".join(parts)

    def build_dense_text(self, row: pd.Series) -> str:
        name = ""
        for field in self.name_fields:
            name = self.clean_value(row.get(field, ""))
            if name:
                break
        desc = self.clean_value(row.get("refined_description", ""))
        if name and desc:
            return f"{name}. {desc}"
        if desc:
            return desc
        return name

    def build_payload(self, row: pd.Series, image_path: str) -> dict:
        payload = {
            "article_id": str(row.get("article_id", "")),
            "image_path": image_path,
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
            "description": "refined_description",
        }

        for payload_key, row_key in mapping.items():
            payload[payload_key] = self.clean_value(row.get(row_key, ""))

        return payload

    def load_image(self, path: str) -> Image.Image | None:
        if not path or not os.path.exists(path):
            return None
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception:
            return None

    def build_sparse_encoder(self, df: pd.DataFrame) -> SparseTfidfEncoder:
        sparse_texts = [self.build_sparse_text(row) for _, row in df.iterrows()]
        encoder = SparseTfidfEncoder()
        encoder.fit(sparse_texts)
        if SPARSE_MODEL_PATH:
            os.makedirs(os.path.dirname(SPARSE_MODEL_PATH), exist_ok=True)
            encoder.save(SPARSE_MODEL_PATH)
        self.sparse_encoder = encoder
        return encoder

    def prepare_collection(self) -> None:
        self.store.ensure_collection(
            dense_name=DENSE_VECTOR_NAME,
            sparse_name=SPARSE_VECTOR_NAME,
            size=768,
            reset=RESET_DB,
            payload_index_fields=self.payload_index_fields,
        )

    def run(self) -> None:
        df = self.load_data()
        self.build_sparse_encoder(df)
        self.prepare_collection()
        self.embedder = HybridEmbeddingService(sparse_encoder=self.sparse_encoder)
        total_items = len(df)
        print(f"build start {total_items}")

        for start_idx in tqdm(range(START_INDEX, total_items, BATCH_SIZE)):
            batch_df = df.iloc[start_idx : start_idx + BATCH_SIZE]
            batch_points: list[PointStruct] = []

            for _, row in batch_df.iterrows():
                article_id = str(row.get("article_id", "")).zfill(10)
                if not article_id:
                    continue
                folder = article_id[:3]
                image_path = os.path.join(IMAGE_DIR, folder, f"{article_id}.jpg")
                image = self.load_image(image_path)
                dense_text = self.build_dense_text(row)
                sparse_text = self.build_sparse_text(row)
                dense_vector, sparse_idx, sparse_val = self.embedder.encode_hybrid(
                    dense_text,
                    sparse_text,
                    image=image,
                    text_weight=0.5,
                    image_weight=0.5,
                )
                payload = self.build_payload(row, image_path)
                sparse_vector = SparseVector(indices=sparse_idx, values=sparse_val)
                point = PointStruct(
                    id=int(article_id),
                    vector={
                        DENSE_VECTOR_NAME: dense_vector,
                        SPARSE_VECTOR_NAME: sparse_vector,
                    },
                    payload=payload,
                )
                batch_points.append(point)
                if image is not None:
                    del image

            if batch_points:
                self.store.client.upsert(collection_name=COLLECTION_NAME, points=batch_points)

            del batch_points
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        info = self.store.client.get_collection(COLLECTION_NAME)
        print(f"build done {info.points_count}")


def main():
    builder = HybridVectorBuilder()
    builder.run()


if __name__ == "__main__":
    main()