"""Fixtures for end-to-end pipeline tests. These load the real models (SigLIP, Qwen,
DeBERTa) + Qdrant, so they are slow and require a GPU and the built artifacts. They are
marked `integration` and skipped by default; run with: python -m pytest -m integration

Requires exclusive access to the embedded Qdrant DB (stop the backend first).
"""
import os
import time

# DeBERTa hits an nvrtc JIT error on some GPUs; disable the fuser before torch loads.
os.environ.setdefault("PYTORCH_JIT", "0")

import pytest

from src.backend.core.config import settings


@pytest.fixture(scope="session")
def rag_full():
    try:
        from src.backend.retrieval.qdrant import QdrantStore
        from src.backend.retrieval.encoders import QueryEncoder, SigLIPEncoder
        from src.backend.retrieval.embeddings import SparseTfidfEncoder
        from src.backend.retrieval.llm import QwenMultimodalService
        from src.backend.services.catalog import FashionCatalog
        from src.backend.services.session_manager import SessionStore
        from src.backend.services.rag_service import FashionRAGService

        catalog = FashionCatalog(settings.meta_file, settings.graph_file, settings.image_dir)
        store = QdrantStore(settings.db_path, settings.collection_name)
        sparse = SparseTfidfEncoder.load(settings.sparse_model_path) if os.path.exists(settings.sparse_model_path) else None
        embedder = QueryEncoder(siglip=SigLIPEncoder(), sparse=sparse)
        llm = QwenMultimodalService()
        sessions = SessionStore(3600, 100)
        return FashionRAGService(embedder, store, llm, catalog, limit=5, personalization=None, sessions=sessions)
    except Exception as exc:  # no GPU / missing artifacts / Qdrant locked
        pytest.skip(f"pipeline unavailable: {exc}")


@pytest.fixture
def run(rag_full):
    def _run(query, anchor="", session="itest", confirmed_intent=None):
        items, _prompt, log, has_ctx, direct = rag_full._prepare_chat(
            query, None, session, "req", time.perf_counter(),
            confirmed_intent=confirmed_intent, customer_id=None, selected_anchor_id=anchor,
        )
        return {"items": items, "log": log, "has_ctx": has_ctx, "direct": direct,
                "types": [it["product_type"] for it in items],
                "colours": [it["colour_group"] for it in items],
                "ids": [it["article_id"] for it in items]}
    return _run
