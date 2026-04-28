from __future__ import annotations

import gc
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.backend.api.chat_router import chat_router
from src.backend.api.session_router import session_router
from src.backend.core.config import settings, setup_environment
from src.backend.retrieval.embeddings import SiglipEmbeddingService
from src.backend.retrieval.llm import QwenMultimodalService
from src.backend.retrieval.qdrant import QdrantStore
from src.backend.services.catalog import FashionCatalog
from src.backend.services.recommender import FashionAssistantService
from src.backend.services.session_manager import SessionStore


setup_environment()


def configure_torch_runtime() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    else:
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    else:
        torch.backends.cudnn.allow_tf32 = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_torch_runtime()

    catalog = FashionCatalog(settings.meta_file, settings.graph_file, settings.image_dir)
    qdrant = QdrantStore(settings.db_path, settings.collection_name)
    sessions = SessionStore(settings.session_ttl_seconds, settings.max_session_context)
    embedding = SiglipEmbeddingService()
    llm = QwenMultimodalService()
    assistant = FashionAssistantService(settings, catalog, qdrant, embedding, llm, sessions)

    app.state.settings = settings
    app.state.catalog = catalog
    app.state.qdrant = qdrant
    app.state.sessions = sessions
    app.state.embedding = embedding
    app.state.llm = llm
    app.state.assistant = assistant

    try:
        yield
    finally:
        if hasattr(qdrant, 'client'):
            qdrant.client.close()
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)
app.include_router(session_router)

if os.path.exists(settings.frontend_dir) and os.path.isdir(settings.frontend_dir):
    app.mount("/frontend", StaticFiles(directory=settings.frontend_dir), name="frontend_assets")
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend_root")