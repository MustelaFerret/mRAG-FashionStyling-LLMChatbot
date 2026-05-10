import traceback
import uuid

from pathlib import Path
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.backend.core.config import Settings, settings
from src.backend.core.query_logger import append_log
from src.backend.core.exceptions import (
    BackendNotReadyException,
    InferenceFailedException,
    StaticAssetNotFoundException,
)
from src.backend.models.schemas import ChatRequest
from src.backend.services.rag_service import FashionRAGService
from src.backend.services.recommender import FashionAssistantService

chat_api_router = APIRouter(prefix="/api/chat", tags=["chat"])
chat_router = APIRouter()

def _resolve_safe_path(base_dir: Path, relative_path: str) -> Path | None:
    base = base_dir.resolve()
    target = (base / relative_path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return None
    return target

def _get_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", settings)

def get_assistant(request: Request) -> FashionAssistantService:
    assistant = getattr(request.app.state, "assistant", None)
    if assistant is None:
        raise BackendNotReadyException()
    return assistant

def get_rag(request: Request) -> FashionRAGService:
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise BackendNotReadyException()
    return rag

@chat_api_router.post("")
async def chat(req: ChatRequest, rag: FashionRAGService = Depends(get_rag)):
    request_id = uuid.uuid4().hex[:12]
    session_id = (req.session_id or "").strip()
    try:
        image = FashionAssistantService.decode_image(req.image)
        message, items = await rag.chat(req.text or "", image=image, session_id=session_id, request_id=request_id)
        return {
            "status": "success",
            "data": {
                "message": message,
                "items": items,
            },
        }
    except Exception as ex:
        append_log(
            settings.log_dir,
            {
                "event": "chat_error",
                "request_id": request_id,
                "session_id": session_id,
                "query": req.text or "",
                "has_image": bool(req.image),
                "error": str(ex),
            },
        )
        traceback.print_exc()
        raise InferenceFailedException(str(ex)) from ex

@chat_router.get("/api/frontend/bootstrap")
async def frontend_bootstrap(assistant: FashionAssistantService = Depends(get_assistant)):
    return assistant.frontend_bootstrap()

@chat_router.get("/images/{folder}/{filename}")
async def image_asset(folder: str, filename: str, request: Request):
    app_settings = _get_settings(request)
    image_file = _resolve_safe_path(Path(app_settings.image_dir), f"{folder}/{filename}")
    if image_file is None or not image_file.exists() or not image_file.is_file():
        raise StaticAssetNotFoundException()
    return FileResponse(path=str(image_file))

chat_router.include_router(chat_api_router)