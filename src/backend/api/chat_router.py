import json
import time
import traceback
import uuid

from pathlib import Path
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse

from src.backend.core.config import Settings, settings
from src.backend.core.query_logger import append_log
from src.backend.core.exceptions import (
    BackendNotReadyException,
    InferenceFailedException,
    StaticAssetNotFoundException,
)
from src.backend.models.schemas import ChatRequest
from src.backend.retrieval.llm import INTENT_CHAT, INTENT_COMPOSITE, INTENT_GRAPH
from src.backend.services.rag_service import FashionRAGService, NO_PAIRING_MESSAGE, NO_RESULTS_MESSAGE
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

def _format_sse(event: str, data: dict) -> str:
    payload = json.dumps(data or {}, ensure_ascii=True)
    return f"event: {event}\ndata: {payload}\n\n"

@chat_api_router.post("")
async def chat(req: ChatRequest, request: Request, rag: FashionRAGService = Depends(get_rag)):
    request_id = uuid.uuid4().hex[:12]
    session_id = (req.session_id or "").strip()
    customer_id = (req.customer_id or "").strip()
    selected_anchor_id = (req.selected_anchor_id or "").strip()
    confirmed_intent = (req.confirmed_intent or "").strip()
    wants_stream = bool(getattr(req, "stream", False)) or "text/event-stream" in request.headers.get("accept", "")
    try:
        image = FashionAssistantService.decode_image(req.image)
        if not wants_stream:
            message, items, extra = await rag.chat(
                req.text or "",
                image=image,
                session_id=session_id,
                request_id=request_id,
                confirmed_intent=confirmed_intent,
                customer_id=customer_id,
                selected_anchor_id=selected_anchor_id,
            )
            payload = {"message": message, "items": items}
            payload.update(extra or {})
            return {
                "status": "success",
                "data": {
                    **payload,
                },
            }

        started_at = time.perf_counter()
        items, full_prompt, log_payload, has_context, direct_response = rag.prepare_chat(
            query=req.text or "",
            image=image,
            session_id=session_id,
            request_id=request_id,
            started_at=started_at,
            confirmed_intent=confirmed_intent,
            customer_id=customer_id,
            selected_anchor_id=selected_anchor_id,
        )
        intent_hint = log_payload.get("intent_hint", "")

        def event_generator():
            if direct_response is not None:
                response_type = direct_response.get("type")
                if response_type == INTENT_COMPOSITE:
                    message = direct_response.get("message", "")
                    payload = {
                        "request_id": request_id,
                        "items": [],
                        "intent": INTENT_COMPOSITE,
                        "intent_options": direct_response.get("intent_options", []),
                        "intent_query": direct_response.get("intent_query", req.text or ""),
                    }
                    yield _format_sse("meta", payload)
                    rag.finalize_log(log_payload, started_at, 0)
                    yield _format_sse("done", {"message": message, **payload})
                    return
                if response_type == INTENT_CHAT:
                    yield _format_sse(
                        "meta",
                        {"request_id": request_id, "items": [], "intent": INTENT_CHAT},
                    )
                    message_parts: list[str] = []
                    try:
                        for token in rag.llm.generate_chitchat_stream(req.text or ""):
                            if not token:
                                continue
                            message_parts.append(token)
                            yield _format_sse("delta", {"delta": token})
                        full_message = "".join(message_parts).strip()
                        rag.finalize_log(log_payload, started_at, 0)
                        yield _format_sse("done", {"message": full_message, "intent": INTENT_CHAT})
                    except Exception as ex:
                        rag.finalize_log(log_payload, started_at, 0)
                        yield _format_sse("error", {"error": str(ex)})
                    return

            yield _format_sse(
                "meta",
                {"request_id": request_id, "items": items, "intent": intent_hint},
            )
            if not has_context:
                message = NO_PAIRING_MESSAGE if intent_hint == INTENT_GRAPH else NO_RESULTS_MESSAGE
                rag.finalize_log(log_payload, started_at, 0)
                yield _format_sse("delta", {"delta": message})
                yield _format_sse("done", {"message": message, "intent": intent_hint})
                return

            message_parts: list[str] = []
            try:
                for token in rag.stream_answer(full_prompt, image=image):
                    if not token:
                        continue
                    message_parts.append(token)
                    yield _format_sse("delta", {"delta": token})
                full_message = "".join(message_parts).strip()
                rag.finalize_log(log_payload, started_at, len(items))
                yield _format_sse("done", {"message": full_message, "intent": intent_hint})
            except Exception as ex:
                rag.finalize_log(log_payload, started_at, len(items))
                yield _format_sse("error", {"error": str(ex)})

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
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
                "stream": wants_stream,
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