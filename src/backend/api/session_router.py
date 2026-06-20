from fastapi import APIRouter, Depends, Request

from src.backend.core.exceptions import BackendNotReadyException, SessionIdRequiredException
from src.backend.models.schemas import SessionResetRequest
from src.backend.services.session_manager import SessionStore


session_router = APIRouter(prefix="/api/session", tags=["session"])


def get_session_store(request: Request) -> SessionStore:
    store = getattr(request.app.state, "sessions", None)
    if store is None:
        raise BackendNotReadyException()
    return store


@session_router.get("/{session_id}")
async def session_snapshot(session_id: str, sessions: SessionStore = Depends(get_session_store)):
    sid = (session_id or "").strip()
    if not sid:
        raise SessionIdRequiredException()

    state = sessions.get(sid)
    if state is None:
        return {
            "session_id": sid,
            "exists": False,
            "anchor_id": "",
            "recent_item_ids": [],
            "updated_at": None,
        }

    return {
        "session_id": sid,
        "exists": True,
        "anchor_id": state.anchor_id,
        "recent_item_ids": state.recent_item_ids,
        "updated_at": state.updated_at,
    }


@session_router.post("/reset")
async def reset_session(req: SessionResetRequest, sessions: SessionStore = Depends(get_session_store)):
    sid = (req.session_id or "").strip()
    if not sid:
        raise SessionIdRequiredException()

    existed = sessions.reset_by_session_id(sid)
    # do NOT clear the shared query log on a per-session reset -- that wiped global state
    # (other sessions / a running analysis) for one user's reset. The log is cleared once at
    # startup (main.py); nothing reads it live.
    return {
        "ok": True,
        "session_id": sid,
        "existed": existed,
    }

