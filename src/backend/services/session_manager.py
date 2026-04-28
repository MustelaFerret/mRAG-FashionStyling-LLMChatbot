from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SessionState:
    anchor_id: str = ""
    recent_item_ids: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_seconds: int, max_sessions: int):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: Dict[str, SessionState] = {}

    def _cleanup(self) -> None:
        now = time.time()
        expired = [sid for sid, st in self._sessions.items() if now - st.updated_at > self.ttl_seconds]
        for sid in expired:
            self._sessions.pop(sid, None)

        if len(self._sessions) <= self.max_sessions:
            return

        oldest = sorted(self._sessions.items(), key=lambda x: x[1].updated_at)
        over = len(self._sessions) - self.max_sessions
        for sid, _ in oldest[:over]:
            self._sessions.pop(sid, None)

    def get_or_create(self, session_id: str | None) -> Tuple[str, SessionState]:
        self._cleanup()
        sid = (session_id or "anon").strip() or "anon"
        state = self._sessions.get(sid)
        if state is None:
            state = SessionState()
            self._sessions[sid] = state
        state.updated_at = time.time()
        return sid, state

    def get(self, session_id: str | None) -> SessionState | None:
        self._cleanup()
        sid = (session_id or "").strip()
        if not sid:
            return None
        return self._sessions.get(sid)

    def reset_by_session_id(self, session_id: str | None) -> bool:
        state = self.get(session_id)
        if state is None:
            return False
        self.reset(state)
        return True

    def reset(self, state: SessionState) -> None:
        state.anchor_id = ""
        state.recent_item_ids = []
        state.updated_at = time.time()

    def touch_anchor(self, state: SessionState, anchor_id: str, item_ids: List[str]) -> None:
        state.anchor_id = anchor_id
        state.recent_item_ids = item_ids[:12]
        state.updated_at = time.time()
