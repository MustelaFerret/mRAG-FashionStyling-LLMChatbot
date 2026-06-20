from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SessionState:
    # the sticky anchor the USER chose (typed #id or clicked a card). It persists across turns
    # and is changed ONLY by an explicit user action; returned results never overwrite it.
    user_anchor_id: str = ""
    recent_item_ids: List[str] = field(default_factory=list)  # last results (cache; not the anchor)
    last_intent: str = ""  # last retrieval intent, so a terse refinement can continue it
    last_product_type: str = ""  # last target product_type, inherited by a terse refinement
    history: List[Dict[str, str]] = field(default_factory=list)
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
        # an empty session_id gets its own fresh session: a shared "anon" bucket would
        # leak anchor/history state between concurrent anonymous users
        sid = (session_id or "").strip() or f"anon-{uuid.uuid4().hex[:12]}"
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
        state.user_anchor_id = ""
        state.recent_item_ids = []
        state.last_intent = ""
        state.last_product_type = ""
        state.history = []
        state.updated_at = time.time()

    def set_user_anchor(self, state: SessionState, anchor_id: str) -> None:
        """Set the sticky anchor from an explicit user action (typed id / picked card).
        Persists until the user changes or clears it; results never overwrite it."""
        anchor = (anchor_id or "").strip()
        if not anchor:
            return
        state.user_anchor_id = anchor
        state.updated_at = time.time()

    def clear_user_anchor(self, state: SessionState) -> None:
        state.user_anchor_id = ""
        state.updated_at = time.time()

    def touch_results(self, state: SessionState, item_ids: List[str]) -> None:
        """Cache the latest results for reference only. Deliberately does NOT touch the anchor:
        the anchor stays whatever the user last chose, so a follow-up refinement keeps it."""
        state.recent_item_ids = item_ids[:12]
        state.updated_at = time.time()

    def add_message(self, state: SessionState, role: str, text: str, max_items: int) -> None:
        role_value = (role or "").strip().lower()
        text_value = (text or "").strip()
        if not role_value or not text_value:
            return
        state.history.append({"role": role_value, "text": text_value})
        if max_items > 0 and len(state.history) > max_items:
            state.history = state.history[-max_items:]
        state.updated_at = time.time()

    def get_history(self, state: SessionState, max_items: int) -> List[Dict[str, str]]:
        if max_items <= 0:
            return []
        return list(state.history[-max_items:])
