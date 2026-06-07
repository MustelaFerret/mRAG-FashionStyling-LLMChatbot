from pydantic import BaseModel


class ChatRequest(BaseModel):
    text: str = ""
    image: str | None = None
    session_id: str | None = None
    customer_id: str | None = None
    selected_anchor_id: str | None = None
    confirmed_intent: str | None = None
    stream: bool = False


class SessionResetRequest(BaseModel):
    session_id: str
