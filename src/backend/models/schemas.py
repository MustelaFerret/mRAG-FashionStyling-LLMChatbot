from pydantic import BaseModel


class ChatRequest(BaseModel):
    text: str = ""
    image: str | None = None
    session_id: str | None = None
    selected_anchor_id: str | None = None
    new_image_context: bool = False
    embedding_text_weight: float | None = None
    embedding_image_weight: float | None = None
    response_mode: str = "rich"
    include_debug: bool = True
    max_ui_items: int | None = None


class SessionResetRequest(BaseModel):
    session_id: str
