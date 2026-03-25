"""Message and session schemas."""

from datetime import datetime

from pydantic import BaseModel

from aineko.models.message import Role


class MessageSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    session_id: int
    role: Role
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    token_count: int | None = None
    created_at: datetime


class SessionSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    room_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageSchema] = []


class IncomingMessage(BaseModel):
    """Normalized inbound message from Matrix."""

    room_id: str
    sender: str
    body: str
    timestamp: datetime
    event_id: str
