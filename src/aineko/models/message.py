"""Conversation session and message models."""

import enum
import json as _json
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aineko.db import Base


class Role(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    room_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    last_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        order_by="Message.created_at",
        cascade="all, delete-orphan",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    role: Mapped[Role] = mapped_column(Enum(Role))
    content: Mapped[str] = mapped_column(Text)
    blocks: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped[Session] = relationship(back_populates="messages")


class ToolLog(Base):
    """Log of every tool call made during an agent loop."""

    __tablename__ = "tool_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), index=True)
    tool_name: Mapped[str] = mapped_column(String, index=True)
    arguments: Mapped[str] = mapped_column(Text)  # JSON
    result: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped[Session] = relationship()
    message: Mapped[Message] = relationship()

    @property
    def arguments_dict(self) -> dict:
        return _json.loads(self.arguments) if self.arguments else {}

    def summary(self, result_limit: int = 500) -> str:
        """One-line summary for search results."""
        result_preview = self.result[:result_limit] if self.result else ""
        return (
            f"[{self.created_at}] {self.tool_name}({self.arguments}) "
            f"-> {result_preview}"
        )
