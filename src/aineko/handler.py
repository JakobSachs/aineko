"""Decomposed message handler — extracted from app.py."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select

from aineko.models.message import Message, Role, Session, ToolLog
from aineko.tools.messaging import (
    make_background_task_tools,
    make_send_file_tool,
    make_send_message_tool,
)
from aineko.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from aineko.kimi.client import ChatResponse, ToolCallRecord
    from aineko.matrix.client import MatrixConnector
    from aineko.schemas.message import IncomingMessage
    from aineko.tools.background_task import BackgroundTaskManager

logger = logging.getLogger(__name__)

_COMMAND_WORDS = frozenset(("/reset", "/clear", "/forget"))

# Tools that produce user-visible output (matrix messages/files). Excluded from
# the tool-count footer since the user already sees them.
_VISIBLE_TOOLS = frozenset(("send_message", "send_file"))
_HIDDEN_TOOLS = frozenset(("memory",))

_MEMORY_WRITE_ACTIONS = frozenset(("store", "facts_add"))


def format_tool_footer(tool_history: list[ToolCallRecord]) -> str:
    """Return an italic footer with the count of non-visible tools used.

    Appended to final model responses so the user can see at a glance whether
    the model did background work (bash, web_search, file reads, etc.) beyond
    just talking. Also shows how many new memory entries were created.
    Returns an empty string if no background tools ran and no memories stored.
    """
    excluded = _VISIBLE_TOOLS | _HIDDEN_TOOLS
    tool_count = sum(1 for rec in tool_history if rec.tool_name not in excluded)

    memory_writes = 0
    for rec in tool_history:
        if rec.tool_name == "memory":
            try:
                args = json.loads(rec.arguments) if rec.arguments else {}
            except (json.JSONDecodeError, TypeError):
                continue
            if args.get("action") in _MEMORY_WRITE_ACTIONS:
                memory_writes += 1

    parts: list[str] = []
    if tool_count:
        parts.append(f"{tool_count} tool{'s' if tool_count != 1 else ''}")
    if memory_writes:
        parts.append(f"{memory_writes} memor{'ies' if memory_writes != 1 else 'y'}")
    if not parts:
        return ""
    return f"\n\n*— {', '.join(parts)}*"


async def handle_command(
    db: AsyncSession,
    msg: IncomingMessage,
    matrix: MatrixConnector,
) -> bool:
    """Handle slash commands. Returns True if the message was a command."""
    if msg.body.strip().lower() not in _COMMAND_WORDS:
        return False

    result = await db.execute(select(Session).where(Session.room_id == msg.room_id))
    session: Session | None = result.scalar_one_or_none()
    if session:
        await db.execute(delete(Message).where(Message.session_id == session.id))
        await db.commit()

    await matrix.send_message(msg.room_id, "conversation cleared, fresh start")
    return True


def build_request_tools(
    base_tools: ToolRegistry,
    matrix: MatrixConnector,
    room_id: str,
    sent_messages: list[str],
    bg_tasks: BackgroundTaskManager | None = None,
) -> ToolRegistry:
    """Clone base tools and add per-request messaging tools."""
    registry = ToolRegistry()
    for tool_def in base_tools._tools.values():
        registry.register(tool_def)
    registry.register(make_send_message_tool(matrix, room_id, sent_messages))
    registry.register(make_send_file_tool(matrix, room_id))

    if bg_tasks is not None:
        for tool_def in make_background_task_tools(bg_tasks, room_id, matrix):
            registry.register(tool_def)

    return registry


async def load_conversation(
    db: AsyncSession,
    msg: IncomingMessage,
    system_prompt: str,
) -> tuple[Session, Message, list[dict[str, Any]]]:
    """Load or create session, save user message, build messages list for kimi."""
    # Get or create session
    result = await db.execute(select(Session).where(Session.room_id == msg.room_id))
    session: Session | None = result.scalar_one_or_none()
    if session is None:
        session = Session(room_id=msg.room_id)
        db.add(session)
        await db.flush()

    # Save incoming message
    user_msg: Message = Message(session_id=session.id, role=Role.USER, content=msg.body)
    db.add(user_msg)
    await db.flush()

    # Load history
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session.id)
        .order_by(Message.created_at)
    )
    history: list[Message] = list(result.scalars().all())

    # Build messages for kimi
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    for m in history:
        if m.role == Role.ASSISTANT and m.tool_name == "__has_tool_calls__":
            try:
                blocks = json.loads(m.content)
                messages.append({"role": "assistant", "content": blocks})
            except json.JSONDecodeError:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": m.content}],
                    }
                )
        elif m.role == Role.TOOL and m.tool_name == "__tool_results__":
            try:
                blocks = json.loads(m.content)
                messages.append({"role": "user", "content": blocks})
            except json.JSONDecodeError:
                pass
        elif m.role == Role.ASSISTANT:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": m.content}]}
            )
        elif m.role == Role.SYSTEM:
            messages.append({"role": "system", "content": m.content})
        elif m.role == Role.USER:
            # Tag every user message with its timestamp so the model perceives time
            tag = f"\n[sent {m.created_at.strftime('%Y-%m-%d %H:%M UTC')}]"
            messages.append({"role": "user", "content": m.content + tag})
        else:
            messages.append({"role": m.role.value, "content": m.content})

    # Image attachment
    if msg.image_b64 and msg.image_mime:
        last_user: dict[str, Any] = messages[-1]
        last_user["content"] = [
            {"type": "text", "text": last_user["content"]},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": msg.image_mime,
                    "data": msg.image_b64,
                },
            },
        ]

    return session, user_msg, messages


async def persist_response(
    db: AsyncSession,
    session: Session,
    user_msg: Message,
    response: ChatResponse,
    sent_messages: list[str],
) -> None:
    """Save the assistant response and tool logs to DB."""
    if response.tool_history:
        # Assistant message with tool call blocks
        assistant_blocks: list[dict[str, Any]] = []
        for sm in sent_messages:
            assistant_blocks.append({"type": "text", "text": sm})
        for rec in response.tool_history:
            assistant_blocks.append(
                {
                    "type": "tool_use",
                    "id": f"persisted_{rec.tool_name}_{id(rec)}",
                    "name": rec.tool_name,
                    "input": json.loads(rec.arguments) if rec.arguments else {},
                }
            )
        if response.content:
            assistant_blocks.append({"type": "text", "text": response.content})

        db.add(
            Message(
                session_id=session.id,
                role=Role.ASSISTANT,
                content=json.dumps(assistant_blocks),
                token_count=response.usage.get("total_tokens"),
                tool_name="__has_tool_calls__",
            )
        )

        # Tool results
        result_blocks: list[dict[str, Any]] = []
        for rec in response.tool_history:
            result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": f"persisted_{rec.tool_name}_{id(rec)}",
                    "content": rec.result[:2000],
                }
            )
        db.add(
            Message(
                session_id=session.id,
                role=Role.TOOL,
                content=json.dumps(result_blocks),
                tool_name="__tool_results__",
            )
        )
    else:
        # Plain assistant message
        all_visible: list[str] = sent_messages.copy()
        if response.content:
            all_visible.append(response.content)
        visible: str = "\n".join(all_visible) if all_visible else ""
        if visible:
            db.add(
                Message(
                    session_id=session.id,
                    role=Role.ASSISTANT,
                    content=visible,
                    token_count=response.usage.get("total_tokens"),
                )
            )

    # Tool call logs
    for rec in response.tool_history:
        db.add(
            ToolLog(
                session_id=session.id,
                message_id=user_msg.id,
                tool_name=rec.tool_name,
                arguments=rec.arguments,
                result=rec.result,
            )
        )

    await db.commit()
