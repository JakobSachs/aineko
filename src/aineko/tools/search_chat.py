"""Tool to search past chat messages."""

import logging

from sqlalchemy import select

from aineko.db import get_session
from aineko.models.message import Message, Role
from aineko.tools.registry import ToolDef

logger = logging.getLogger(__name__)


async def _search_chat(
    keyword: str,
    role: str = "",
    limit: int = 20,
) -> str:
    """Search past chat messages by keyword, optionally filtered by role."""
    limit = min(limit, 50)

    async for db in get_session():
        query = (
            select(Message)
            .where(Message.content.ilike(f"%{keyword}%"))
            .order_by(Message.created_at.desc())
        )

        if role:
            try:
                role_enum = Role(role.lower())
            except ValueError:
                return (
                    f"Invalid role '{role}'. Valid roles: user, assistant, system, tool"
                )
            query = query.where(Message.role == role_enum)

        # Exclude tool-result messages (noisy internal plumbing)
        query = query.where(Message.role != Role.TOOL)

        query = query.limit(limit)
        result = await db.execute(query)
        messages = result.scalars().all()

    if not messages:
        return "No messages found matching your query."

    lines: list[str] = []
    for msg in reversed(messages):  # chronological order
        preview = msg.content[:500]
        lines.append(f"[{msg.created_at}] {msg.role.value}: {preview}")

    return "\n\n".join(lines)


search_chat_tool = ToolDef(
    name="search_chat",
    description=(
        "Search past chat messages by keyword. Finds messages from conversation "
        "history that match a search term. Useful for recalling past conversations, "
        "finding when something was discussed, or looking up information shared earlier."
    ),
    parameters={
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "Search term to match in message content.",
            },
            "role": {
                "type": "string",
                "description": "Filter by role: 'user' or 'assistant'. Leave empty for both.",
                "enum": ["", "user", "assistant"],
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 20, max 50).",
            },
        },
        "required": ["keyword"],
    },
    handler=_search_chat,
)
