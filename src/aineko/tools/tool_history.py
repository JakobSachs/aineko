"""Tool to search past tool call history."""

import logging

from sqlalchemy import select

from aineko.db import get_session
from aineko.models.message import Message, ToolLog
from aineko.tools.registry import ToolDef

logger = logging.getLogger(__name__)


async def _search_tool_history(
    tool_name: str = "",
    keyword: str = "",
    limit: int = 20,
) -> str:
    """Search tool call logs, optionally filtered by tool name and keyword."""
    limit = min(limit, 50)

    async for db in get_session():
        query = select(ToolLog).order_by(ToolLog.created_at.desc())

        if tool_name:
            query = query.where(ToolLog.tool_name == tool_name)

        if keyword:
            kw = f"%{keyword}%"
            query = query.where(ToolLog.arguments.ilike(kw) | ToolLog.result.ilike(kw))

        query = query.limit(limit)
        result = await db.execute(query)
        logs = result.scalars().all()

    if not logs:
        return "No tool call history found matching your query."

    # Include the triggering user message for context
    lines: list[str] = []
    msg_cache: dict[int, str] = {}

    async for db in get_session():
        for log in reversed(logs):  # chronological order
            if log.message_id not in msg_cache:
                msg_result = await db.execute(
                    select(Message.content).where(Message.id == log.message_id)
                )
                row = msg_result.scalar_one_or_none()
                msg_cache[log.message_id] = (row or "")[:200]

            trigger = msg_cache[log.message_id]
            lines.append(
                f'[{log.created_at}] triggered by: "{trigger}"\n'
                f"  {log.tool_name}({log.arguments})\n"
                f"  -> {log.result[:500]}"
            )

    return "\n\n".join(lines)


search_tool_history_tool = ToolDef(
    name="search_tool_history",
    description=(
        "Search your past tool call history. Find what tools you've used, "
        "what commands you ran, what files you read/wrote, etc. "
        "Each result includes the user message that triggered the tool call."
    ),
    parameters={
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Filter by tool name (e.g. 'bash', 'read_file', 'web_search'). Leave empty for all tools.",
            },
            "keyword": {
                "type": "string",
                "description": "Search keyword to match in tool arguments or results.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 20, max 50).",
            },
        },
        "required": [],
    },
    handler=_search_tool_history,
)
