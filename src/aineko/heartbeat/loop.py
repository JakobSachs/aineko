"""Async heartbeat tick loop — extracted from app.py lifespan."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from aineko.handler import format_tool_footer
from aineko.tools.messaging import make_send_message_tool
from aineko.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from aineko.heartbeat.runner import HeartbeatRunner
    from aineko.kimi.client import KimiClient
    from aineko.matrix.client import MatrixConnector

logger = logging.getLogger(__name__)


async def heartbeat_tick_loop(
    heartbeat: HeartbeatRunner,
    kimi: KimiClient,
    tools: ToolRegistry,
    matrix: MatrixConnector,
    heartbeat_room: str,
    sys_prompt: str,
    *,
    interval: int = 0,
) -> None:
    """Run heartbeat ticks forever. Designed to be wrapped in asyncio.create_task."""
    while True:
        await asyncio.sleep(interval)
        try:
            if not heartbeat.should_run():
                continue

            tasks_content = heartbeat.get_tasks()
            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": (
                        "Heartbeat check-in. Review your tasks and take "
                        "any appropriate action.\n\n" + tasks_content
                    ),
                },
            ]

            hb_tools = ToolRegistry()
            for tool_def in tools._tools.values():
                hb_tools.register(tool_def)
            hb_tools.register(make_send_message_tool(matrix, heartbeat_room, []))

            response = await kimi.chat_loop(messages, hb_tools)

            if response.content and heartbeat.should_deliver(response.content):
                if heartbeat_room:
                    footer = format_tool_footer(response.tool_history)
                    await matrix.send_message(heartbeat_room, response.content + footer)
        except Exception:
            logger.exception("Heartbeat tick failed")
