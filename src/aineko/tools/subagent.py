"""Subagent tool — spawns a fresh agent loop for isolated research/task execution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aineko.tools.registry import ToolDef, ToolRegistry

if TYPE_CHECKING:
    from aineko.kimi.client import KimiClient

logger = logging.getLogger(__name__)

_SUBAGENT_SYSTEM = (
    "you are a focused task-execution agent. complete the given task thoroughly using your tools. "
    "return your findings as a clear, structured response. "
    "you do not interact with the user directly — your output goes back to the agent that spawned you."
)

# Tools the subagent must not have
_EXCLUDED_TOOLS = frozenset(
    {
        "create_skill",
        "memory",
        "search_tool_history",
        "search_chat",
        "read_calendar",
        "send_message",
        "send_file",
        "spawn_task",
        "list_background_tasks",
        "get_task_result",
        "run_subagent",  # no recursion
    }
)

_SUBAGENT_MAX_ROUNDS = 40


def make_subagent_tool(kimi: KimiClient, base_tools: ToolRegistry) -> ToolDef:
    """Create the run_subagent tool, wired to the given kimi client and base tool registry."""
    subagent_registry = ToolRegistry()
    for name, tool_def in base_tools._tools.items():
        if name not in _EXCLUDED_TOOLS:
            subagent_registry.register(tool_def)

    async def handler(task: str, context: str = "") -> str:
        content = task
        if context:
            content = f"Context:\n{context}\n\nTask:\n{task}"

        messages = [
            {"role": "system", "content": _SUBAGENT_SYSTEM},
            {"role": "user", "content": content},
        ]

        logger.info(
            "subagent spawned",
            extra={"event": "subagent_start", "task_preview": task[:120]},
        )

        response = await kimi.chat_loop(
            messages, subagent_registry, max_rounds=_SUBAGENT_MAX_ROUNDS
        )

        logger.info(
            "subagent done",
            extra={
                "event": "subagent_done",
                "tool_calls": len(response.tool_history),
                "result_len": len(response.content),
            },
        )

        return response.content or "(subagent returned no content)"

    return ToolDef(
        name="run_subagent",
        description=(
            "Spawn a focused subagent to complete a task in isolation — fresh context, "
            "no conversation history. Use this for research, multi-step lookups, or anything "
            "that would clutter the current context.\n\n"
            "The subagent has access to bash, file tools, web_search, and web_fetch. "
            "It returns its findings as text. It cannot message the user directly.\n\n"
            "Be specific in the task description — include what to research, "
            "what format to return results in, and any constraints."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "What the subagent should do. Be specific — include what to look up, "
                        "what format to return results in, and any constraints."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Optional background context from the current conversation "
                        "(e.g. user preferences, what's already been tried)."
                    ),
                },
            },
            "required": ["task"],
        },
        handler=handler,
    )
