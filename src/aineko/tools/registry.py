"""Tool registry — maps tool names to callables and provides schema for the LLM."""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

ToolResult = str | list[dict[str, Any]]


@dataclass
class ToolDef:
    """A tool the agent can call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Coroutine[Any, Any, ToolResult]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        """Return tool definitions in OpenAI-compatible function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return await tool.handler(**arguments)
        except Exception as e:
            return f"Error running {name}: {e}"
