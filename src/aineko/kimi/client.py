"""Async Kimi API client with streaming and tool call support."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from aineko.config import KimiSettings
from aineko.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""


class KimiClient:
    def __init__(self, settings: KimiSettings) -> None:
        self._settings = settings
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: ToolRegistry | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to Kimi. Returns parsed response."""
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "stream": False,
        }
        if tools and tools.schemas():
            payload["tools"] = tools.schemas()

        resp = await self._http.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]

        result = ChatResponse(
            content=msg.get("content", "") or "",
            finish_reason=choice.get("finish_reason", ""),
            usage=data.get("usage", {}),
        )

        # Parse tool calls if present
        for tc in msg.get("tool_calls", []):
            fn = tc["function"]
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            result.tool_calls.append(
                ToolCall(id=tc["id"], name=fn["name"], arguments=args)
            )

        return result

    async def chat_loop(
        self,
        messages: list[dict[str, Any]],
        tools: ToolRegistry,
        max_rounds: int = 10,
    ) -> ChatResponse:
        """Run the chat → tool call → chat loop until the agent produces a final response."""
        for _ in range(max_rounds):
            response = await self.chat(messages, tools)

            if not response.tool_calls:
                return response

            # Append assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ],
            })

            # Execute each tool call and append results
            for tc in response.tool_calls:
                logger.info("Tool call: %s(%s)", tc.name, tc.arguments)
                result = await tools.call(tc.name, tc.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        logger.warning("Agent hit max tool call rounds (%d)", max_rounds)
        return response

    async def close(self) -> None:
        await self._http.aclose()
