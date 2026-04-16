"""Async Kimi API client — Anthropic Messages API format."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from aineko.config import KimiSettings
from aineko.messaging_dedupe import MessageDeduplicator
from aineko.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

DOOM_LOOP_THRESHOLD = 3


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallRecord:
    """Record of a tool call + result for persistence."""

    tool_name: str
    arguments: str  # JSON string
    result: str


@dataclass
class ChatResponse:
    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    tool_history: list[ToolCallRecord] = field(default_factory=list)
    intermediate_messages: list[str] = field(default_factory=list)


class KimiClient:
    def __init__(self, settings: KimiSettings) -> None:
        self._settings = settings
        headers: dict[str, str] = {
            "x-api-key": settings.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if settings.user_agent:
            headers["User-Agent"] = settings.user_agent
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            headers=headers,
            timeout=120,
        )

    def _build_tools(self, tools: ToolRegistry) -> list[dict[str, Any]]:
        """Convert tool registry to Anthropic tool format."""
        result = []
        for schema in tools.schemas():
            func = schema["function"]
            result.append(
                {
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func["parameters"],
                }
            )
        return result

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: ToolRegistry | None = None,
    ) -> ChatResponse:
        """Send a chat request using Anthropic Messages API format."""
        # Extract system prompt from messages
        system_text = None
        conversation: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                # Accumulate system messages
                content = m.get("content", "")
                if system_text is None:
                    system_text = content
                else:
                    system_text += "\n\n" + content
            else:
                conversation.append(m)

        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": conversation,
            "max_tokens": self._settings.max_tokens,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
        }
        if system_text:
            payload["system"] = system_text
        if self._settings.thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(16_000, self._settings.max_tokens // 2),
            }
        if tools and tools.schemas():
            payload["tools"] = self._build_tools(tools)
            payload["tool_choice"] = {"type": "auto"}

        logger.info(
            "llm request payload",
            extra={
                "event": "llm_request",
                "endpoint": "/messages",
                "model": payload.get("model"),
                "tool_count": len(payload.get("tools", [])),
                "tool_names": [t["name"] for t in payload.get("tools", [])],
                "thinking": payload.get("thinking"),
                "msg_count": len(payload.get("messages", [])),
            },
        )

        # Retry on transient errors (status codes and network timeouts)
        retryable = {429, 500, 502, 503, 504, 529}
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                resp = await self._http.post("/messages", json=payload)
            except httpx.TimeoutException:
                if attempt < max_retries:
                    delay = 2**attempt
                    logger.warning(
                        "LLM API timeout, retrying in %ds (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            if resp.status_code in retryable and attempt < max_retries:
                delay = 2**attempt
                logger.warning(
                    "LLM API %d, retrying in %ds (attempt %d/%d)",
                    resp.status_code,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)
                continue
            break
        if resp.status_code >= 400:
            logger.error(
                "llm api error", extra={"event": "llm_error", "error": resp.text}
            )
        resp.raise_for_status()
        data = resp.json()

        # Parse Anthropic response format
        content_blocks = data.get("content") or []
        usage = data.get("usage", {})
        stop_reason = data.get("stop_reason", "")

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                reasoning_parts.append(block.get("thinking", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )

        result = ChatResponse(
            content="\n".join(text_parts) if text_parts else "",
            reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=stop_reason,
        )

        if result.reasoning_content:
            logger.info(
                "llm reasoning",
                extra={
                    "event": "llm_reasoning",
                    "model": self._settings.model,
                    "reasoning": result.reasoning_content,
                },
            )

        logger.info(
            "llm response",
            extra={
                "event": "llm_response",
                "model": self._settings.model,
                "tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                "finish_reason": stop_reason,
                "raw_tool_call_count": len(tool_calls),
                "content": (result.content or "")[:300],
            },
        )

        return result

    def _build_assistant_content(self, response: ChatResponse) -> list[dict[str, Any]]:
        """Build Anthropic-format content blocks for an assistant message."""
        blocks: list[dict[str, Any]] = []
        if response.reasoning_content:
            blocks.append({"type": "thinking", "thinking": response.reasoning_content})
        if response.content:
            blocks.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )
        return blocks

    def _build_tool_results(
        self, results: list[tuple[str, str, bool]]
    ) -> dict[str, Any]:
        """Build a user message containing tool_result blocks.

        Args:
            results: list of (tool_use_id, content, is_error)
        """
        blocks = []
        for tool_use_id, content, is_error in results:
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }
            if is_error:
                block["is_error"] = True
            blocks.append(block)
        return {"role": "user", "content": blocks}

    async def chat_loop(
        self,
        messages: list[dict[str, Any]],
        tools: ToolRegistry,
        checkpoint_every: int = 10,
        max_rounds: int = 100,
        on_intermediate: Callable[[str], Awaitable[None]] | None = None,
    ) -> ChatResponse:
        """Run the chat → tool call → chat loop until the agent produces a final response."""
        tool_history: list[ToolCallRecord] = []
        intermediate_messages: list[str] = []
        recent_calls: list[tuple[str, str]] = []
        rounds_since_checkpoint = 0
        dedup = MessageDeduplicator()

        for round_num in range(max_rounds):
            response = await self.chat(messages, tools)

            if not response.tool_calls:
                response.tool_history = tool_history
                response.intermediate_messages = intermediate_messages
                return response

            # Send intermediate text immediately via callback, and track for dedup.
            if response.content:
                intermediate_messages.append(response.content)
                if on_intermediate:
                    await on_intermediate(response.content)

            # Append assistant message with content blocks
            messages.append(
                {
                    "role": "assistant",
                    "content": self._build_assistant_content(response),
                }
            )

            # Execute tool calls and collect results
            tool_results: list[tuple[str, str, bool]] = []

            for tc in response.tool_calls:
                args_json = json.dumps(tc.arguments, sort_keys=True)

                # Doom loop detection
                recent_calls.append((tc.name, args_json))
                if len(recent_calls) >= DOOM_LOOP_THRESHOLD:
                    tail = recent_calls[-DOOM_LOOP_THRESHOLD:]
                    if all(t == tail[0] for t in tail):
                        logger.warning(
                            "doom loop detected",
                            extra={
                                "event": "doom_loop",
                                "tool": tc.name,
                                "tool_args": tc.arguments,
                            },
                        )
                        error_msg = (
                            f"Error: doom loop detected — you called {tc.name} with "
                            f"identical arguments {DOOM_LOOP_THRESHOLD} times in a row. "
                            f"Try a different approach."
                        )
                        tool_results.append((tc.id, error_msg, True))
                        tool_history.append(
                            ToolCallRecord(tc.name, args_json, error_msg)
                        )
                        continue

                # Deduplicate send_message calls
                if tc.name == "send_message":
                    msg_text = tc.arguments.get("message", "")
                    if dedup.is_duplicate(msg_text):
                        logger.info(
                            "skipping duplicate send_message",
                            extra={"event": "send_dedup", "tool": tc.name},
                        )
                        result = "sent (deduplicated)"
                        tool_results.append((tc.id, result, False))
                        tool_history.append(ToolCallRecord(tc.name, args_json, result))
                        continue

                logger.info(
                    "tool call",
                    extra={
                        "event": "tool_call",
                        "tool": tc.name,
                        "tool_args": tc.arguments,
                    },
                )
                result = await tools.call(tc.name, tc.arguments)
                result_str = result if isinstance(result, str) else "[content blocks]"
                tool_history.append(ToolCallRecord(tc.name, args_json, result_str))

                # Track successful send_message for dedup
                if tc.name == "send_message":
                    dedup.track_sent(tc.arguments.get("message", ""))
                logger.info(
                    "tool result",
                    extra={
                        "event": "tool_result",
                        "tool": tc.name,
                        "result_len": (
                            len(result) if isinstance(result, str) else len(result)
                        ),
                        "result_preview": (
                            result[:200]
                            if isinstance(result, str)
                            else str(result[0])[:200]
                        ),
                    },
                )
                tool_results.append((tc.id, result, False))

            # Append tool results as a single user message with tool_result blocks
            messages.append(self._build_tool_results(tool_results))

            rounds_since_checkpoint += 1

            # Checkpoint: force progress update
            if rounds_since_checkpoint >= checkpoint_every:
                logger.info(
                    "checkpoint: requesting progress update (round %d)",
                    round_num + 1,
                    extra={"event": "checkpoint", "round": round_num + 1},
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Progress check — you've been working for a while. "
                            "Send the user a brief status update on what you've done "
                            "and what's left, then continue working."
                        ),
                    }
                )
                update = await self.chat(messages, tools=None)
                if update.content and tools.get("send_message"):
                    await tools.call("send_message", {"message": update.content})
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": update.content or "continuing..."}
                        ],
                    }
                )
                rounds_since_checkpoint = 0

        # Hit hard ceiling
        logger.warning("hit max tool rounds (%d), forcing final response", max_rounds)
        messages.append(
            {
                "role": "user",
                "content": "You've reached the maximum tool call limit. Summarize what you've done so far and respond to the user now.",
            }
        )
        final = await self.chat(messages, tools=None)
        final.tool_history = tool_history
        return final

    async def close(self) -> None:
        await self._http.aclose()
