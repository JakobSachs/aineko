"""Async LLM client backed by LiteLLM."""

import asyncio
import base64
import json
import logging
import os
import platform
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
from litellm import acompletion

from aineko.config import LLMSettings
from aineko.messaging_dedupe import MessageDeduplicator
from aineko.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

DOOM_LOOP_THRESHOLD = 3
OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_ISSUER = "https://auth.openai.com"
OPENAI_CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"

_INTENT_TO_CONTINUE_RE = re.compile(
    r"(?is)"
    r"(?:\b(?:i|we)\s+(?:need|should|will|would|can|must|am|are|['’]?ll|['’]?m)\s+"
    r"(?:going\s+)?to\s+)"
    r"(?:check|inspect|look|run|test|try|restart|fix|patch|debug|investigate|"
    r"verify|confirm|read|search|grep|open|list|install|build|deploy|copy|send|"
    r"start|stop|kill|serve|export|rerun)"
    r"|(?:\bnext\b|\bnow\b|\bthen\b).{0,80}\b"
    r"(?:check|inspect|look|run|test|try|restart|fix|patch|debug|investigate|"
    r"verify|confirm|read|search|grep|open|list|install|build|deploy|copy|send|"
    r"start|stop|kill|serve|export|rerun)"
)
_INTENT_BLOCKED_RE = re.compile(
    r"(?is)\b(?:blocked|need (?:your|user) (?:input|permission|approval)|"
    r"can't proceed|cannot proceed|which .*\?|what .*\?)\b"
)


def _looks_like_unfinished_intent(text: str) -> bool:
    """Heuristic: final answer says it will do toolable work, but stopped."""
    cleaned = text.strip()
    if not cleaned or _INTENT_BLOCKED_RE.search(cleaned):
        return False
    return bool(_INTENT_TO_CONTINUE_RE.search(cleaned))


def _drain_interjections(
    messages: list[dict[str, Any]],
    queue: "asyncio.Queue[str] | None",
) -> None:
    """Pull any pending user messages off the queue and inject them into history.

    Merges into the trailing user message if there is one (tool_result block)
    to avoid consecutive-user-role messages the API rejects.
    """
    if queue is None:
        return
    bodies: list[str] = []
    while True:
        try:
            bodies.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    if not bodies:
        return
    combined = "\n\n".join(f"[user interjection]: {b}" for b in bodies)
    logger.info(
        "injecting user interjections into turn",
        extra={"event": "interject_inject", "count": len(bodies)},
    )
    if messages and messages[-1].get("role") == "user":
        last = messages[-1]
        existing = last.get("content")
        if isinstance(existing, list):
            last["content"] = existing + [{"type": "text", "text": combined}]
        else:
            last["content"] = f"{existing}\n\n{combined}"
    else:
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": combined}]}
        )


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
    thinking_blocks: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    tool_history: list[ToolCallRecord] = field(default_factory=list)
    intermediate_messages: list[str] = field(default_factory=list)
    # API-shape message dicts produced during chat_loop, in order: each
    # assistant turn (with thinking + tool_use blocks) followed by its
    # paired tool_result user turn, ending with the final assistant turn.
    # Used by handler.persist_response to write blocks to DB so thinking
    # blocks survive replay.
    new_messages: list[dict[str, Any]] = field(default_factory=list)


class KimiClient:
    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._completion = acompletion
        self._http = SimpleNamespace(post=None)
        self._anthropic_http = httpx.AsyncClient(timeout=120)
        self._opencode_http = httpx.AsyncClient(timeout=120)
        self._extra_headers: dict[str, str] = {}
        if settings.user_agent:
            self._extra_headers["User-Agent"] = settings.user_agent

    def _build_tools(self, tools: ToolRegistry) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool definitions for LiteLLM."""
        return tools.schemas()

    @staticmethod
    def _tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for schema in tools:
            func = schema["function"]
            result.append(
                {
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func["parameters"],
                }
            )
        return result

    def _chat_kwargs(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = list(payload["messages"])
        if payload.get("system"):
            messages.insert(0, {"role": "system", "content": payload["system"]})
        kwargs: dict[str, Any] = {
            "model": payload["model"],
            "messages": messages,
            "max_tokens": payload["max_tokens"],
            "temperature": payload["temperature"],
            "top_p": payload["top_p"],
            "timeout": 120,
            "num_retries": 3,
        }
        if self._settings.api_key:
            kwargs["api_key"] = self._settings.api_key
        if self._settings.base_url:
            kwargs["api_base"] = self._settings.base_url
        if payload.get("tools"):
            kwargs["tools"] = payload["tools"]
            kwargs["tool_choice"] = payload.get("tool_choice", "auto")
        if self._extra_headers:
            kwargs["extra_headers"] = self._extra_headers
        # LiteLLM forwards unknown provider params when supported. This keeps
        # Kimi/Anthropic reasoning configurable without making it core logic.
        if payload.get("thinking"):
            kwargs["thinking"] = payload["thinking"]
        return kwargs

    def _uses_raw_anthropic_messages(self) -> bool:
        return self._settings.model.startswith("anthropic/") and bool(
            self._settings.base_url
        )

    def _uses_opencode_openai(self) -> bool:
        return self._settings.provider == "opencode-openai"

    def _anthropic_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        result["model"] = result["model"].removeprefix("anthropic/")
        if result.get("tools"):
            result["tools"] = self._tools_to_anthropic(result["tools"])
        return result

    async def _raw_anthropic_completion(self, payload: dict[str, Any]) -> Any:
        headers = {
            "x-api-key": self._settings.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        resp = await self._anthropic_http.post(
            f"{self._settings.base_url.rstrip('/')}/messages",
            headers=headers,
            json=self._anthropic_payload(payload),
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenCode OpenAI returned non-json response "
                f"content-type={resp.headers.get('content-type')}: {resp.text[:1000]}"
            ) from exc

    @staticmethod
    def _jwt_claims(token: str) -> dict[str, Any]:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        try:
            padded = parts[1] + "=" * (-len(parts[1]) % 4)
            return json.loads(base64.urlsafe_b64decode(padded))
        except (ValueError, json.JSONDecodeError):
            return {}

    @classmethod
    def _account_id_from_tokens(cls, tokens: dict[str, Any]) -> str:
        for key in ("id_token", "access_token"):
            token = tokens.get(key)
            if not token:
                continue
            claims = cls._jwt_claims(token)
            auth_claims = claims.get("https://api.openai.com/auth") or {}
            account_id = (
                claims.get("chatgpt_account_id")
                or auth_claims.get("chatgpt_account_id")
                or (claims.get("organizations") or [{}])[0].get("id")
            )
            if account_id:
                return account_id
        return ""

    async def _load_opencode_auth(self) -> dict[str, Any]:
        path = self._settings.opencode_auth_path
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise RuntimeError(f"OpenCode auth file not found: {path}") from exc
        auth = data.get("openai") or {}
        if auth.get("type") != "oauth":
            raise RuntimeError(
                "OpenCode openai auth is not oauth; run opencode /connect"
            )
        if (
            auth.get("access")
            and auth.get("expires", 0) > int(time.time() * 1000) + 30_000
        ):
            return auth
        if not auth.get("refresh"):
            raise RuntimeError("OpenCode openai oauth record has no refresh token")
        resp = await self._opencode_http.post(
            f"{OPENAI_CODEX_ISSUER}/oauth/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": auth["refresh"],
                "client_id": OPENAI_CODEX_CLIENT_ID,
            },
        )
        resp.raise_for_status()
        tokens = resp.json()
        updated = {
            "type": "oauth",
            "refresh": tokens["refresh_token"],
            "access": tokens["access_token"],
            "expires": int(time.time() * 1000)
            + int(tokens.get("expires_in", 3600)) * 1000,
            "accountId": self._account_id_from_tokens(tokens) or auth.get("accountId"),
        }
        data["openai"] = updated
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        os.chmod(tmp, 0o600)
        tmp.replace(path)
        return updated

    @staticmethod
    def _content_to_openai_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)
        parts: list[str] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_result":
                marker = "tool_result"
                if block.get("is_error"):
                    marker = "tool_error"
                parts.append(
                    f"[{marker} {block.get('tool_use_id', '')}]\n{block.get('content', '')}"
                )
            elif btype not in {"thinking", "tool_use"}:
                parts.append(json.dumps(block, ensure_ascii=False))
        return "\n\n".join(p for p in parts if p)

    def _opencode_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        instructions = payload.get("system")
        input_items = []
        for message in payload["messages"]:
            role = message.get("role", "user")
            if role == "assistant":
                role = "assistant"
            elif role not in {"user", "developer", "system"}:
                role = "user"
            text = self._content_to_openai_text(message.get("content", ""))
            if text:
                content_type = "output_text" if role == "assistant" else "input_text"
                input_items.append(
                    {
                        "role": role,
                        "content": [{"type": content_type, "text": text}],
                    }
                )
        result: dict[str, Any] = {
            "model": payload["model"].removeprefix("openai/"),
            "input": input_items,
            "store": False,
            "stream": True,
            "reasoning": {"effort": "low"},
        }
        result["instructions"] = instructions or "You are aineko, a concise assistant."
        if payload.get("tools"):
            result["tools"] = [
                {
                    "type": "function",
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "parameters": tool["function"].get("parameters", {}),
                }
                for tool in payload["tools"]
            ]
            result["tool_choice"] = "auto"
        return result

    async def _opencode_openai_completion(self, payload: dict[str, Any]) -> Any:
        auth = await self._load_opencode_auth()
        headers = {
            "Authorization": f"Bearer {auth['access']}",
            "Content-Type": "application/json",
            "originator": "opencode",
            "User-Agent": f"opencode/0.0.0 ({platform.system().lower()} {platform.release()}; {platform.machine()})",
            "session-id": "aineko",
        }
        if auth.get("accountId"):
            headers["ChatGPT-Account-Id"] = auth["accountId"]
        resp = await self._opencode_http.post(
            OPENAI_CODEX_ENDPOINT,
            headers=headers,
            json=self._opencode_payload(payload),
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"OpenCode OpenAI request failed: {resp.text}") from exc
        if "text/event-stream" in resp.headers.get(
            "content-type", ""
        ) or resp.text.startswith("event:"):
            completed: dict[str, Any] | None = None
            text_parts: list[str] = []
            output_items: list[dict[str, Any]] = []
            event_name = ""
            for line in resp.text.splitlines():
                if line.startswith("event: "):
                    event_name = line.removeprefix("event: ")
                    continue
                if not line.startswith("data: "):
                    continue
                raw = line.removeprefix("data: ")
                if raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if (
                    event_name == "response.output_text.delta"
                    or event.get("type") == "response.output_text.delta"
                ):
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        text_parts.append(delta)
                if (
                    event_name == "response.output_item.done"
                    or event.get("type") == "response.output_item.done"
                ):
                    item = event.get("item")
                    if isinstance(item, dict):
                        output_items.append(item)
                if (
                    event_name == "response.completed"
                    or event.get("type") == "response.completed"
                ):
                    completed = event.get("response")
            if completed is not None:
                if output_items:
                    completed["output"] = output_items
                elif text_parts and not completed.get("output"):
                    completed["output"] = [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "".join(text_parts)}
                            ],
                        }
                    ]
                return completed
            raise RuntimeError(
                f"OpenCode OpenAI stream missing completion: {resp.text[:1000]}"
            )
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenCode OpenAI returned non-json response "
                f"content-type={resp.headers.get('content-type')}: {resp.text[:1000]}"
            ) from exc

    @staticmethod
    def _value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _parse_response(self, data: Any) -> ChatResponse:
        if self._value(data, "output") is not None:
            return self._parse_openai_responses_response(data)
        # Anthropic-shaped responses can come back when using compatible APIs.
        if self._value(data, "content") is not None:
            return self._parse_anthropic_response(data)

        choices = self._value(data, "choices", []) or []
        choice = choices[0] if choices else {}
        message = self._value(choice, "message", {}) or {}
        usage = self._value(data, "usage", {}) or {}
        finish_reason = self._value(choice, "finish_reason", "") or ""

        content = self._value(message, "content", "") or ""
        reasoning_content = self._value(message, "reasoning_content")
        tool_calls: list[ToolCall] = []
        for tc in self._value(message, "tool_calls", []) or []:
            function = self._value(tc, "function", {}) or {}
            raw_args = self._value(function, "arguments", {}) or {}
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"arguments": raw_args}
            else:
                args = raw_args
            tool_calls.append(
                ToolCall(
                    id=self._value(tc, "id", ""),
                    name=self._value(function, "name", ""),
                    arguments=args,
                )
            )
        return ChatResponse(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )

    def _parse_openai_responses_response(self, data: Any) -> ChatResponse:
        output = self._value(data, "output", []) or []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in output:
            item_type = self._value(item, "type", "")
            if item_type == "message":
                for part in self._value(item, "content", []) or []:
                    ptype = self._value(part, "type", "")
                    if ptype in {"output_text", "text"}:
                        text_parts.append(self._value(part, "text", "") or "")
            elif item_type == "function_call":
                raw_args = self._value(item, "arguments", "{}") or "{}"
                try:
                    args = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )
                except json.JSONDecodeError:
                    args = {"arguments": raw_args}
                tool_calls.append(
                    ToolCall(
                        id=self._value(item, "call_id", "")
                        or self._value(item, "id", ""),
                        name=self._value(item, "name", ""),
                        arguments=args,
                    )
                )
            elif item_type == "reasoning":
                for part in self._value(item, "summary", []) or []:
                    text = self._value(part, "text", "")
                    if text:
                        reasoning_parts.append(text)
        usage = self._value(data, "usage", {}) or {}
        return ChatResponse(
            content="\n".join(text_parts),
            reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
            tool_calls=tool_calls,
            usage={
                "input_tokens": self._value(usage, "input_tokens", 0) or 0,
                "output_tokens": self._value(usage, "output_tokens", 0) or 0,
            },
            finish_reason=self._value(data, "status", ""),
        )

    def _parse_anthropic_response(self, data: Any) -> ChatResponse:
        content_blocks = self._value(data, "content", []) or []
        usage = self._value(data, "usage", {}) or {}
        stop_reason = self._value(data, "stop_reason", "") or ""

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        thinking_blocks: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                reasoning_parts.append(block.get("thinking", ""))
                thinking_blocks.append(
                    {k: v for k, v in block.items() if k != "type"}
                    | {"type": "thinking"}
                )
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )
        return ChatResponse(
            content="\n".join(text_parts) if text_parts else "",
            reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
            thinking_blocks=thinking_blocks,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=stop_reason,
        )

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
        thinking_enabled = self._settings.thinking
        if thinking_enabled:
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
                "tool_names": [
                    t.get("function", {}).get("name", t.get("name", ""))
                    for t in payload.get("tools", [])
                ],
                "thinking": payload.get("thinking"),
                "msg_count": len(payload.get("messages", [])),
            },
        )

        # Legacy test seam for old HTTP-level unit tests. Real requests go
        # through LiteLLM via `_completion`.
        if self._http.post is not None:
            legacy_payload = dict(payload)
            if payload.get("tools"):
                legacy_payload["tools"] = self._tools_to_anthropic(payload["tools"])
            resp = await self._http.post("/messages", json=legacy_payload)
            resp.raise_for_status()
            data = resp.json()
        elif self._uses_raw_anthropic_messages():
            data = await self._raw_anthropic_completion(payload)
        elif self._uses_opencode_openai():
            data = await self._opencode_openai_completion(payload)
        else:
            data = await self._completion(**self._chat_kwargs(payload))
        result = self._parse_response(data)

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
                "tokens": result.usage.get("input_tokens", 0)
                + result.usage.get("output_tokens", 0),
                "finish_reason": result.finish_reason,
                "raw_tool_call_count": len(result.tool_calls),
                "content": (result.content or "")[:300],
            },
        )

        return result

    def _build_assistant_content(self, response: ChatResponse) -> list[dict[str, Any]]:
        """Build Anthropic-format content blocks for an assistant message.

        Replays thinking blocks verbatim (preserving signatures) so the API
        accepts them in multi-turn context. When tool_use is present, drops
        any accompanying text content — the model tends to hallucinate
        tool-result-shaped preambles there, and feeding them back in-context
        trains it to do so more.
        """
        blocks: list[dict[str, Any]] = []
        # Replay thinking blocks verbatim (including signature).
        for tb in response.thinking_blocks:
            blocks.append(tb)
        # Only include text when there are no tool calls — otherwise it's a
        # pre-tool preamble and often contains fabricated results.
        if response.content and not response.tool_calls:
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
        interject_queue: asyncio.Queue[str] | None = None,
    ) -> ChatResponse:
        """Run the chat → tool call → chat loop until the agent produces a final response."""
        tool_history: list[ToolCallRecord] = []
        intermediate_messages: list[str] = []
        rounds_since_checkpoint = 0
        intent_guard_count = 0
        dedup = MessageDeduplicator()
        start_len = len(messages)

        # Outer loop wraps the round-runner so there is exactly *one* exit
        # gate. Anything that wants to leave chat_loop must pass through the
        # interjection check at the bottom — making it structurally impossible
        # to return while a user message is still pending in the queue.
        round_num = 0
        while True:
            pending: ChatResponse | None = None
            recent_calls = []

            while round_num < max_rounds:
                _drain_interjections(messages, interject_queue)
                response = await self.chat(messages, tools)
                round_num += 1

                if not response.tool_calls:
                    pending = response
                    break

                # Don't stream pre-tool text to the user: when content arrives
                # alongside tool_use, it's a preamble the model wrote before any
                # tool ran, and has been observed to contain hallucinated
                # tool-result-shaped output. The real answer lands after tools
                # finish, in the no-tool-calls branch above.
                if response.content:
                    logger.info(
                        "discarding pre-tool content",
                        extra={
                            "event": "pretool_content_discarded",
                            "preview": response.content[:200],
                        },
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": self._build_assistant_content(response),
                    }
                )

                tool_results: list[tuple[str, str, bool]] = []

                # When the model batches multiple send_message calls in one
                # response, they'd otherwise land back-to-back with no breathing
                # room. Space them out so the user sees a conversation, not a wall.
                send_message_count = sum(
                    1 for tc in response.tool_calls if tc.name == "send_message"
                )
                seen_send_messages = 0

                for tc in response.tool_calls:
                    args_json = json.dumps(tc.arguments, sort_keys=True)

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

                    if tc.name == "send_message":
                        seen_send_messages += 1
                        if send_message_count > 1 and seen_send_messages > 1:
                            await asyncio.sleep(1.5)

                    if tc.name == "send_message":
                        msg_text = tc.arguments.get("message", "")
                        if dedup.is_duplicate(msg_text):
                            logger.info(
                                "skipping duplicate send_message",
                                extra={"event": "send_dedup", "tool": tc.name},
                            )
                            result = "sent (deduplicated)"
                            tool_results.append((tc.id, result, False))
                            tool_history.append(
                                ToolCallRecord(tc.name, args_json, result)
                            )
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
                    result_str = (
                        result if isinstance(result, str) else "[content blocks]"
                    )
                    tool_history.append(ToolCallRecord(tc.name, args_json, result_str))

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

                messages.append(self._build_tool_results(tool_results))

                rounds_since_checkpoint += 1

                if rounds_since_checkpoint >= checkpoint_every:
                    logger.info(
                        "checkpoint: requesting progress update (round %d)",
                        round_num,
                        extra={"event": "checkpoint", "round": round_num},
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
                    checkpoint_blocks: list[dict[str, Any]] = list(
                        update.thinking_blocks
                    )
                    checkpoint_blocks.append(
                        {"type": "text", "text": update.content or "continuing..."}
                    )
                    messages.append({"role": "assistant", "content": checkpoint_blocks})
                    rounds_since_checkpoint = 0

            # Inner loop exited: either via `break` (model produced no tool
            # calls → `pending` is set) or by exhausting `max_rounds`.
            if pending is None:
                logger.warning(
                    "hit max tool rounds (%d), forcing final response", max_rounds
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You've reached the maximum tool call limit. "
                            "Summarize what you've done so far and respond to the user now."
                        ),
                    }
                )
                pending = await self.chat(messages, tools=None)

            # If the model produced a "I'll do X next" status instead of
            # actually doing X, do not expose that as a final answer. Feed it
            # back once/twice as an internal nudge so the next round performs
            # the concrete tool action. This fixes the common Matrix UX where
            # the user has to say "and??" to make the agent continue.
            if (
                pending.content
                and intent_guard_count < 2
                and _looks_like_unfinished_intent(pending.content)
            ):
                logger.info(
                    "unfinished intent detected, continuing turn",
                    extra={
                        "event": "intent_continue_guard",
                        "preview": pending.content[:200],
                    },
                )
                pending_blocks = self._build_assistant_content(pending)
                if pending_blocks:
                    messages.append({"role": "assistant", "content": pending_blocks})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue now: perform the next concrete action with "
                            "tools instead of ending with intent. If you are "
                            "blocked, say exactly what input or permission you need."
                        ),
                    }
                )
                intent_guard_count += 1
                max_rounds = round_num + max_rounds
                continue

            # ---- single exit gate ----
            # If a user message landed in the interject queue while we were
            # awaiting the LLM, don't return — commit the assistant turn and
            # loop back so the next round picks up the interjection via
            # `_drain_interjections`. Bump max_rounds so we don't immediately
            # trip the ceiling on a fresh user message.
            if interject_queue is not None and not interject_queue.empty():
                logger.info(
                    "interjection arrived during exit, re-entering loop",
                    extra={"event": "interject_at_exit"},
                )
                pending_blocks = self._build_assistant_content(pending)
                if pending_blocks:
                    messages.append({"role": "assistant", "content": pending_blocks})
                # Dispatch the in-flight final reply before pivoting to the
                # interjection — otherwise it's committed to history but never
                # reaches the user, and they only see the answer to the newer
                # message. on_intermediate routes to matrix.send_message.
                if pending.content and on_intermediate is not None:
                    await on_intermediate(pending.content)
                    intermediate_messages.append(pending.content)
                max_rounds = round_num + max_rounds
                continue

            pending.tool_history = tool_history
            pending.intermediate_messages = intermediate_messages
            final_blocks = self._build_assistant_content(pending)
            pending.new_messages = list(messages[start_len:])
            if final_blocks:
                pending.new_messages.append(
                    {"role": "assistant", "content": final_blocks}
                )
            return pending

    async def close(self) -> None:
        await self._anthropic_http.aclose()
        await self._opencode_http.aclose()
