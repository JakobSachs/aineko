"""Tests for Anthropic Messages API format (Kimi Coding endpoint)."""

import asyncio

import pytest
from unittest.mock import MagicMock

from aineko.kimi.client import KimiClient, ChatResponse, ToolCall
from aineko.tools.registry import ToolDef, ToolRegistry


def _make_settings(**overrides):
    defaults = {
        "api_key": "test-key",
        "base_url": "https://api.kimi.com/coding/v1",
        "model": "k2p5",
        "user_agent": "claude-code/0.1.0",
        "max_context_tokens": 262144,
        "thinking": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 32000,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


# --- Request format ---


class TestRequestFormat:
    """Verify we send Anthropic Messages API format."""

    @pytest.mark.asyncio
    async def test_posts_to_messages_endpoint(self):
        """Should POST to /messages, not /chat/completions."""
        client = KimiClient(_make_settings())
        posted_url = None

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }

        async def capture_post(url, **kwargs):
            nonlocal posted_url
            posted_url = url
            return FakeResp()

        client._http.post = capture_post
        await client.chat([{"role": "user", "content": "hi"}])
        assert posted_url == "/messages"

    @pytest.mark.asyncio
    async def test_system_prompt_extracted_to_top_level(self):
        """System messages should become the top-level 'system' field."""
        client = KimiClient(_make_settings())
        captured_payload = {}

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }

        async def capture_post(url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeResp()

        client._http.post = capture_post
        await client.chat(
            [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hello"},
            ]
        )

        # System should be top-level, not in messages
        assert captured_payload.get("system") == "you are helpful"
        msg_roles = [m["role"] for m in captured_payload["messages"]]
        assert "system" not in msg_roles
        assert "user" in msg_roles

    @pytest.mark.asyncio
    async def test_thinking_includes_budget_tokens(self):
        """Thinking config should include budget_tokens."""
        client = KimiClient(_make_settings(thinking=True))
        captured_payload = {}

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }

        async def capture_post(url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeResp()

        client._http.post = capture_post
        await client.chat([{"role": "user", "content": "hi"}])

        thinking = captured_payload.get("thinking", {})
        assert thinking["type"] == "enabled"
        assert "budget_tokens" in thinking
        assert thinking["budget_tokens"] == 16000

    @pytest.mark.asyncio
    async def test_tools_use_input_schema(self):
        """Tool definitions should use 'input_schema', not 'parameters'."""
        client = KimiClient(_make_settings())
        captured_payload = {}

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }

        async def capture_post(url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeResp()

        client._http.post = capture_post

        registry = ToolRegistry()

        async def fake(**kw):
            return "ok"

        registry.register(
            ToolDef(
                name="bash",
                description="run commands",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                handler=fake,
            )
        )

        await client.chat([{"role": "user", "content": "hi"}], tools=registry)

        tools = captured_payload.get("tools", [])
        assert len(tools) == 1
        assert "input_schema" in tools[0]
        assert "parameters" not in tools[0]
        assert tools[0]["name"] == "bash"


# --- Response parsing ---


class TestResponseParsing:
    """Verify we parse Anthropic response format correctly."""

    @pytest.mark.asyncio
    async def test_parse_text_response(self):
        client = KimiClient(_make_settings())

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello world"}],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }

        async def fake_post(url, **kw):
            return FakeResp()

        client._http.post = fake_post

        result = await client.chat([{"role": "user", "content": "hi"}])
        assert result.content == "hello world"
        assert result.finish_reason == "end_turn"
        assert len(result.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_parse_tool_use_response(self):
        client = KimiClient(_make_settings())

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "let me check"},
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "bash",
                            "input": {"command": "ls"},
                        },
                    ],
                    "model": "k2p5",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 20, "output_tokens": 15},
                }

        async def fake_post(url, **kw):
            return FakeResp()

        client._http.post = fake_post

        result = await client.chat([{"role": "user", "content": "list files"}])
        assert result.content == "let me check"
        assert result.finish_reason == "tool_use"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_123"
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[0].arguments == {"command": "ls"}

    @pytest.mark.asyncio
    async def test_parse_thinking_response(self):
        client = KimiClient(_make_settings())

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I should analyze this..."},
                        {"type": "text", "text": "here's my answer"},
                    ],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                }

        async def fake_post(url, **kw):
            return FakeResp()

        client._http.post = fake_post

        result = await client.chat([{"role": "user", "content": "think about this"}])
        assert result.content == "here's my answer"
        assert result.reasoning_content == "I should analyze this..."


# --- Tool result format in chat_loop ---


class TestToolResultFormat:
    """Verify tool results are sent as Anthropic tool_result blocks."""

    @pytest.mark.asyncio
    async def test_tool_result_sent_as_content_block(self):
        """Tool results should be user messages with tool_result content blocks."""
        client = KimiClient(_make_settings())
        messages: list[dict] = [{"role": "user", "content": "list files"}]

        call_count = 0

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="toolu_1", name="bash", arguments={"command": "ls"})
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done", finish_reason="end_turn")

        client.chat = mock_chat

        async def fake_bash(**kw):
            return "file.txt"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="bash",
                description="run",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=fake_bash,
            )
        )

        await client.chat_loop(messages, registry)

        # Find the tool result message
        tool_result_msgs = [
            m
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), list)
        ]
        assert len(tool_result_msgs) >= 1
        content = tool_result_msgs[0]["content"]
        assert any(
            block.get("type") == "tool_result" and block.get("tool_use_id") == "toolu_1"
            for block in content
        )

    @pytest.mark.asyncio
    async def test_assistant_tool_use_preserved_in_messages(self):
        """Assistant messages with tool_use should preserve content blocks."""
        client = KimiClient(_make_settings())
        messages: list[dict] = [{"role": "user", "content": "hi"}]

        call_count = 0

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="thinking...",
                    tool_calls=[
                        ToolCall(id="toolu_1", name="bash", arguments={"command": "ls"})
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done")

        client.chat = mock_chat

        async def fake_bash(**kw):
            return "ok"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="bash",
                description="",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=fake_bash,
            )
        )

        await client.chat_loop(messages, registry)

        # Find assistant message — should have content blocks, not just a string
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) >= 1
        content = assistant_msgs[0]["content"]
        assert isinstance(content, list)
        types = [block["type"] for block in content]
        assert "tool_use" in types


# --- Interjection during final (end_turn) response ---


class TestInterjectionRace:
    """Regression: a user message that lands in the interject queue while the
    LLM is producing its final (no-tool-calls) response must not be silently
    dropped on the floor when chat_loop returns.

    Real-world incident (2026-04-28 09:59:56): user sent a message ~100ms
    before the final end_turn arrived; queue.py logged `interject_enqueue`,
    but the agent never addressed it because chat_loop only drains the queue
    at the *start* of each round.
    """

    @pytest.mark.asyncio
    async def test_interject_during_final_turn_is_not_dropped(self):
        client = KimiClient(_make_settings())
        messages: list[dict] = [{"role": "user", "content": "thanks"}]

        interject_queue: asyncio.Queue[str] = asyncio.Queue()
        call_count = 0

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate the LLM producing a final response while a user
                # message arrives in the interject queue mid-flight.
                interject_queue.put_nowait("Can ya help me make a list")
                # tiny await so the put_nowait settles like a real race
                await asyncio.sleep(0)
                return ChatResponse(
                    content="safe travels tomorrow",
                    finish_reason="end_turn",
                )
            # If the loop correctly re-runs after seeing the interjection,
            # this second call is what addresses the new user message.
            return ChatResponse(
                content="sure, what are you packing?",
                finish_reason="end_turn",
            )

        client.chat = mock_chat
        registry = ToolRegistry()

        await client.chat_loop(messages, registry, interject_queue=interject_queue)

        # The interjection must have been consumed (not stranded in the queue).
        assert (
            interject_queue.empty()
        ), "interjection left in queue — message was dropped on end_turn"

        # And the LLM must have been given a chance to actually respond to it:
        # either by re-running the loop, or by some equivalent mechanism that
        # surfaces the user's text into the conversation.
        all_user_text = " ".join(
            (m["content"] if isinstance(m["content"], str) else "")
            for m in messages
            if m.get("role") == "user"
        )
        assert "Can ya help me make a list" in all_user_text or call_count >= 2, (
            "interjected message was never injected into history nor triggered "
            "a follow-up LLM round"
        )

    @pytest.mark.asyncio
    async def test_interject_during_max_rounds_final_call_is_not_dropped(self):
        """Same race, but on the max_rounds exit path: the forced final
        `chat(...)` at the bottom of the loop also runs while interjections
        can land. That exit must go through the same gate."""
        client = KimiClient(_make_settings())
        messages: list[dict] = [{"role": "user", "content": "do stuff"}]
        interject_queue: asyncio.Queue[str] = asyncio.Queue()

        call_count = 0
        forced_final_round = False
        interjected_once = False

        async def mock_chat(msgs, tools=None):
            nonlocal call_count, forced_final_round, interjected_once
            call_count += 1
            # Keep emitting tool calls until we hit max_rounds → forces the
            # "summarize now" path. tools=None signals the forced final call.
            if tools is None:
                forced_final_round = True
                if not interjected_once:
                    interject_queue.put_nowait("wait one more thing")
                    interjected_once = True
                    await asyncio.sleep(0)
                return ChatResponse(content="summary", finish_reason="end_turn")
            return ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"t{call_count}",
                        name="bash",
                        arguments={"command": f"echo {call_count}"},
                    )
                ],
                finish_reason="tool_use",
            )

        client.chat = mock_chat

        async def fake_bash(**kw):
            return "ok"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="bash",
                description="",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=fake_bash,
            )
        )

        await client.chat_loop(
            messages,
            registry,
            max_rounds=2,
            interject_queue=interject_queue,
        )

        assert forced_final_round, "test setup failed — never hit max_rounds path"
        assert (
            interject_queue.empty()
        ), "interjection during max_rounds final chat() was dropped"
