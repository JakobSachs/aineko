"""Tests for Anthropic Messages API format (Kimi Coding endpoint)."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

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
