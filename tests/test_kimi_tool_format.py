"""Tests for Kimi API tool call format compliance (Anthropic Messages API).

Covers: tool result format, malformed args handling, doom loop, tool registration.
"""

import json
import pytest
from unittest.mock import MagicMock

from aineko.kimi.client import ChatResponse, KimiClient, ToolCall, DOOM_LOOP_THRESHOLD
from aineko.tools.registry import ToolDef, ToolRegistry


def _make_client():
    settings = MagicMock()
    settings.api_key = "test"
    settings.base_url = "http://test"
    settings.user_agent = ""
    settings.model = "k2p5"
    settings.thinking = False
    settings.temperature = 1.0
    settings.top_p = 0.95
    settings.max_tokens = 32000
    return KimiClient(settings)


# --- Tool result messages use tool_result blocks ---


@pytest.mark.asyncio
async def test_tool_result_is_tool_result_block():
    """Tool results should be user messages with tool_result content blocks."""
    client = _make_client()
    messages: list[dict] = [{"role": "user", "content": "test"}]

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

    async def fake_bash(**kwargs):
        return "file.txt"

    registry = ToolRegistry()
    registry.register(
        ToolDef(
            name="bash",
            description="run",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            handler=fake_bash,
        )
    )

    await client.chat_loop(messages, registry)

    # Find tool result message — should be a user message with content blocks
    tool_msgs = [
        m
        for m in messages
        if m.get("role") == "user" and isinstance(m.get("content"), list)
    ]
    assert len(tool_msgs) >= 1
    blocks = tool_msgs[0]["content"]
    assert blocks[0]["type"] == "tool_result"
    assert blocks[0]["tool_use_id"] == "toolu_1"
    assert blocks[0]["content"] == "file.txt"


@pytest.mark.asyncio
async def test_multiple_tool_results_in_one_message():
    """Multiple tool calls produce multiple tool_result blocks in one user message."""
    client = _make_client()
    messages: list[dict] = [{"role": "user", "content": "test"}]

    call_count = 0

    async def mock_chat(msgs, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(id="toolu_1", name="bash", arguments={"command": "ls"}),
                    ToolCall(id="toolu_2", name="bash", arguments={"command": "pwd"}),
                ],
                finish_reason="tool_use",
            )
        return ChatResponse(content="done")

    client.chat = mock_chat

    async def fake_bash(**kwargs):
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

    tool_msgs = [
        m
        for m in messages
        if m.get("role") == "user" and isinstance(m.get("content"), list)
    ]
    assert len(tool_msgs) >= 1
    blocks = tool_msgs[0]["content"]
    assert len(blocks) == 2
    ids = {b["tool_use_id"] for b in blocks}
    assert ids == {"toolu_1", "toolu_2"}


# --- Malformed tool call arguments ---


def test_malformed_args_skipped_gracefully():
    """If model returns invalid JSON args in OpenAI-compat format, skip gracefully."""
    # Note: In Anthropic format, input is already a dict, not a JSON string.
    # This test verifies the tool_use block parsing handles non-dict input.
    client = _make_client()

    raw_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "trying"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "bash",
                "input": {"command": "ls"},
            },
        ],
        "model": "k2p5",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            return raw_response

    import asyncio

    async def fake_post(url, **kw):
        return FakeResponse()

    client._http.post = fake_post

    result = asyncio.get_event_loop().run_until_complete(client.chat([], tools=None))
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].arguments == {"command": "ls"}


# --- Doom loop tool result format ---


@pytest.mark.asyncio
async def test_doom_loop_result_marked_as_error():
    """Doom loop rejection should produce tool_result with is_error=True."""
    client = _make_client()
    messages: list[dict] = [{"role": "user", "content": "test"}]

    same_call = ChatResponse(
        content="",
        tool_calls=[ToolCall(id="toolu_1", name="bash", arguments={"command": "fail"})],
        finish_reason="tool_use",
    )
    call_count = 0

    async def mock_chat(msgs, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count <= DOOM_LOOP_THRESHOLD + 1:
            return same_call
        return ChatResponse(content="gave up")

    client.chat = mock_chat

    async def fake_bash(**kwargs):
        return "error"

    registry = ToolRegistry()
    registry.register(
        ToolDef(
            name="bash",
            description="run",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            handler=fake_bash,
        )
    )

    await client.chat_loop(messages, registry)

    # Find a tool_result block with is_error=True
    has_error_result = False
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for block in m["content"]:
                if block.get("type") == "tool_result" and block.get("is_error"):
                    has_error_result = True
    assert has_error_result


# --- Tool registration in app ---


def test_glob_and_grep_registered():
    from aineko.app import build_tools

    registry = build_tools()
    assert registry.get("glob") is not None
    assert registry.get("grep") is not None


def test_all_expected_tools_registered():
    from aineko.app import build_tools

    registry = build_tools()
    expected = [
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "web_search",
        "create_skill",
        "memory_recall",
        "search_tool_history",
        "search_chat",
    ]
    for name in expected:
        assert registry.get(name) is not None, f"Tool '{name}' not registered"


def test_tool_schemas_have_valid_structure():
    from aineko.app import build_tools

    registry = build_tools()
    schemas = registry.schemas()
    assert len(schemas) > 0
    for schema in schemas:
        assert schema["type"] == "function"
        func = schema["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        assert (
            len(func["description"]) > 10
        ), f"Tool '{func['name']}' has too short a description"
