"""Tests for doom loop detection in chat_loop."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from aineko.kimi.client import KimiClient, ChatResponse, ToolCall, ToolCallRecord
from aineko.tools.registry import ToolRegistry, ToolDef

DOOM_LOOP_THRESHOLD = 3


def _make_settings(**overrides):
    defaults = {
        "api_key": "test",
        "base_url": "http://localhost",
        "model": "test-model",
        "user_agent": "",
        "max_context_tokens": 8000,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_tool_response(tool_name: str, args: dict, content: str = "") -> ChatResponse:
    return ChatResponse(
        content=content,
        tool_calls=[ToolCall(id=f"call_{tool_name}", name=tool_name, arguments=args)],
        usage={"total_tokens": 100},
        finish_reason="tool_calls",
    )


def _make_final_response(content: str = "done") -> ChatResponse:
    return ChatResponse(
        content=content,
        tool_calls=[],
        usage={"total_tokens": 100},
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_doom_loop_detected_after_3_identical_calls():
    """Same tool + same args 3x in a row should be rejected."""
    client = KimiClient(_make_settings())

    call_count = 0
    responses = []
    # Model keeps calling bash with same args 3 times, then gives final answer
    same_call = _make_tool_response("bash", {"command": "failing_cmd"})
    for _ in range(DOOM_LOOP_THRESHOLD + 1):
        responses.append(same_call)
    responses.append(_make_final_response("gave up"))

    response_iter = iter(responses)

    async def fake_chat(messages, tools=None):
        return next(response_iter)

    client.chat = fake_chat

    tool_results = []

    async def fake_bash(command: str, **kwargs) -> str:
        tool_results.append(command)
        return "error: command failed"

    registry = ToolRegistry()
    registry.register(
        ToolDef(
            name="bash",
            description="run command",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            handler=fake_bash,
        )
    )

    result = await client.chat_loop([{"role": "user", "content": "test"}], registry)

    # The tool should have been called fewer than DOOM_LOOP_THRESHOLD + 1 times
    # because the 3rd identical call gets rejected
    assert len(tool_results) <= DOOM_LOOP_THRESHOLD


@pytest.mark.asyncio
async def test_different_args_dont_trigger_doom_loop():
    """Different args on same tool should NOT trigger doom loop."""
    client = KimiClient(_make_settings())

    responses = [
        _make_tool_response("bash", {"command": "cmd1"}),
        _make_tool_response("bash", {"command": "cmd2"}),
        _make_tool_response("bash", {"command": "cmd3"}),
        _make_final_response("done"),
    ]
    response_iter = iter(responses)

    async def fake_chat(messages, tools=None):
        return next(response_iter)

    client.chat = fake_chat

    tool_results = []

    async def fake_bash(command: str, **kwargs) -> str:
        tool_results.append(command)
        return "ok"

    registry = ToolRegistry()
    registry.register(
        ToolDef(
            name="bash",
            description="run command",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            handler=fake_bash,
        )
    )

    result = await client.chat_loop([{"role": "user", "content": "test"}], registry)
    # All 3 should have executed since args differ
    assert len(tool_results) == 3


@pytest.mark.asyncio
async def test_different_tools_dont_trigger_doom_loop():
    """Different tools with same args should NOT trigger doom loop."""
    client = KimiClient(_make_settings())

    responses = [
        _make_tool_response("bash", {"command": "x"}),
        _make_tool_response("read_file", {"command": "x"}),
        _make_tool_response("bash", {"command": "x"}),
        _make_final_response("done"),
    ]
    response_iter = iter(responses)

    async def fake_chat(messages, tools=None):
        try:
            return next(response_iter)
        except StopIteration:
            return _make_final_response("fallback")

    client.chat = fake_chat

    call_log = []

    async def fake_handler(**kwargs) -> str:
        call_log.append(kwargs)
        return "ok"

    registry = ToolRegistry()
    for name in ("bash", "read_file"):
        registry.register(
            ToolDef(
                name=name,
                description="test",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                handler=fake_handler,
            )
        )

    result = await client.chat_loop([{"role": "user", "content": "test"}], registry)
    assert len(call_log) == 3
