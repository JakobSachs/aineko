"""Tests for tool history logging and search."""

import pytest
from datetime import datetime

from aineko.kimi.client import ToolCallRecord


def test_tool_call_record_fields():
    rec = ToolCallRecord(
        tool_name="bash",
        arguments='{"command": "ls"}',
        result="file1.txt\nfile2.txt",
    )
    assert rec.tool_name == "bash"
    assert rec.arguments == '{"command": "ls"}'
    assert rec.result == "file1.txt\nfile2.txt"


def test_tool_call_record_in_list():
    """Tool history is accumulated as a list of records."""
    history: list[ToolCallRecord] = []
    history.append(ToolCallRecord("bash", '{"command": "ls"}', "file1.txt"))
    history.append(ToolCallRecord("read_file", '{"path": "/data/x.md"}', "contents"))
    assert len(history) == 2
    assert history[0].tool_name == "bash"
    assert history[1].tool_name == "read_file"


@pytest.mark.asyncio
async def test_chat_loop_returns_tool_history():
    """chat_loop should accumulate tool call records and return them."""
    from unittest.mock import AsyncMock, MagicMock
    from aineko.kimi.client import ChatResponse, KimiClient, ToolCall
    from aineko.tools.registry import ToolRegistry

    # Set up a KimiClient with mocked HTTP
    settings = MagicMock()
    settings.api_key = "test"
    settings.base_url = "http://test"
    settings.user_agent = ""
    settings.model = "test-model"
    client = KimiClient(settings)

    # Mock chat to return one tool call round, then a final response
    call_count = 0

    async def mock_chat(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ChatResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="bash", arguments={"command": "ls"})],
            )
        return ChatResponse(content="done", tool_calls=[])

    client.chat = mock_chat

    # Set up a tool registry with a bash tool
    registry = ToolRegistry()
    from aineko.tools.registry import ToolDef

    async def fake_bash(command: str = "") -> str:
        return "file1.txt"

    registry.register(ToolDef(
        name="bash",
        description="run bash",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        handler=fake_bash,
    ))

    response = await client.chat_loop([], registry)

    assert response.content == "done"
    assert len(response.tool_history) == 1
    assert response.tool_history[0].tool_name == "bash"
    assert response.tool_history[0].result == "file1.txt"


@pytest.mark.asyncio
async def test_chat_loop_multi_round_history():
    """Multiple rounds of tool calls all appear in history."""
    from unittest.mock import MagicMock
    from aineko.kimi.client import ChatResponse, KimiClient, ToolCall
    from aineko.tools.registry import ToolDef, ToolRegistry

    settings = MagicMock()
    settings.api_key = "test"
    settings.base_url = "http://test"
    settings.user_agent = ""
    settings.model = "test-model"
    client = KimiClient(settings)

    call_count = 0

    async def mock_chat(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ChatResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="bash", arguments={"command": "ls"})],
            )
        if call_count == 2:
            return ChatResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="read_file", arguments={"path": "/data/x"})],
            )
        return ChatResponse(content="all done")

    client.chat = mock_chat

    registry = ToolRegistry()

    async def fake_bash(command: str = "") -> str:
        return "output1"

    async def fake_read(path: str = "") -> str:
        return "file contents"

    registry.register(ToolDef(name="bash", description="", parameters={"type": "object", "properties": {}, "required": []}, handler=fake_bash))
    registry.register(ToolDef(name="read_file", description="", parameters={"type": "object", "properties": {}, "required": []}, handler=fake_read))

    response = await client.chat_loop([], registry)

    assert len(response.tool_history) == 2
    assert response.tool_history[0].tool_name == "bash"
    assert response.tool_history[1].tool_name == "read_file"
    assert response.tool_history[0].result == "output1"
    assert response.tool_history[1].result == "file contents"
