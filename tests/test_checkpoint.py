"""Tests for checkpoint-based tool loop (progress updates instead of hard stop)."""

import pytest
from unittest.mock import MagicMock

from aineko.kimi.client import ChatResponse, KimiClient, ToolCall
from aineko.tools.registry import ToolDef, ToolRegistry


def _make_client():
    settings = MagicMock()
    settings.api_key = "test"
    settings.base_url = "http://test"
    settings.user_agent = ""
    settings.model = "test-model"
    return KimiClient(settings)


def _make_registry():
    registry = ToolRegistry()

    sent: list[str] = []

    async def fake_bash(**kwargs):
        return "ok"

    async def fake_send(message: str = "", **kwargs):
        sent.append(message)
        return "sent"

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
    registry.register(
        ToolDef(
            name="send_message",
            description="send",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            handler=fake_send,
        )
    )
    return registry, sent


@pytest.mark.asyncio
async def test_checkpoint_forces_progress_update():
    """At checkpoint interval, model is forced to give a progress update, then continues."""
    client = _make_client()
    registry, sent = _make_registry()

    call_count = 0

    async def mock_chat(msgs, tools=None):
        nonlocal call_count
        call_count += 1

        # Check if we're being asked for a progress update (no tools passed)
        if tools is None:
            return ChatResponse(content="halfway done, continuing...")

        # First 4 rounds: tool calls
        if call_count <= 4:
            return ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"tc{call_count}",
                        name="bash",
                        arguments={"command": f"step{call_count}"},
                    )
                ],
            )
        # After checkpoint: 2 more tool calls then done
        if call_count <= 7:
            return ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"tc{call_count}",
                        name="bash",
                        arguments={"command": f"step{call_count}"},
                    )
                ],
            )
        return ChatResponse(content="all done")

    client.chat = mock_chat

    result = await client.chat_loop(
        [{"role": "user", "content": "do stuff"}],
        registry,
        checkpoint_every=4,
    )

    # Should have completed (not been cut off)
    assert result.content == "all done"
    # Should have more than 4 tool calls (continued after checkpoint)
    assert len(result.tool_history) > 4


@pytest.mark.asyncio
async def test_checkpoint_sends_update_via_send_message():
    """The checkpoint progress update should be sent to the user."""
    client = _make_client()
    registry, sent = _make_registry()

    call_count = 0
    checkpoint_hit = False

    async def mock_chat(msgs, tools=None):
        nonlocal call_count, checkpoint_hit
        call_count += 1

        if tools is None:
            checkpoint_hit = True
            return ChatResponse(content="working on it...")

        if call_count <= 3:
            return ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"tc{call_count}",
                        name="bash",
                        arguments={"command": f"s{call_count}"},
                    )
                ],
            )
        return ChatResponse(content="finished")

    client.chat = mock_chat

    await client.chat_loop(
        [{"role": "user", "content": "go"}],
        registry,
        checkpoint_every=3,
    )

    assert checkpoint_hit
    # The progress message should have been sent via send_message
    assert any("working on it" in m for m in sent)


@pytest.mark.asyncio
async def test_hard_ceiling_still_stops():
    """Even with checkpoints, there's a hard ceiling to prevent runaway."""
    client = _make_client()
    registry, sent = _make_registry()

    async def mock_chat(msgs, tools=None):
        if tools is None:
            return ChatResponse(content="still going...")
        # Always return tool calls — never stops voluntarily
        return ChatResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="bash", arguments={"command": "loop"})],
        )

    client.chat = mock_chat

    result = await client.chat_loop(
        [{"role": "user", "content": "go"}],
        registry,
        checkpoint_every=5,
        max_rounds=12,
    )

    # Should have stopped eventually
    assert result is not None
    # Should not have run forever — tool calls capped
    assert len(result.tool_history) <= 12


@pytest.mark.asyncio
async def test_no_checkpoint_if_model_stops_early():
    """If model finishes before checkpoint, no forced update happens."""
    client = _make_client()
    registry, sent = _make_registry()

    call_count = 0

    async def mock_chat(msgs, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="bash", arguments={"command": "quick"})
                ],
            )
        return ChatResponse(content="done quick")

    client.chat = mock_chat

    result = await client.chat_loop(
        [{"role": "user", "content": "go"}],
        registry,
        checkpoint_every=10,
    )

    assert result.content == "done quick"
    assert len(result.tool_history) == 1
    # No progress updates sent
    assert len(sent) == 0


@pytest.mark.asyncio
async def test_multiple_checkpoints():
    """Loop should checkpoint multiple times if work continues."""
    client = _make_client()
    registry, sent = _make_registry()

    call_count = 0
    checkpoint_count = 0

    async def mock_chat(msgs, tools=None):
        nonlocal call_count, checkpoint_count
        call_count += 1

        if tools is None:
            checkpoint_count += 1
            return ChatResponse(content=f"checkpoint {checkpoint_count}")

        # 8 tool call rounds — with checkpoint_every=3, should hit ~2 checkpoints
        if call_count <= 10:
            return ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"tc{call_count}",
                        name="bash",
                        arguments={"command": f"s{call_count}"},
                    )
                ],
            )
        return ChatResponse(content="finally done")

    client.chat = mock_chat

    result = await client.chat_loop(
        [{"role": "user", "content": "go"}],
        registry,
        checkpoint_every=3,
    )

    assert result.content == "finally done"
    assert checkpoint_count >= 2
