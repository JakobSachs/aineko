"""Tests for the heartbeat tick loop extracted from app.py lifespan."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aineko.heartbeat.loop import heartbeat_tick_loop
from aineko.tools.registry import ToolRegistry


@pytest.fixture
def heartbeat():
    hb = MagicMock()
    hb.should_run.return_value = True
    hb.get_tasks.return_value = "- [ ] check logs"
    hb.should_deliver.return_value = True
    return hb


@pytest.fixture
def kimi():
    k = AsyncMock()
    resp = MagicMock()
    resp.content = "All clear."
    k.chat_loop.return_value = resp
    return k


@pytest.fixture
def tools():
    return ToolRegistry()


@pytest.fixture
def matrix():
    return AsyncMock()


async def _run_loop(coro, duration=0.05):
    """Run the heartbeat loop briefly then cancel it."""
    task = asyncio.create_task(coro)
    await asyncio.sleep(duration)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


class TestHeartbeatTickLoop:
    @pytest.mark.asyncio
    async def test_single_tick_sends_response(self, heartbeat, kimi, tools, matrix):
        """A tick where should_run=True and should_deliver=True sends to matrix."""
        # Run once then block forever
        heartbeat.should_run.side_effect = [True] + [False] * 100

        await _run_loop(
            heartbeat_tick_loop(
                heartbeat, kimi, tools, matrix, "!room:x", "sys", interval=0
            )
        )

        kimi.chat_loop.assert_awaited_once()
        matrix.send_message.assert_awaited_once_with("!room:x", "All clear.")

    @pytest.mark.asyncio
    async def test_skips_when_should_run_false(self, heartbeat, kimi, tools, matrix):
        heartbeat.should_run.return_value = False

        await _run_loop(
            heartbeat_tick_loop(
                heartbeat, kimi, tools, matrix, "!room:x", "sys", interval=0
            )
        )

        kimi.chat_loop.assert_not_awaited()
        matrix.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_suppresses_when_should_deliver_false(
        self, heartbeat, kimi, tools, matrix
    ):
        heartbeat.should_run.side_effect = [True] + [False] * 100
        heartbeat.should_deliver.return_value = False

        await _run_loop(
            heartbeat_tick_loop(
                heartbeat, kimi, tools, matrix, "!room:x", "sys", interval=0
            )
        )

        kimi.chat_loop.assert_awaited()
        matrix.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tools_include_send_message(self, heartbeat, kimi, tools, matrix):
        heartbeat.should_run.side_effect = [True] + [False] * 100

        await _run_loop(
            heartbeat_tick_loop(
                heartbeat, kimi, tools, matrix, "!room:x", "sys", interval=0
            )
        )

        call_args = kimi.chat_loop.call_args
        hb_tools = call_args[0][1]
        assert hb_tools.get("send_message") is not None

    @pytest.mark.asyncio
    async def test_exception_does_not_kill_loop(self, heartbeat, kimi, tools, matrix):
        """An exception in one tick doesn't stop the loop."""
        heartbeat.should_run.return_value = True
        kimi.chat_loop.side_effect = RuntimeError("boom")

        await _run_loop(
            heartbeat_tick_loop(
                heartbeat, kimi, tools, matrix, "!room:x", "sys", interval=0
            )
        )

        # Loop survived — should_run was called multiple times
        assert heartbeat.should_run.call_count > 1

    @pytest.mark.asyncio
    async def test_no_send_when_room_empty(self, heartbeat, kimi, tools, matrix):
        heartbeat.should_run.side_effect = [True] + [False] * 100

        await _run_loop(
            heartbeat_tick_loop(heartbeat, kimi, tools, matrix, "", "sys", interval=0)
        )

        matrix.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_send_when_content_empty(self, heartbeat, kimi, tools, matrix):
        heartbeat.should_run.side_effect = [True] + [False] * 100
        resp = MagicMock()
        resp.content = ""
        kimi.chat_loop.return_value = resp

        await _run_loop(
            heartbeat_tick_loop(
                heartbeat, kimi, tools, matrix, "!room:x", "sys", interval=0
            )
        )

        matrix.send_message.assert_not_awaited()
