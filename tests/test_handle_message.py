"""Integration test for handle_message — catches wiring bugs between app.py and its dependencies."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from aineko.app import handle_message
from aineko.kimi.client import ChatResponse
from aineko.schemas.message import IncomingMessage
from aineko.tools.registry import ToolRegistry


@pytest_asyncio.fixture
async def setup_db(pg_db):
    yield


@pytest.fixture
def msg():
    return IncomingMessage(
        room_id="!room:test",
        sender="@user:test",
        body="hello",
        event_id="$evt1",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def kimi():
    k = AsyncMock()
    k.chat_loop.return_value = ChatResponse(
        content="hi there",
        usage={"total_tokens": 100},
    )
    return k


@pytest.fixture
def matrix():
    return AsyncMock()


@pytest.fixture
def skills():
    s = MagicMock()
    s.summaries.return_value = []
    return s


@pytest.fixture
def soul_path(tmp_path):
    p = tmp_path / "soul.md"
    p.write_text("You are aineko.")
    return p


@pytest.fixture
def memory_dir(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    return d


class TestHandleMessageIntegration:
    @pytest.mark.asyncio
    async def test_basic_message_flow(
        self, setup_db, msg, kimi, matrix, skills, soul_path, memory_dir
    ):
        """handle_message runs end-to-end without errors."""
        tools = ToolRegistry()

        await handle_message(
            msg,
            kimi,
            tools,
            skills,
            matrix,
            soul_path,
            memory_dir,
            max_context_tokens=100_000,
        )

        kimi.chat_loop.assert_awaited_once()
        matrix.send_message.assert_awaited_once_with("!room:test", "hi there")

    @pytest.mark.asyncio
    async def test_reset_command(
        self, setup_db, kimi, matrix, skills, soul_path, memory_dir
    ):
        """The /reset command clears conversation and replies."""
        tools = ToolRegistry()
        reset_msg = IncomingMessage(
            room_id="!room:test",
            sender="@user:test",
            body="/reset",
            event_id="$evt2",
            timestamp=datetime.now(timezone.utc),
        )

        await handle_message(
            reset_msg,
            kimi,
            tools,
            skills,
            matrix,
            soul_path,
            memory_dir,
            max_context_tokens=100_000,
        )

        kimi.chat_loop.assert_not_awaited()
        matrix.send_message.assert_awaited_once()
        assert "cleared" in matrix.send_message.call_args[0][1]

    @pytest.mark.asyncio
    async def test_response_persisted_across_calls(
        self, setup_db, msg, kimi, matrix, skills, soul_path, memory_dir
    ):
        """Second call includes history from the first."""
        tools = ToolRegistry()

        await handle_message(
            msg,
            kimi,
            tools,
            skills,
            matrix,
            soul_path,
            memory_dir,
            max_context_tokens=100_000,
        )

        # Send a second message
        msg2 = IncomingMessage(
            room_id="!room:test",
            sender="@user:test",
            body="follow up",
            event_id="$evt3",
            timestamp=datetime.now(timezone.utc),
        )
        await handle_message(
            msg2,
            kimi,
            tools,
            skills,
            matrix,
            soul_path,
            memory_dir,
            max_context_tokens=100_000,
        )

        # Second call should have more messages (history from first call)
        second_call_messages = kimi.chat_loop.call_args_list[1][0][0]
        # system + user("hello") + assistant("hi there") + user("follow up")
        assert len(second_call_messages) >= 4
