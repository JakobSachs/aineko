"""Tests for message queue with debounce and batching."""

import asyncio
import re
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from hypothesis import given, settings as hsettings
from hypothesis import strategies as st

from aineko.queue import MessageQueue
from aineko.schemas.message import IncomingMessage

# Use a tiny debounce for all tests so we don't wait around
_TEST_DEBOUNCE_MS = 20


def _msg(body: str, room: str = "!room:test") -> IncomingMessage:
    return IncomingMessage(
        room_id=room,
        sender="@user:test",
        body=body,
        timestamp=datetime.now(timezone.utc),
        event_id=f"evt_{body}",
    )


@pytest.fixture(autouse=True)
def fast_debounce():
    with patch("aineko.queue.DEBOUNCE_MS", _TEST_DEBOUNCE_MS):
        yield


@pytest.mark.asyncio
async def test_single_message_delivered():
    """A single message is delivered after debounce."""
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        received.append(msg)

    q = MessageQueue(handler)
    await q.enqueue(_msg("hello"))
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].body == "hello"


@pytest.mark.asyncio
async def test_rapid_messages_batched():
    """Multiple messages sent quickly are combined into one."""
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        received.append(msg)

    q = MessageQueue(handler)
    await q.enqueue(_msg("first"))
    await asyncio.sleep(0.005)  # within debounce window
    await q.enqueue(_msg("second"))
    await asyncio.sleep(0.005)
    await q.enqueue(_msg("third"))
    await asyncio.sleep(0.05)  # debounce fires

    assert len(received) == 1
    assert "[message 1]: first" in received[0].body
    assert "[message 2]: second" in received[0].body
    assert "[message 3]: third" in received[0].body


@pytest.mark.asyncio
async def test_messages_during_processing_queued():
    """Messages arriving while handler is busy are queued and processed after."""
    received: list[IncomingMessage] = []
    handler_entered = asyncio.Event()

    async def slow_handler(msg: IncomingMessage) -> None:
        handler_entered.set()
        await asyncio.sleep(0.1)  # simulate processing
        received.append(msg)

    q = MessageQueue(slow_handler)
    await q.enqueue(_msg("first"))
    await asyncio.sleep(0.03)  # debounce fires
    await handler_entered.wait()

    # Send another while handler is busy
    await q.enqueue(_msg("while busy"))
    await asyncio.sleep(0.3)  # wait for everything

    assert len(received) == 2
    assert received[0].body == "first"
    assert received[1].body == "while busy"


@pytest.mark.asyncio
async def test_messages_share_single_queue():
    """Single-room deployment: all enqueued messages batch into one handler call
    regardless of msg.room_id — there is no per-room partitioning anymore."""
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        received.append(msg)

    q = MessageQueue(handler)
    await q.enqueue(_msg("a", room="!room1:test"))
    await q.enqueue(_msg("b", room="!room2:test"))
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert "[message 1]: a" in received[0].body
    assert "[message 2]: b" in received[0].body


# ---------------------------------------------------------------------------
# Helpers for property tests
# ---------------------------------------------------------------------------


def _extract_bodies(received: list[IncomingMessage]) -> set[str]:
    """Extract all original message bodies from handler calls (single or batched)."""
    bodies: set[str] = set()
    for msg in received:
        matches = re.findall(r"\[message \d+\]: (.+)", msg.body)
        if matches:
            bodies.update(matches)
        else:
            bodies.add(msg.body)
    return bodies


def _extract_bodies_ordered(received: list[IncomingMessage]) -> list[str]:
    """Extract all original message bodies preserving order."""
    flat: list[str] = []
    for msg in received:
        matches = re.findall(r"\[message \d+\]: (.+)", msg.body)
        if matches:
            flat.extend(matches)
        else:
            flat.append(msg.body)
    return flat


# Strategies — keep durations small
_handler_ms = st.integers(min_value=5, max_value=50)
_delay_ms = st.integers(min_value=0, max_value=30)


class TestQueueProperties:
    @given(
        handler_ms=_handler_ms,
        delays=st.lists(_delay_ms, min_size=2, max_size=5),
    )
    @hsettings(max_examples=20, deadline=None)
    @pytest.mark.asyncio
    async def test_no_messages_lost(self, handler_ms: int, delays: list[int]):
        """Every enqueued body appears in exactly one handler call."""
        received: list[IncomingMessage] = []

        async def handler(msg: IncomingMessage) -> None:
            await asyncio.sleep(handler_ms / 1000)
            received.append(msg)

        q = MessageQueue(handler)
        expected: set[str] = set()
        for i, delay in enumerate(delays):
            body = f"msg_{i}"
            expected.add(body)
            await q.enqueue(_msg(body))
            await asyncio.sleep(delay / 1000)

        budget = (_TEST_DEBOUNCE_MS + handler_ms) * len(delays) + 200
        await asyncio.sleep(budget / 1000)

        assert _extract_bodies(received) == expected

    @given(
        handler_ms=_handler_ms,
        delays=st.lists(_delay_ms, min_size=2, max_size=5),
    )
    @hsettings(max_examples=20, deadline=None)
    @pytest.mark.asyncio
    async def test_no_concurrent_handlers(self, handler_ms: int, delays: list[int]):
        """Handler never runs concurrently for the same room."""
        max_concurrent = 0
        concurrent = 0

        async def handler(msg: IncomingMessage) -> None:
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(handler_ms / 1000)
            concurrent -= 1

        q = MessageQueue(handler)
        for i, delay in enumerate(delays):
            await q.enqueue(_msg(f"msg_{i}"))
            await asyncio.sleep(delay / 1000)

        budget = (_TEST_DEBOUNCE_MS + handler_ms) * len(delays) + 200
        await asyncio.sleep(budget / 1000)

        assert max_concurrent <= 1

    @given(
        handler_ms=_handler_ms,
        delays=st.lists(_delay_ms, min_size=2, max_size=5),
    )
    @hsettings(max_examples=20, deadline=None)
    @pytest.mark.asyncio
    async def test_arrival_order_preserved(self, handler_ms: int, delays: list[int]):
        """Within each handler call, messages appear in arrival order."""
        received: list[IncomingMessage] = []

        async def handler(msg: IncomingMessage) -> None:
            await asyncio.sleep(handler_ms / 1000)
            received.append(msg)

        q = MessageQueue(handler)
        order: list[str] = []
        for i, delay in enumerate(delays):
            body = f"msg_{i}"
            order.append(body)
            await q.enqueue(_msg(body))
            await asyncio.sleep(delay / 1000)

        budget = (_TEST_DEBOUNCE_MS + handler_ms) * len(delays) + 200
        await asyncio.sleep(budget / 1000)

        assert _extract_bodies_ordered(received) == order


# ---------------------------------------------------------------------------
# Regression: the exact scenario that caused the bug
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_message_during_slow_handler_not_lost():
    """Message arriving while a slow handler runs must not cancel the handler.

    Regression for: enqueue() cancelled the debounce task while it was already
    past the sleep phase and running _drain → the in-flight LLM request was killed.
    """
    received: list[IncomingMessage] = []
    handler_entered = asyncio.Event()

    async def slow_handler(msg: IncomingMessage) -> None:
        handler_entered.set()
        await asyncio.sleep(0.1)  # simulate LLM call
        received.append(msg)

    q = MessageQueue(slow_handler)

    await q.enqueue(_msg("first"))
    await asyncio.sleep(0.03)  # debounce fires
    await handler_entered.wait()

    # Send two more while handler is busy
    await q.enqueue(_msg("second"))
    await asyncio.sleep(0.01)
    await q.enqueue(_msg("third"))

    await asyncio.sleep(0.5)

    assert _extract_bodies(received) == {"first", "second", "third"}
