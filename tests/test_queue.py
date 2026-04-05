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


def _msg(body: str, room: str = "!room:test") -> IncomingMessage:
    return IncomingMessage(
        room_id=room,
        sender="@user:test",
        body=body,
        timestamp=datetime.now(timezone.utc),
        event_id=f"evt_{body}",
    )


@pytest.mark.asyncio
async def test_single_message_delivered():
    """A single message is delivered after debounce."""
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        received.append(msg)

    q = MessageQueue(handler)
    await q.enqueue(_msg("hello"))
    await asyncio.sleep(2)  # wait for debounce

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
    await asyncio.sleep(0.2)
    await q.enqueue(_msg("second"))
    await asyncio.sleep(0.2)
    await q.enqueue(_msg("third"))
    await asyncio.sleep(2)  # wait for debounce

    assert len(received) == 1
    assert "[message 1]: first" in received[0].body
    assert "[message 2]: second" in received[0].body
    assert "[message 3]: third" in received[0].body


@pytest.mark.asyncio
async def test_messages_during_processing_queued():
    """Messages arriving while handler is busy are queued and processed after."""
    received: list[IncomingMessage] = []

    async def slow_handler(msg: IncomingMessage) -> None:
        received.append(msg)
        await asyncio.sleep(1)  # simulate slow processing

    q = MessageQueue(slow_handler)
    await q.enqueue(_msg("first"))
    await asyncio.sleep(2)  # debounce fires, handler starts (takes 1s)

    # Send another while handler is busy
    await asyncio.sleep(0.2)
    await q.enqueue(_msg("while busy"))
    await asyncio.sleep(3)  # wait for everything to finish

    assert len(received) == 2
    assert received[0].body == "first"
    assert received[1].body == "while busy"


@pytest.mark.asyncio
async def test_different_rooms_independent():
    """Messages in different rooms are processed independently."""
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        received.append(msg)

    q = MessageQueue(handler)
    await q.enqueue(_msg("room1 msg", room="!room1:test"))
    await q.enqueue(_msg("room2 msg", room="!room2:test"))
    await asyncio.sleep(2)

    assert len(received) == 2
    rooms = {r.room_id for r in received}
    assert rooms == {"!room1:test", "!room2:test"}


# ---------------------------------------------------------------------------
# Helpers for property tests
# ---------------------------------------------------------------------------

_FAST_DEBOUNCE = 30  # ms — patched into queue module during property tests

# Strategies — keep durations small so 20 examples finish in ~30s total.
# handler_ms must be > 0 so the handler actually awaits (where cancellation strikes).
_handler_ms = st.integers(min_value=10, max_value=120)
_delay_ms = st.integers(min_value=0, max_value=100)


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


# ---------------------------------------------------------------------------
# Property tests
#
# Key modelling decision: the handler records the message AFTER an async
# operation (sleep), matching the real handler where the LLM response +
# persistence happen after the long await.  If the handler is cancelled
# mid-await, the message is lost — exactly the bug we're testing for.
# ---------------------------------------------------------------------------


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

        with patch("aineko.queue.DEBOUNCE_MS", _FAST_DEBOUNCE):
            q = MessageQueue(handler)
            expected: set[str] = set()
            for i, delay in enumerate(delays):
                body = f"msg_{i}"
                expected.add(body)
                await q.enqueue(_msg(body))
                await asyncio.sleep(delay / 1000)

            # Generous wait: debounce + each message could be handled serially
            budget = (_FAST_DEBOUNCE + handler_ms) * len(delays) + 500
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

        with patch("aineko.queue.DEBOUNCE_MS", _FAST_DEBOUNCE):
            q = MessageQueue(handler)
            for i, delay in enumerate(delays):
                await q.enqueue(_msg(f"msg_{i}"))
                await asyncio.sleep(delay / 1000)

            budget = (_FAST_DEBOUNCE + handler_ms) * len(delays) + 500
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

        with patch("aineko.queue.DEBOUNCE_MS", _FAST_DEBOUNCE):
            q = MessageQueue(handler)
            order: list[str] = []
            for i, delay in enumerate(delays):
                body = f"msg_{i}"
                order.append(body)
                await q.enqueue(_msg(body))
                await asyncio.sleep(delay / 1000)

            budget = (_FAST_DEBOUNCE + handler_ms) * len(delays) + 500
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
        await asyncio.sleep(0.3)  # simulate LLM call — cancellation hits HERE
        received.append(msg)  # simulate persistence — only runs if not cancelled

    with patch("aineko.queue.DEBOUNCE_MS", 30):
        q = MessageQueue(slow_handler)

        await q.enqueue(_msg("first"))
        await asyncio.sleep(0.05)  # debounce fires at 30ms
        await handler_entered.wait()

        # Send two more while handler is busy — triggers the old cancel bug
        await q.enqueue(_msg("second"))
        await asyncio.sleep(0.05)
        await q.enqueue(_msg("third"))

        await asyncio.sleep(1.5)

    assert _extract_bodies(received) == {"first", "second", "third"}
