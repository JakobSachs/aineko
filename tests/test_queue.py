"""Tests for message queue with debounce and batching."""

import asyncio
from datetime import datetime, timezone

import pytest

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
