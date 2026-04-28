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


def _img_msg(body: str, room: str = "!room:test") -> IncomingMessage:
    return IncomingMessage(
        room_id=room,
        sender="@user:test",
        body=body,
        timestamp=datetime.now(timezone.utc),
        event_id=f"evt_{body}",
        image_b64="aGVsbG8=",  # any non-None marker
        image_mime="image/png",
    )


@pytest.fixture(autouse=True)
def fast_debounce():
    with patch("aineko.queue.DEBOUNCE_MS", _TEST_DEBOUNCE_MS):
        yield


@pytest.mark.asyncio
async def test_single_message_delivered():
    """A single message is delivered after debounce."""
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage, interject: asyncio.Queue) -> None:
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

    async def handler(msg: IncomingMessage, interject: asyncio.Queue) -> None:
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
async def test_text_messages_during_processing_interject():
    """Text messages arriving while handler is busy are interjected into the
    handler's interject queue, not re-queued as a separate turn."""
    received_main: list[IncomingMessage] = []
    received_interjected: list[str] = []
    handler_entered = asyncio.Event()

    async def slow_handler(msg: IncomingMessage, interject: asyncio.Queue) -> None:
        handler_entered.set()
        received_main.append(msg)
        await asyncio.sleep(0.1)  # simulate processing
        while not interject.empty():
            received_interjected.append(interject.get_nowait())

    q = MessageQueue(slow_handler)
    await q.enqueue(_msg("first"))
    await asyncio.sleep(0.03)  # debounce fires
    await handler_entered.wait()

    # Send another while handler is busy — should interject, not re-queue.
    await q.enqueue(_msg("while busy"))
    await asyncio.sleep(0.3)  # wait for everything

    assert len(received_main) == 1
    assert received_main[0].body == "first"
    assert received_interjected == ["while busy"]


@pytest.mark.asyncio
async def test_messages_share_single_queue():
    """Single-room deployment: all enqueued messages batch into one handler call
    regardless of msg.room_id — there is no per-room partitioning anymore."""
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage, interject: asyncio.Queue) -> None:
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
        """Every enqueued body surfaces either in a handler call or via interjection."""
        received: list[IncomingMessage] = []
        interjected: list[str] = []

        async def handler(msg: IncomingMessage, interject: asyncio.Queue) -> None:
            await asyncio.sleep(handler_ms / 1000)
            received.append(msg)
            while not interject.empty():
                interjected.append(interject.get_nowait())

        q = MessageQueue(handler)
        expected: set[str] = set()
        for i, delay in enumerate(delays):
            body = f"msg_{i}"
            expected.add(body)
            await q.enqueue(_msg(body))
            await asyncio.sleep(delay / 1000)

        budget = (_TEST_DEBOUNCE_MS + handler_ms) * len(delays) + 200
        await asyncio.sleep(budget / 1000)

        assert _extract_bodies(received) | set(interjected) == expected

    @given(
        handler_ms=_handler_ms,
        delays=st.lists(_delay_ms, min_size=2, max_size=5),
    )
    @hsettings(max_examples=20, deadline=None)
    @pytest.mark.asyncio
    async def test_no_concurrent_handlers(self, handler_ms: int, delays: list[int]):
        """Handler never runs concurrently."""
        max_concurrent = 0
        concurrent = 0

        async def handler(msg: IncomingMessage, interject: asyncio.Queue) -> None:
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
        """Messages (across handler calls + interjections) surface in arrival order."""
        flat: list[str] = []

        async def handler(msg: IncomingMessage, interject: asyncio.Queue) -> None:
            matches = re.findall(r"\[message \d+\]: (.+)", msg.body)
            if matches:
                flat.extend(matches)
            else:
                flat.append(msg.body)
            await asyncio.sleep(handler_ms / 1000)
            while not interject.empty():
                flat.append(interject.get_nowait())

        q = MessageQueue(handler)
        order: list[str] = []
        for i, delay in enumerate(delays):
            body = f"msg_{i}"
            order.append(body)
            await q.enqueue(_msg(body))
            await asyncio.sleep(delay / 1000)

        budget = (_TEST_DEBOUNCE_MS + handler_ms) * len(delays) + 200
        await asyncio.sleep(budget / 1000)

        assert flat == order

    @given(
        handler_ms=_handler_ms,
        kinds=st.lists(st.sampled_from(["text", "image"]), min_size=2, max_size=6),
        delays=st.lists(_delay_ms, min_size=2, max_size=6),
    )
    @hsettings(max_examples=20, deadline=None)
    @pytest.mark.asyncio
    async def test_text_interjects_image_requeues(
        self, handler_ms: int, kinds: list[str], delays: list[int]
    ):
        """During a slow handler, text messages must interject into the live
        turn while image messages must be re-queued for a fresh handler call
        (because vision input needs full message context, not just a string)."""
        n = min(len(kinds), len(delays))
        kinds = kinds[:n]
        delays = delays[:n]

        received_main: list[IncomingMessage] = []
        received_interjected: list[str] = []

        async def handler(msg: IncomingMessage, interject: asyncio.Queue) -> None:
            received_main.append(msg)
            await asyncio.sleep(handler_ms / 1000)
            while not interject.empty():
                received_interjected.append(interject.get_nowait())

        q = MessageQueue(handler)
        text_bodies: list[str] = []
        image_bodies: list[str] = []
        for i, (kind, delay) in enumerate(zip(kinds, delays)):
            body = f"{kind}_{i}"
            if kind == "text":
                text_bodies.append(body)
                await q.enqueue(_msg(body))
            else:
                image_bodies.append(body)
                await q.enqueue(_img_msg(body))
            await asyncio.sleep(delay / 1000)

        budget = (_TEST_DEBOUNCE_MS + handler_ms) * len(kinds) + 300
        await asyncio.sleep(budget / 1000)

        # Every body must surface somewhere — no message lost.
        seen = _extract_bodies(received_main) | set(received_interjected)
        assert seen == set(text_bodies) | set(image_bodies)

        # Every image arrived as a (re-)queued main handler call, never as
        # an interjection — vision input must trigger a fresh turn.
        for body in image_bodies:
            assert (
                body not in received_interjected
            ), f"image {body} was interjected; it should have been re-queued"


# ---------------------------------------------------------------------------
# Regression: the exact scenario that caused the bug
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_images_in_batch_all_reach_handler():
    """BUG: when N>1 messages debounce-batch and at least two carry image_b64,
    `_drain` does `batch[-1].model_copy(update={"body": combined_body})` —
    only the LAST message's image_b64 survives. Every earlier image is
    discarded. The model sees the captions but only one of the images.

    Send two image messages within the debounce window; the handler must
    receive both image payloads (e.g. by getting two separate calls, or by
    receiving a message that carries every image). Today it gets one call
    with only image B's bytes, image A is lost."""
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage, interject: asyncio.Queue) -> None:
        received.append(msg)

    q = MessageQueue(handler)
    img_a = _img_msg("first_image")
    img_a_b64 = "AAAA_image_a_bytes"
    img_a = img_a.model_copy(update={"image_b64": img_a_b64})
    img_b = _img_msg("second_image")
    img_b_b64 = "BBBB_image_b_bytes"
    img_b = img_b.model_copy(update={"image_b64": img_b_b64})

    await q.enqueue(img_a)
    await asyncio.sleep(0.005)  # within debounce window
    await q.enqueue(img_b)
    await asyncio.sleep(0.1)

    seen_b64s: set[str] = set()
    for m in received:
        if m.image_b64:
            seen_b64s.add(m.image_b64)

    assert img_a_b64 in seen_b64s, (
        f"image A's bytes were dropped during batching — handler only saw "
        f"{seen_b64s}. The model has no way to look at image A."
    )
    assert img_b_b64 in seen_b64s, "image B's bytes missing too"


@pytest.mark.asyncio
async def test_message_during_slow_handler_not_lost():
    """Messages arriving while a slow handler runs must not cancel it, and
    must surface to the handler — either as interjections (text) or a
    follow-up drain (images, never interjected).

    Regression for: enqueue() cancelled the debounce task while it was already
    past the sleep phase and running _drain → the in-flight LLM request was killed.
    """
    seen: list[str] = []
    handler_entered = asyncio.Event()

    async def slow_handler(msg: IncomingMessage, interject: asyncio.Queue) -> None:
        handler_entered.set()
        await asyncio.sleep(0.1)  # simulate LLM call
        matches = re.findall(r"\[message \d+\]: (.+)", msg.body)
        seen.extend(matches if matches else [msg.body])
        while not interject.empty():
            seen.append(interject.get_nowait())

    q = MessageQueue(slow_handler)

    await q.enqueue(_msg("first"))
    await asyncio.sleep(0.03)  # debounce fires
    await handler_entered.wait()

    # Send two more while handler is busy — these interject into the live turn.
    await q.enqueue(_msg("second"))
    await asyncio.sleep(0.01)
    await q.enqueue(_msg("third"))

    await asyncio.sleep(0.5)

    assert set(seen) == {"first", "second", "third"}
