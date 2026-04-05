"""Message queue with debounce and processing lock.

Collects rapid-fire messages into batches and queues messages
that arrive while the agent is busy processing.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

from aineko.schemas.message import IncomingMessage

logger = logging.getLogger(__name__)

DEBOUNCE_MS = 1500  # wait this long after last message before processing
MAX_QUEUE = 20  # max queued messages per room

MessageHandler = Callable[[IncomingMessage], Coroutine[Any, Any, None]]


class MessageQueue:
    def __init__(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._queues: dict[str, list[IncomingMessage]] = defaultdict(list)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._debounce_tasks: dict[str, asyncio.Task] = {}

    async def enqueue(self, msg: IncomingMessage) -> None:
        """Add a message to the room's queue and schedule processing."""
        room = msg.room_id
        queue = self._queues[room]

        if len(queue) >= MAX_QUEUE:
            logger.warning(
                "queue full, dropping oldest",
                extra={
                    "event": "queue_drop",
                    "room": room,
                },
            )
            queue.pop(0)

        queue.append(msg)
        logger.info(
            "message queued",
            extra={
                "event": "queue_add",
                "room": room,
                "body": msg.body,
                "queue_len": len(queue),
            },
        )

        # Cancel existing debounce timer and start a new one
        if room in self._debounce_tasks:
            self._debounce_tasks[room].cancel()
        self._debounce_tasks[room] = asyncio.create_task(
            self._debounce_then_drain(room)
        )

    async def _debounce_then_drain(self, room: str) -> None:
        """Wait for debounce period, then drain the queue."""
        try:
            await asyncio.sleep(DEBOUNCE_MS / 1000)
        except asyncio.CancelledError:
            return  # new message arrived, debounce reset
        # Remove ourselves from the cancellable debounce map.  Once the sleep
        # phase is over we are an *active handler* — enqueue() must not cancel
        # an in-progress drain (which would kill the running LLM request).
        self._debounce_tasks.pop(room, None)
        await self._drain(room)

    async def _drain(self, room: str) -> None:
        """Process all queued messages for a room."""
        lock = self._locks[room]

        async with lock:
            queue = self._queues[room]
            if not queue:
                return

            # Take all messages from the queue
            batch = list(queue)
            queue.clear()

            if len(batch) == 1:
                # Single message, pass directly
                msg = batch[0]
            else:
                # Combine multiple messages into one, clearly separated
                parts = [f"[message {i+1}]: {m.body}" for i, m in enumerate(batch)]
                combined_body = "\n".join(parts)
                # Use the last message as the base (most recent timestamp, may have image)
                msg = batch[-1].model_copy(update={"body": combined_body})
                logger.info(
                    "messages combined",
                    extra={
                        "event": "queue_batch",
                        "room": room,
                        "count": len(batch),
                    },
                )

            try:
                await self._handler(msg)
            except Exception:
                logger.exception("Error processing queued message in %s", room)

        # Check if new messages arrived while we were processing
        if self._queues[room]:
            logger.info(
                "draining follow-up messages",
                extra={
                    "event": "queue_followup",
                    "room": room,
                    "queue_len": len(self._queues[room]),
                },
            )
            await self._drain(room)
