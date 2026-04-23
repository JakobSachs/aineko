"""Message queue with debounce and processing lock.

Collects rapid-fire messages into batches and queues messages
that arrive while the agent is busy processing. Single-room only.
"""

import asyncio
import logging
from collections import deque
from collections.abc import Callable, Coroutine
from typing import Any

from aineko.schemas.message import IncomingMessage

logger = logging.getLogger(__name__)

DEBOUNCE_MS = 400  # wait this long after last message before processing
MAX_QUEUE = 20  # max queued messages

MessageHandler = Callable[
    [IncomingMessage, "asyncio.Queue[str]"], Coroutine[Any, Any, None]
]


class MessageQueue:
    def __init__(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._queue: deque[IncomingMessage] = deque(maxlen=MAX_QUEUE)
        self._lock: asyncio.Lock = asyncio.Lock()
        self._debounce_task: asyncio.Task | None = None
        # While a handler is running, text-only messages go here so the
        # agent can pick them up mid-turn instead of waiting for the current
        # turn to finish.
        self._active_interject: asyncio.Queue[str] | None = None

    async def enqueue(self, msg: IncomingMessage) -> None:
        """Add a message to the queue and schedule processing."""
        # Interject text-only messages into an in-progress turn if possible.
        # Images still fall through to the normal queue — vision input needs
        # a fresh turn with full message context.
        if self._active_interject is not None and msg.body and msg.image_b64 is None:
            self._active_interject.put_nowait(msg.body)
            logger.info(
                "message interjected into in-progress turn",
                extra={
                    "event": "interject_enqueue",
                    "room": msg.room_id,
                    "body": msg.body,
                },
            )
            return

        self._queue.append(msg)
        logger.info(
            "message queued",
            extra={
                "event": "queue_add",
                "room": msg.room_id,
                "body": msg.body,
                "queue_len": len(self._queue),
            },
        )

        if self._debounce_task is not None:
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounce_then_drain())

    async def _debounce_then_drain(self) -> None:
        """Wait for debounce period, then drain the queue."""
        try:
            await asyncio.sleep(DEBOUNCE_MS / 1000)
        except asyncio.CancelledError:
            return  # new message arrived, debounce reset
        # Once past the sleep we are an *active handler* — enqueue() must not
        # cancel an in-progress drain (which would kill the running LLM request).
        self._debounce_task = None
        await self._drain()

    async def _drain(self) -> None:
        """Process all queued messages."""
        async with self._lock:
            if not self._queue:
                return

            batch = list(self._queue)
            self._queue.clear()

            if len(batch) == 1:
                msg = batch[0]
            else:
                parts = [f"[message {i+1}]: {m.body}" for i, m in enumerate(batch)]
                combined_body = "\n".join(parts)
                # Use the last message as the base (most recent timestamp, may have image)
                msg = batch[-1].model_copy(update={"body": combined_body})
                logger.info(
                    "messages combined",
                    extra={"event": "queue_batch", "count": len(batch)},
                )

            interject: asyncio.Queue[str] = asyncio.Queue()
            self._active_interject = interject
            try:
                await self._handler(msg, interject)
            except Exception:
                logger.exception("Error processing queued message")
            finally:
                self._active_interject = None

        # Check if new messages arrived while we were processing
        if self._queue:
            logger.info(
                "draining follow-up messages",
                extra={"event": "queue_followup", "queue_len": len(self._queue)},
            )
            await self._drain()
