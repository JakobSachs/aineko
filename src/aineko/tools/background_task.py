"""Background task tool — spawn long-running commands that report back on completion."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aineko.tools import bash as bash_mod

logger = logging.getLogger(__name__)


@dataclass
class TaskRecord:
    id: str
    command: str
    label: str
    room_id: str
    created_at: datetime
    asyncio_task: asyncio.Task | None = None
    result: str | None = None
    finished: bool = False


class BackgroundTaskManager:
    """Manages background tasks that inject results back into the message queue."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}

    def spawn(
        self,
        command: str,
        label: str,
        room_id: str,
        inject_fn: Any = None,
        timeout: int = 600,
    ) -> TaskRecord:
        task_id = uuid.uuid4().hex[:8]
        record = TaskRecord(
            id=task_id,
            command=command,
            label=label or command[:60],
            room_id=room_id,
            created_at=datetime.now(timezone.utc),
        )
        self._tasks[task_id] = record

        async def _run() -> None:
            try:
                result = await bash_mod.run_bash(command, timeout=timeout)
                record.result = result
                record.finished = True
                logger.info("background task %s finished", task_id)
                if inject_fn:
                    await inject_fn(task_id, record.label, result)
            except Exception as e:
                record.result = f"Error: {e}"
                record.finished = True
                logger.exception("background task %s failed", task_id)
                if inject_fn:
                    await inject_fn(task_id, record.label, record.result)

        record.asyncio_task = asyncio.create_task(_run(), name=f"bg-task-{task_id}")
        return record

    def list_tasks(self) -> list[dict]:
        now = datetime.now(timezone.utc)
        results = []
        for t in self._tasks.values():
            elapsed_s = (now - t.created_at).total_seconds()
            if elapsed_s < 60:
                elapsed = f"{elapsed_s:.0f}s"
            else:
                elapsed = f"{elapsed_s / 60:.1f}m"
            results.append(
                {
                    "id": t.id,
                    "label": t.label,
                    "command": t.command,
                    "room_id": t.room_id,
                    "finished": t.finished,
                    "created_at": t.created_at.isoformat(),
                    "elapsed": elapsed,
                }
            )
        return results

    def get(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def cleanup_finished(self) -> int:
        """Remove finished tasks from tracking. Returns count removed."""
        finished = [tid for tid, t in self._tasks.items() if t.finished]
        for tid in finished:
            del self._tasks[tid]
        return len(finished)
