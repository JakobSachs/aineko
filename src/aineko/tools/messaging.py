"""Factory functions for context-dependent tools (send_message, send_file, background tasks)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aineko.matrix.client import send_file as send_file_via_matrix
from aineko.tools.registry import ToolDef

if TYPE_CHECKING:
    from aineko.matrix.client import MatrixConnector
    from aineko.tools.background_task import BackgroundTaskManager


def make_send_message_tool(
    matrix: MatrixConnector,
    default_room: str,
    sent_messages: list[str],
) -> ToolDef:
    async def handler(message: str, room: str = "") -> str:
        target = room or default_room
        sent_messages.append(message)
        await matrix.send_message(target, message)
        return "sent"

    return ToolDef(
        name="send_message",
        description=(
            "Send a message to the user. Call this to reply — your response "
            "text is NOT automatically sent. Use this for every message you "
            "want the user to see."
        ),
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message text to send.",
                },
                "room": {
                    "type": "string",
                    "description": "Room ID to send to (defaults to current room).",
                },
            },
            "required": ["message"],
        },
        handler=handler,
    )


def make_send_file_tool(
    matrix: MatrixConnector,
    default_room: str,
) -> ToolDef:
    async def handler(path: str, filename: str = "", room: str = "") -> str:
        target = room or default_room
        return await send_file_via_matrix(matrix, target, path, filename=filename)

    return ToolDef(
        name="send_file",
        description=(
            "Send a file from /data to the user via Matrix. "
            "Path is relative to /data. Images are displayed inline."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to /data.",
                },
                "filename": {
                    "type": "string",
                    "description": "Override the displayed filename (optional).",
                },
                "room": {
                    "type": "string",
                    "description": "Room ID to send to (defaults to current room).",
                },
            },
            "required": ["path"],
        },
        handler=handler,
    )


def make_background_task_tools(
    bg_tasks: BackgroundTaskManager,
    room_id: str,
    matrix: MatrixConnector,
) -> list[ToolDef]:
    async def _inject_result(task_id: str, label: str, result: str) -> None:
        body = f"[Background task `{label}` (id: {task_id}) completed]\n\n{result}"
        await matrix.inject_message(room_id, body, sender="background_task")

    async def spawn_handler(command: str, label: str = "", timeout: int = 600) -> str:
        record = bg_tasks.spawn(
            command=command,
            label=label or command[:60],
            room_id=room_id,
            inject_fn=_inject_result,
            timeout=timeout,
        )
        return (
            f"Task spawned (id: {record.id}, label: {record.label}). "
            f"The result will be delivered as a new message when complete. "
            f"You can continue with other work now."
        )

    async def list_handler() -> str:
        tasks = bg_tasks.list_tasks()
        if not tasks:
            return "No background tasks."
        lines = []
        for t in tasks:
            status = "finished" if t["finished"] else "running"
            elapsed = t.get("elapsed", "")
            line = f"- {t['id']}: [{status}] {t['label']}"
            if elapsed:
                line += f" ({elapsed})"
            lines.append(line)
        return "\n".join(lines)

    async def get_handler(task_id: str) -> str:
        record = bg_tasks.get(task_id)
        if record is None:
            return f"No task with id '{task_id}' found."
        if not record.finished:
            elapsed = (datetime.now(timezone.utc) - record.created_at).total_seconds()
            return (
                f"Task '{record.label}' is still running "
                f"(started {elapsed:.0f}s ago).\n"
                f"Command: {record.command}"
            )
        return (
            f"Task '{record.label}' finished.\n"
            f"Command: {record.command}\n\n"
            f"Result:\n{record.result}"
        )

    return [
        ToolDef(
            name="spawn_task",
            description=(
                "Spawn a long-running shell command as a background task. "
                "Returns immediately — the result will be injected back into "
                "the conversation as a new message when the task finishes. "
                "Use this for commands that take a long time (builds, downloads, "
                "large file processing, etc.) so you can continue responding."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run in the background.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Short human-readable label for the task.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 600, max 3600).",
                        "default": 600,
                    },
                },
                "required": ["command"],
            },
            handler=spawn_handler,
        ),
        ToolDef(
            name="list_background_tasks",
            description="List all background tasks and their status.",
            parameters={"type": "object", "properties": {}},
            handler=list_handler,
        ),
        ToolDef(
            name="get_task_result",
            description=(
                "Get the status and result of a background task by its ID. "
                "Returns the full output if finished, or elapsed time if still running."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID returned by spawn_task.",
                    },
                },
                "required": ["task_id"],
            },
            handler=get_handler,
        ),
    ]
