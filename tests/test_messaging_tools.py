"""Tests for messaging tool factories (send_message, send_file, background tasks)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aineko.tools.messaging import (
    make_send_message_tool,
    make_send_file_tool,
    make_background_task_tools,
)
from aineko.tools.registry import ToolDef

# --- send_message ---


class TestMakeSendMessageTool:
    def test_returns_tooldef(self):
        tool = make_send_message_tool(AsyncMock(), "!room:x", [])
        assert isinstance(tool, ToolDef)
        assert tool.name == "send_message"

    @pytest.mark.asyncio
    async def test_sends_to_default_room(self):
        matrix = AsyncMock()
        sent = []
        tool = make_send_message_tool(matrix, "!room:x", sent)

        result = await tool.handler(message="hello")
        assert result == "sent"
        matrix.send_message.assert_awaited_once_with("!room:x", "hello")
        assert sent == ["hello"]

    @pytest.mark.asyncio
    async def test_sends_to_explicit_room(self):
        matrix = AsyncMock()
        sent = []
        tool = make_send_message_tool(matrix, "!room:x", sent)

        result = await tool.handler(message="hi", room="!other:x")
        assert result == "sent"
        matrix.send_message.assert_awaited_once_with("!other:x", "hi")

    @pytest.mark.asyncio
    async def test_tracks_multiple_messages(self):
        matrix = AsyncMock()
        sent = []
        tool = make_send_message_tool(matrix, "!room:x", sent)

        await tool.handler(message="one")
        await tool.handler(message="two")
        assert sent == ["one", "two"]


# --- send_file ---


class TestMakeSendFileTool:
    def test_returns_tooldef(self):
        tool = make_send_file_tool(AsyncMock(), "!room:x")
        assert isinstance(tool, ToolDef)
        assert tool.name == "send_file"

    @pytest.mark.asyncio
    async def test_delegates_to_matrix_send_file(self):
        matrix = AsyncMock()
        tool = make_send_file_tool(matrix, "!room:x")

        with pytest.MonkeyPatch.context() as mp:
            mock_send = AsyncMock(return_value="Sent file.txt")
            mp.setattr("aineko.tools.messaging.send_file_via_matrix", mock_send)

            result = await tool.handler(path="file.txt")
            assert result == "Sent file.txt"
            mock_send.assert_awaited_once_with(
                matrix, "!room:x", "file.txt", filename=""
            )

    @pytest.mark.asyncio
    async def test_explicit_room_and_filename(self):
        matrix = AsyncMock()
        tool = make_send_file_tool(matrix, "!room:x")

        with pytest.MonkeyPatch.context() as mp:
            mock_send = AsyncMock(return_value="Sent")
            mp.setattr("aineko.tools.messaging.send_file_via_matrix", mock_send)

            await tool.handler(path="a.csv", filename="b.csv", room="!other:x")
            mock_send.assert_awaited_once_with(
                matrix, "!other:x", "a.csv", filename="b.csv"
            )


# --- background task tools ---


class TestMakeBackgroundTaskTools:
    def test_returns_three_tools(self):
        bg = MagicMock()
        tools = make_background_task_tools(bg, "!room:x", AsyncMock())
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"spawn_task", "list_background_tasks", "get_task_result"}

    def test_all_are_tooldefs(self):
        bg = MagicMock()
        tools = make_background_task_tools(bg, "!room:x", AsyncMock())
        for t in tools:
            assert isinstance(t, ToolDef)

    @pytest.mark.asyncio
    async def test_list_background_tasks(self):
        bg = MagicMock()
        bg.list_tasks.return_value = []
        tools = make_background_task_tools(bg, "!room:x", AsyncMock())
        list_tool = next(t for t in tools if t.name == "list_background_tasks")

        result = await list_tool.handler()
        assert result == "No background tasks."

    @pytest.mark.asyncio
    async def test_list_background_tasks_with_entries(self):
        bg = MagicMock()
        bg.list_tasks.return_value = [
            {"id": "abc", "label": "build", "finished": False, "elapsed": "5s"},
        ]
        tools = make_background_task_tools(bg, "!room:x", AsyncMock())
        list_tool = next(t for t in tools if t.name == "list_background_tasks")

        result = await list_tool.handler()
        assert "abc" in result
        assert "build" in result
        assert "running" in result

    @pytest.mark.asyncio
    async def test_get_task_result_not_found(self):
        bg = MagicMock()
        bg.get.return_value = None
        tools = make_background_task_tools(bg, "!room:x", AsyncMock())
        get_tool = next(t for t in tools if t.name == "get_task_result")

        result = await get_tool.handler(task_id="nope")
        assert "nope" in result

    @pytest.mark.asyncio
    async def test_spawn_task_calls_bg_manager(self):
        bg = MagicMock()
        record = MagicMock()
        record.id = "abc123"
        record.label = "test build"
        bg.spawn.return_value = record

        tools = make_background_task_tools(bg, "!room:x", AsyncMock())
        spawn_tool = next(t for t in tools if t.name == "spawn_task")

        result = await spawn_tool.handler(command="make build")
        assert "abc123" in result
        bg.spawn.assert_called_once()
        # Verify room_id was passed through
        call_kwargs = bg.spawn.call_args
        assert (
            call_kwargs.kwargs.get("room_id") == "!room:x"
            or call_kwargs[1].get("room_id") == "!room:x"
        )

    @pytest.mark.asyncio
    async def test_inject_result_sends_to_matrix(self):
        """_inject_result calls matrix.inject_message with formatted body."""
        matrix = AsyncMock()
        bg = MagicMock()
        tools = make_background_task_tools(bg, "!room:x", matrix)

        # The inject_fn is passed to bg_tasks.spawn — extract it from the spawn call
        spawn_tool = next(t for t in tools if t.name == "spawn_task")
        record = MagicMock()
        record.id = "t1"
        record.label = "build"
        bg.spawn.return_value = record
        await spawn_tool.handler(command="make")

        inject_fn = bg.spawn.call_args.kwargs.get("inject_fn") or bg.spawn.call_args[
            1
        ].get("inject_fn")
        await inject_fn("t1", "build", "success")

        matrix.inject_message.assert_awaited_once()
        body = matrix.inject_message.call_args[0][1]
        assert "build" in body
        assert "t1" in body
        assert "success" in body

    @pytest.mark.asyncio
    async def test_get_task_result_running(self):
        """get_task_result shows elapsed time for running tasks."""
        from aineko.tools.background_task import TaskRecord

        bg = MagicMock()
        record = TaskRecord(
            id="r1",
            command="sleep 60",
            label="long task",
            room_id="!room:x",
            created_at=datetime.now(timezone.utc),
            finished=False,
        )
        bg.get.return_value = record

        tools = make_background_task_tools(bg, "!room:x", AsyncMock())
        get_tool = next(t for t in tools if t.name == "get_task_result")

        result = await get_tool.handler(task_id="r1")
        assert "still running" in result
        assert "long task" in result

    @pytest.mark.asyncio
    async def test_get_task_result_finished(self):
        """get_task_result shows result for finished tasks."""
        from aineko.tools.background_task import TaskRecord

        bg = MagicMock()
        record = TaskRecord(
            id="f1",
            command="echo done",
            label="quick",
            room_id="!room:x",
            created_at=datetime.now(timezone.utc),
            finished=True,
            result="done\n[exit code: 0]",
        )
        bg.get.return_value = record

        tools = make_background_task_tools(bg, "!room:x", AsyncMock())
        get_tool = next(t for t in tools if t.name == "get_task_result")

        result = await get_tool.handler(task_id="f1")
        assert "finished" in result
        assert "done" in result
