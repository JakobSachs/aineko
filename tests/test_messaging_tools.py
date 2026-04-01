"""Tests for messaging tool factories (send_message, send_file, background tasks)."""

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

            result = await tool.handler(path="a.csv", filename="b.csv", room="!other:x")
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
