"""Tests for search_chat and search_tool_history tools."""

import pytest
import pytest_asyncio

from aineko.db import init_engine, create_tables, dispose_engine, get_session
from aineko.models.message import Message, Role, Session, ToolLog
from aineko.tools.search_chat import _search_chat
from aineko.tools.tool_history import _search_tool_history


@pytest_asyncio.fixture
async def setup_db(tmp_path):
    db_path = tmp_path / "test.db"
    init_engine(f"sqlite+aiosqlite:///{db_path}")
    await create_tables()
    yield
    await dispose_engine()


@pytest_asyncio.fixture
async def populated_db(setup_db):
    """Seed DB with a session, messages, and tool logs."""
    async for db in get_session():
        session = Session(room_id="!room:test")
        db.add(session)
        await db.flush()

        msgs = [
            Message(session_id=session.id, role=Role.USER, content="hello world"),
            Message(
                session_id=session.id, role=Role.ASSISTANT, content="hi there, world"
            ),
            Message(session_id=session.id, role=Role.USER, content="deploy the app"),
            Message(
                session_id=session.id, role=Role.SYSTEM, content="system init world"
            ),
            Message(session_id=session.id, role=Role.TOOL, content="tool result world"),
        ]
        db.add_all(msgs)
        await db.flush()

        # Add tool logs linked to the first user message
        user_msg = msgs[0]
        logs = [
            ToolLog(
                session_id=session.id,
                message_id=user_msg.id,
                tool_name="bash",
                arguments='{"command": "ls"}',
                result="file1.txt\nfile2.txt",
            ),
            ToolLog(
                session_id=session.id,
                message_id=user_msg.id,
                tool_name="read_file",
                arguments='{"path": "/data/config.json"}',
                result='{"key": "value"}',
            ),
            ToolLog(
                session_id=session.id,
                message_id=user_msg.id,
                tool_name="bash",
                arguments='{"command": "deploy --prod"}',
                result="deployed successfully",
            ),
        ]
        db.add_all(logs)
        await db.commit()


class TestSearchChat:
    @pytest.mark.asyncio
    async def test_keyword_match(self, populated_db):
        result = await _search_chat(keyword="world")
        assert "hello world" in result
        assert "hi there, world" in result

    @pytest.mark.asyncio
    async def test_excludes_tool_messages(self, populated_db):
        result = await _search_chat(keyword="world")
        assert "tool result world" not in result

    @pytest.mark.asyncio
    async def test_role_filter_user(self, populated_db):
        result = await _search_chat(keyword="world", role="user")
        assert "hello world" in result
        assert "hi there, world" not in result

    @pytest.mark.asyncio
    async def test_role_filter_assistant(self, populated_db):
        result = await _search_chat(keyword="world", role="assistant")
        assert "hi there, world" in result
        assert "hello world" not in result

    @pytest.mark.asyncio
    async def test_invalid_role(self, populated_db):
        result = await _search_chat(keyword="hello", role="invalid_role")
        assert "Invalid role" in result

    @pytest.mark.asyncio
    async def test_no_matches(self, populated_db):
        result = await _search_chat(keyword="zzzznonexistent")
        assert "No messages found" in result

    @pytest.mark.asyncio
    async def test_limit_respected(self, populated_db):
        result = await _search_chat(keyword="world", limit=1)
        # Should have at most 1 message in output
        lines = [line for line in result.split("\n\n") if line.strip()]
        assert len(lines) == 1

    @pytest.mark.asyncio
    async def test_limit_capped_at_50(self, populated_db):
        # Passing limit > 50 should be capped
        result = await _search_chat(keyword="world", limit=100)
        # Should not error, just cap internally
        assert isinstance(result, str)


class TestSearchToolHistory:
    @pytest.mark.asyncio
    async def test_all_logs(self, populated_db):
        result = await _search_tool_history()
        assert "bash" in result
        assert "read_file" in result

    @pytest.mark.asyncio
    async def test_filter_by_tool_name(self, populated_db):
        result = await _search_tool_history(tool_name="read_file")
        assert "read_file" in result
        assert "deploy" not in result

    @pytest.mark.asyncio
    async def test_filter_by_keyword(self, populated_db):
        result = await _search_tool_history(keyword="deploy")
        assert "deploy" in result

    @pytest.mark.asyncio
    async def test_no_matches(self, populated_db):
        result = await _search_tool_history(tool_name="nonexistent_tool")
        assert "No tool call history found" in result

    @pytest.mark.asyncio
    async def test_includes_triggering_message(self, populated_db):
        result = await _search_tool_history(tool_name="bash")
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_limit(self, populated_db):
        result = await _search_tool_history(limit=1)
        lines = [line for line in result.split("\n\n") if line.strip()]
        assert len(lines) == 1
