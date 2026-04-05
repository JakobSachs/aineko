"""Tests for decomposed handle_message functions."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aineko.db import Base
from aineko.models.message import Message, Role, Session, ToolLog
from aineko.schemas.message import IncomingMessage
from aineko.tools.registry import ToolDef, ToolRegistry

# --- DB fixture (real in-memory SQLite) ---


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture
def matrix():
    return AsyncMock()


@pytest.fixture
def msg():
    return IncomingMessage(
        room_id="!room:test",
        sender="@user:test",
        body="hello",
        event_id="$evt1",
        timestamp=datetime.now(timezone.utc),
    )


# --- handle_command ---


class TestHandleCommand:
    @pytest.mark.asyncio
    async def test_reset_clears_messages(self, db, matrix):
        from aineko.handler import handle_command

        # Set up a session with messages
        session = Session(room_id="!room:test")
        db.add(session)
        await db.flush()
        db.add(Message(session_id=session.id, role=Role.USER, content="hi"))
        db.add(Message(session_id=session.id, role=Role.ASSISTANT, content="hello"))
        await db.commit()

        msg = IncomingMessage(
            room_id="!room:test",
            sender="@u:t",
            body="/reset",
            event_id="$e",
            timestamp=datetime.now(timezone.utc),
        )
        handled = await handle_command(db, msg, matrix)

        assert handled is True
        matrix.send_message.assert_awaited_once()
        result = await db.execute(
            select(Message).where(Message.session_id == session.id)
        )
        assert result.scalars().all() == []

    @pytest.mark.asyncio
    async def test_clear_also_works(self, db, matrix):
        from aineko.handler import handle_command

        msg = IncomingMessage(
            room_id="!room:test",
            sender="@u:t",
            body="/clear",
            event_id="$e",
            timestamp=datetime.now(timezone.utc),
        )
        handled = await handle_command(db, msg, matrix)
        assert handled is True

    @pytest.mark.asyncio
    async def test_normal_message_not_handled(self, db, matrix):
        from aineko.handler import handle_command

        msg = IncomingMessage(
            room_id="!room:test",
            sender="@u:t",
            body="hello",
            event_id="$e",
            timestamp=datetime.now(timezone.utc),
        )
        handled = await handle_command(db, msg, matrix)
        assert handled is False
        matrix.send_message.assert_not_awaited()


# --- build_request_tools ---


class TestBuildRequestTools:
    def test_includes_base_tools(self):
        from aineko.handler import build_request_tools

        base = ToolRegistry()
        base.register(
            ToolDef(name="bash", description="run", parameters={}, handler=AsyncMock())
        )
        result = build_request_tools(base, AsyncMock(), "!room:x", [])
        assert result.get("bash") is not None

    def test_includes_send_message(self):
        from aineko.handler import build_request_tools

        result = build_request_tools(ToolRegistry(), AsyncMock(), "!room:x", [])
        assert result.get("send_message") is not None

    def test_includes_send_file(self):
        from aineko.handler import build_request_tools

        result = build_request_tools(ToolRegistry(), AsyncMock(), "!room:x", [])
        assert result.get("send_file") is not None

    def test_includes_bg_task_tools_when_provided(self):
        from aineko.handler import build_request_tools

        bg = MagicMock()
        result = build_request_tools(
            ToolRegistry(), AsyncMock(), "!room:x", [], bg_tasks=bg
        )
        assert result.get("spawn_task") is not None
        assert result.get("list_background_tasks") is not None
        assert result.get("get_task_result") is not None

    def test_no_bg_tools_when_none(self):
        from aineko.handler import build_request_tools

        result = build_request_tools(ToolRegistry(), AsyncMock(), "!room:x", [])
        assert result.get("spawn_task") is None


# --- load_conversation ---


class TestLoadConversation:
    @pytest.mark.asyncio
    async def test_creates_new_session(self, db, msg):
        from aineko.handler import load_conversation

        session, user_msg, messages = await load_conversation(
            db, msg, system_prompt="You are aineko."
        )
        assert session.room_id == "!room:test"
        assert user_msg.content == "hello"
        assert user_msg.role == Role.USER
        # Messages: system + user
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_reuses_existing_session(self, db, msg):
        from aineko.handler import load_conversation

        existing = Session(room_id="!room:test")
        db.add(existing)
        await db.flush()

        session, _, _ = await load_conversation(db, msg, system_prompt="sys")
        assert session.id == existing.id

    @pytest.mark.asyncio
    async def test_includes_history(self, db, msg):
        from aineko.handler import load_conversation

        s = Session(room_id="!room:test")
        db.add(s)
        await db.flush()
        db.add(Message(session_id=s.id, role=Role.USER, content="older"))
        db.add(Message(session_id=s.id, role=Role.ASSISTANT, content="reply"))
        await db.flush()

        _, _, messages = await load_conversation(db, msg, system_prompt="sys")
        # system + older user + assistant reply + new user
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]

    @pytest.mark.asyncio
    async def test_reconstructs_tool_call_blocks(self, db, msg):
        from aineko.handler import load_conversation

        s = Session(room_id="!room:test")
        db.add(s)
        await db.flush()
        blocks = [
            {"type": "text", "text": "thinking"},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
        ]
        db.add(
            Message(
                session_id=s.id,
                role=Role.ASSISTANT,
                content=json.dumps(blocks),
                tool_name="__has_tool_calls__",
            )
        )
        result_blocks = [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]
        db.add(
            Message(
                session_id=s.id,
                role=Role.TOOL,
                content=json.dumps(result_blocks),
                tool_name="__tool_results__",
            )
        )
        await db.flush()

        _, _, messages = await load_conversation(db, msg, system_prompt="sys")
        # system + assistant(blocks) + user(tool_results) + new user
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == blocks
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == result_blocks

    @pytest.mark.asyncio
    async def test_image_attachment(self, db):
        from aineko.handler import load_conversation

        img_msg = IncomingMessage(
            room_id="!room:test",
            sender="@u:t",
            body="what is this?",
            event_id="$e",
            timestamp=datetime.now(timezone.utc),
            image_b64="abc123",
            image_mime="image/png",
        )
        _, _, messages = await load_conversation(db, img_msg, system_prompt="sys")
        last = messages[-1]
        assert isinstance(last["content"], list)
        assert last["content"][0]["type"] == "text"
        assert last["content"][1]["type"] == "image_url"


# --- persist_response ---


class TestPersistResponse:
    @pytest.mark.asyncio
    async def test_saves_plain_response(self, db):
        from aineko.handler import persist_response

        s = Session(room_id="!room:test")
        db.add(s)
        await db.flush()
        user_msg = Message(session_id=s.id, role=Role.USER, content="hi")
        db.add(user_msg)
        await db.flush()

        response = MagicMock()
        response.content = "hello back"
        response.tool_history = []
        response.usage = {"total_tokens": 100}

        await persist_response(db, s, user_msg, response, sent_messages=[])

        result = await db.execute(
            select(Message).where(
                Message.session_id == s.id, Message.role == Role.ASSISTANT
            )
        )
        saved = result.scalar_one()
        assert saved.content == "hello back"
        assert saved.token_count == 100

    @pytest.mark.asyncio
    async def test_saves_tool_history(self, db):
        from aineko.handler import persist_response

        s = Session(room_id="!room:test")
        db.add(s)
        await db.flush()
        user_msg = Message(session_id=s.id, role=Role.USER, content="hi")
        db.add(user_msg)
        await db.flush()

        tool_rec = MagicMock()
        tool_rec.tool_name = "bash"
        tool_rec.arguments = '{"command": "ls"}'
        tool_rec.result = "file.txt"

        response = MagicMock()
        response.content = "here are your files"
        response.tool_history = [tool_rec]
        response.usage = {"total_tokens": 200}

        await persist_response(db, s, user_msg, response, sent_messages=[])

        # Check tool log
        result = await db.execute(select(ToolLog).where(ToolLog.session_id == s.id))
        log = result.scalar_one()
        assert log.tool_name == "bash"
        assert log.arguments == '{"command": "ls"}'

        # Check assistant message has tool_use blocks
        result = await db.execute(
            select(Message).where(
                Message.session_id == s.id, Message.tool_name == "__has_tool_calls__"
            )
        )
        saved = result.scalar_one()
        blocks = json.loads(saved.content)
        tool_uses = [b for b in blocks if b["type"] == "tool_use"]
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "bash"

    @pytest.mark.asyncio
    async def test_includes_sent_messages_in_blocks(self, db):
        from aineko.handler import persist_response

        s = Session(room_id="!room:test")
        db.add(s)
        await db.flush()
        user_msg = Message(session_id=s.id, role=Role.USER, content="hi")
        db.add(user_msg)
        await db.flush()

        tool_rec = MagicMock()
        tool_rec.tool_name = "bash"
        tool_rec.arguments = "{}"
        tool_rec.result = "ok"

        response = MagicMock()
        response.content = "done"
        response.tool_history = [tool_rec]
        response.usage = {"total_tokens": 50}

        await persist_response(
            db, s, user_msg, response, sent_messages=["intermediate msg"]
        )

        result = await db.execute(
            select(Message).where(
                Message.session_id == s.id, Message.tool_name == "__has_tool_calls__"
            )
        )
        blocks = json.loads(result.scalar_one().content)
        texts = [b["text"] for b in blocks if b["type"] == "text"]
        assert "intermediate msg" in texts

    @pytest.mark.asyncio
    async def test_no_content_no_save(self, db):
        from aineko.handler import persist_response

        s = Session(room_id="!room:test")
        db.add(s)
        await db.flush()
        user_msg = Message(session_id=s.id, role=Role.USER, content="hi")
        db.add(user_msg)
        await db.flush()

        response = MagicMock()
        response.content = ""
        response.tool_history = []
        response.usage = {}

        await persist_response(db, s, user_msg, response, sent_messages=[])

        result = await db.execute(
            select(Message).where(
                Message.session_id == s.id, Message.role == Role.ASSISTANT
            )
        )
        assert result.scalar_one_or_none() is None
