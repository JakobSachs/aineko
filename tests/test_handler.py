"""Tests for decomposed handle_message functions."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import event as sa_event, select

from aineko.models.message import Message, Role, Session, ToolLog
from aineko.schemas.message import IncomingMessage
from aineko.tools.registry import ToolDef, ToolRegistry


@pytest_asyncio.fixture
async def db(pg_session):
    yield pg_session


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


# --- format_tool_footer ---


class TestFormatToolFooter:
    def _rec(self, name, arguments="{}", result="ok"):
        from aineko.kimi.client import ToolCallRecord

        return ToolCallRecord(tool_name=name, arguments=arguments, result=result)

    def test_empty_history_returns_empty(self):
        from aineko.handler import format_tool_footer

        assert format_tool_footer([]) == ""

    def test_only_visible_tools_returns_empty(self):
        from aineko.handler import format_tool_footer

        history = [self._rec("send_message"), self._rec("send_file")]
        assert format_tool_footer(history) == ""

    def test_single_tool_uses_singular(self):
        from aineko.handler import format_tool_footer

        assert format_tool_footer([self._rec("bash")]) == "\n\n*— 1 tool*"

    def test_multiple_tools_uses_plural(self):
        from aineko.handler import format_tool_footer

        history = [self._rec("bash"), self._rec("read_file"), self._rec("grep")]
        assert format_tool_footer(history) == "\n\n*— 3 tools*"

    def test_excludes_visible_tools_from_count(self):
        from aineko.handler import format_tool_footer

        history = [
            self._rec("bash"),
            self._rec("send_message"),
            self._rec("send_message"),
            self._rec("web_search"),
        ]
        assert format_tool_footer(history) == "\n\n*— 2 tools*"

    def test_memory_search_excluded_from_tool_count(self):
        from aineko.handler import format_tool_footer

        history = [
            self._rec("bash"),
            self._rec("memory", json.dumps({"action": "search", "query": "test"})),
        ]
        assert format_tool_footer(history) == "\n\n*— 1 tool*"

    def test_memory_store_shows_memory_count(self):
        from aineko.handler import format_tool_footer

        history = [
            self._rec("memory", json.dumps({"action": "store", "content": "note"})),
        ]
        assert format_tool_footer(history) == "\n\n*— 1 memory*"

    def test_memory_facts_add_shows_memory_count(self):
        from aineko.handler import format_tool_footer

        history = [
            self._rec(
                "memory",
                json.dumps(
                    {
                        "action": "facts_add",
                        "subject": "X",
                        "predicate": "is",
                        "object": "Y",
                    }
                ),
            ),
        ]
        assert format_tool_footer(history) == "\n\n*— 1 memory*"

    def test_memory_writes_plural(self):
        from aineko.handler import format_tool_footer

        history = [
            self._rec("memory", json.dumps({"action": "store", "content": "a"})),
            self._rec(
                "memory",
                json.dumps(
                    {
                        "action": "facts_add",
                        "subject": "X",
                        "predicate": "is",
                        "object": "Y",
                    }
                ),
            ),
        ]
        assert format_tool_footer(history) == "\n\n*— 2 memories*"

    def test_tools_and_memories_combined(self):
        from aineko.handler import format_tool_footer

        history = [
            self._rec("bash"),
            self._rec("web_search"),
            self._rec("memory", json.dumps({"action": "search", "query": "q"})),
            self._rec("memory", json.dumps({"action": "store", "content": "note"})),
        ]
        assert format_tool_footer(history) == "\n\n*— 2 tools, 1 memory*"

    def test_only_memory_reads_returns_empty(self):
        from aineko.handler import format_tool_footer

        history = [
            self._rec("memory", json.dumps({"action": "search", "query": "q"})),
            self._rec("memory", json.dumps({"action": "facts_query", "entity": "X"})),
        ]
        assert format_tool_footer(history) == ""

    def test_malformed_memory_arguments_ignored(self):
        from aineko.handler import format_tool_footer

        history = [
            self._rec("memory", "not valid json"),
            self._rec("bash"),
        ]
        assert format_tool_footer(history) == "\n\n*— 1 tool*"


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
        assert "hello" in messages[-1]["content"]
        assert "[sent " in messages[-1]["content"]

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
    async def test_history_ordering_is_deterministic_under_ties(self, db, msg):
        """Regression: messages with identical created_at must come out in
        insertion order so tool_use stays paired before tool_result.
        SQLite tie-broke by rowid; PG returns ties arbitrarily unless the
        ORDER BY also names id as a secondary key.

        Structural check — asserts the query as executed by load_conversation
        includes an id tie-breaker. The behavioural alternative (insert ties
        and observe order) is flaky: for small row counts PG happens to
        return insert-order even without the tie-breaker.
        """
        import re

        from aineko.handler import load_conversation

        s = Session(room_id="!room:test")
        db.add(s)
        await db.flush()

        tied = datetime(2026, 1, 1, 12, 0, 0)
        db.add(
            Message(
                session_id=s.id,
                role=Role.ASSISTANT,
                content=json.dumps(
                    [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]
                ),
                tool_name="__has_tool_calls__",
                created_at=tied,
            )
        )
        await db.flush()
        db.add(
            Message(
                session_id=s.id,
                role=Role.TOOL,
                content=json.dumps(
                    [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]
                ),
                tool_name="__tool_results__",
                created_at=tied,
            )
        )
        await db.flush()

        # Behavioural sanity check — in PG small queries this tends to pass
        # even without the fix, but it's still worth keeping as a smoke test.
        _, _, messages = await load_conversation(db, msg, system_prompt="sys")
        use_idx = next(
            i
            for i, m in enumerate(messages)
            if m["role"] == "assistant"
            and isinstance(m["content"], list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        )
        res_idx = next(
            i
            for i, m in enumerate(messages)
            if m["role"] == "user"
            and isinstance(m["content"], list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        )
        assert use_idx < res_idx

        # Structural guard — capture the actual SQL load_conversation emits
        # and assert the ORDER BY includes id as a tie-breaker.
        captured: list[str] = []
        engine = db.bind
        assert engine is not None

        @sa_event.listens_for(engine.sync_engine, "before_cursor_execute")
        def _capture(
            conn, cursor, statement, params, context, executemany
        ):  # noqa: ANN001
            if "FROM messages" in statement and "ORDER BY" in statement:
                captured.append(statement)

        try:
            await load_conversation(db, msg, system_prompt="sys")
        finally:
            sa_event.remove(engine.sync_engine, "before_cursor_execute", _capture)

        history_sqls = [s for s in captured if "ORDER BY" in s]
        assert history_sqls, "load_conversation did not issue an ordered history query"
        order_by = re.split(r"\bORDER BY\b", history_sqls[-1], maxsplit=1)[1]
        assert (
            "messages.id" in order_by
        ), f"ORDER BY must include messages.id as tie-breaker; got: {order_by.strip()}"

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
        img_block = last["content"][1]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/png"
        assert img_block["source"]["data"] == "abc123"


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

        await persist_response(db, s.id, user_msg.id, response, sent_messages=[])

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

        await persist_response(db, s.id, user_msg.id, response, sent_messages=[])

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
            db, s.id, user_msg.id, response, sent_messages=["intermediate msg"]
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

        await persist_response(db, s.id, user_msg.id, response, sent_messages=[])

        result = await db.execute(
            select(Message).where(
                Message.session_id == s.id, Message.role == Role.ASSISTANT
            )
        )
        assert result.scalar_one_or_none() is None


# --- new_messages round-trip ---


class TestNewMessagesRoundTrip:
    """End-to-end: chat_loop produces blocks-with-thinking → persist → load
    → assert thinking blocks reappear in replayed messages.

    Regression for the hallucination caused by stripped thinking blocks
    tripping kimi.client._history_missing_reasoning.
    """

    @pytest.mark.asyncio
    async def test_thinking_blocks_survive_persist_and_load(self, db, msg):
        from aineko.handler import load_conversation, persist_response
        from aineko.kimi.client import _history_missing_reasoning

        s = Session(room_id="!room:test")
        db.add(s)
        await db.flush()
        user_msg = Message(session_id=s.id, role=Role.USER, content="hi")
        db.add(user_msg)
        await db.flush()

        # Simulate what chat_loop appends across one tool round + final.
        new_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "i should grep for it",
                        "signature": "sig-1",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "grep",
                        "input": {"pattern": "x"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "match",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "found it, replying",
                        "signature": "sig-2",
                    },
                    {"type": "text", "text": "found it"},
                ],
            },
        ]

        response = MagicMock()
        response.content = "found it"
        response.tool_history = []
        response.new_messages = new_messages
        response.usage = {"total_tokens": 123}

        await persist_response(db, s.id, user_msg.id, response, sent_messages=[])

        # New incoming user msg triggers replay.
        next_msg = IncomingMessage(
            room_id="!room:test",
            sender="@user:test",
            body="and then?",
            event_id="$evt2",
            timestamp=datetime.now(timezone.utc),
        )
        _, _, replayed = await load_conversation(db, next_msg, system_prompt="sys")

        # Drop the system prompt; everything else is the conversation.
        convo = [m for m in replayed if m.get("role") != "system"]

        # Must NOT trigger the thinking-disable guard — every assistant
        # message with tool_use carries a thinking block.
        assert not _history_missing_reasoning(convo), (
            "history-missing-reasoning guard tripped: "
            f"{json.dumps(convo, default=str)[:500]}"
        )

        # Spot-check: the assistant tool turn round-trips with both blocks.
        tool_turn = next(
            m
            for m in convo
            if m["role"] == "assistant"
            and isinstance(m["content"], list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        )
        types = [b["type"] for b in tool_turn["content"]]
        assert "thinking" in types
        assert "tool_use" in types

    @pytest.mark.asyncio
    async def test_legacy_sentinel_rows_still_replay(self, db, msg):
        """Pre-0003 rows have blocks=NULL. load_conversation must still
        decode them via the legacy fallback path so we keep working
        through the deploy/migration window."""
        from aineko.handler import load_conversation

        s = Session(room_id="!room:test")
        db.add(s)
        await db.flush()

        legacy_blocks = [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]
        db.add(
            Message(
                session_id=s.id,
                role=Role.ASSISTANT,
                content=json.dumps(legacy_blocks),
                tool_name="__has_tool_calls__",
            )
        )
        db.add(
            Message(
                session_id=s.id,
                role=Role.TOOL,
                content=json.dumps(
                    [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]
                ),
                tool_name="__tool_results__",
            )
        )
        await db.flush()

        _, _, messages = await load_conversation(db, msg, system_prompt="sys")
        roles_with_blocks = [m for m in messages if isinstance(m["content"], list)]
        assert any(
            any(b.get("type") == "tool_use" for b in m["content"])
            for m in roles_with_blocks
        )
        assert any(
            any(b.get("type") == "tool_result" for b in m["content"])
            for m in roles_with_blocks
        )
