"""Tests for messaging deduplication logic.

Ported from OpenClaw's messaging-dedupe.ts — ensures intermediate content
and send_message tool calls don't produce duplicate messages.
"""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from aineko.messaging_dedupe import (
    MIN_DUPLICATE_LENGTH,
    MessageDeduplicator,
    is_duplicate,
    normalize_text,
)

# Strategy: strings that might trip up parsers (unicode, control chars, emoji, etc.)
_nasty_text = st.text(
    alphabet=st.characters(categories=("L", "M", "N", "P", "S", "Z", "C")),
    min_size=0,
    max_size=500,
)


# --- Property-based tests ---


class TestNormalizeTextProperties:
    @given(text=_nasty_text)
    def test_never_crashes(self, text: str):
        normalize_text(text)

    @given(text=_nasty_text)
    def test_result_is_lowercase(self, text: str):
        assert normalize_text(text) == normalize_text(text).lower()

    @given(text=_nasty_text)
    def test_no_leading_trailing_whitespace(self, text: str):
        result = normalize_text(text)
        assert result == result.strip()

    @given(text=_nasty_text)
    def test_no_double_spaces(self, text: str):
        assert "  " not in normalize_text(text)

    @given(text=st.text(min_size=1, max_size=200))
    def test_idempotent(self, text: str):
        once = normalize_text(text)
        assert normalize_text(once) == once


class TestIsDuplicateProperties:
    @given(text=st.text(min_size=MIN_DUPLICATE_LENGTH, max_size=200))
    def test_reflexive(self, text: str):
        """A sufficiently long text is always a duplicate of itself."""
        assume(len(normalize_text(text)) >= MIN_DUPLICATE_LENGTH)
        assert is_duplicate(text, [text]) is True

    @given(text=st.text(min_size=0, max_size=MIN_DUPLICATE_LENGTH - 1))
    def test_short_text_never_duplicate(self, text: str):
        assert is_duplicate(text, [text]) is False

    @given(text=_nasty_text)
    def test_empty_sent_list_never_duplicate(self, text: str):
        assert is_duplicate(text, []) is False

    @given(
        a=st.text(min_size=MIN_DUPLICATE_LENGTH, max_size=100),
        b=st.text(min_size=MIN_DUPLICATE_LENGTH, max_size=100),
    )
    def test_symmetric(self, a: str, b: str):
        assert is_duplicate(a, [b]) == is_duplicate(b, [a])


class TestMessageDeduplicatorProperties:
    @given(text=st.text(min_size=MIN_DUPLICATE_LENGTH, max_size=200))
    def test_tracked_message_detected(self, text: str):
        assume(len(normalize_text(text)) >= MIN_DUPLICATE_LENGTH)
        dd = MessageDeduplicator()
        dd.track_sent(text)
        assert dd.is_duplicate(text) is True

    @given(text=st.text(min_size=MIN_DUPLICATE_LENGTH, max_size=200))
    def test_discarded_pending_not_duplicate(self, text: str):
        dd = MessageDeduplicator()
        dd.track_pending("call-1", text)
        dd.discard_pending("call-1")
        assert dd.is_duplicate(text) is False

    @given(text=st.text(min_size=MIN_DUPLICATE_LENGTH, max_size=200))
    def test_committed_pending_is_duplicate(self, text: str):
        assume(len(normalize_text(text)) >= MIN_DUPLICATE_LENGTH)
        dd = MessageDeduplicator()
        dd.track_pending("call-1", text)
        dd.commit_pending("call-1")
        assert dd.is_duplicate(text) is True


# --- Unit tests: normalize_text_for_comparison ---


class TestNormalizeText:
    def test_trims_whitespace(self):
        from aineko.messaging_dedupe import normalize_text

        assert normalize_text("  hello world  ") == "hello world"

    def test_lowercases(self):
        from aineko.messaging_dedupe import normalize_text

        assert normalize_text("Hello World") == "hello world"

    def test_strips_emoji(self):
        from aineko.messaging_dedupe import normalize_text

        assert normalize_text("hello 🎉 world 🚀") == "hello world"

    def test_collapses_whitespace(self):
        from aineko.messaging_dedupe import normalize_text

        assert normalize_text("hello   world\n\tfoo") == "hello world foo"

    def test_combined_normalization(self):
        from aineko.messaging_dedupe import normalize_text

        assert normalize_text("  Hello 🎉  World  ") == "hello world"

    def test_empty_string(self):
        from aineko.messaging_dedupe import normalize_text

        assert normalize_text("") == ""

    def test_only_emoji(self):
        from aineko.messaging_dedupe import normalize_text

        assert normalize_text("🎉🚀✨") == ""


# --- Unit tests: is_duplicate ---


class TestIsDuplicate:
    def test_no_sent_texts(self):
        from aineko.messaging_dedupe import is_duplicate

        assert is_duplicate("hello world foo", []) is False

    def test_exact_match(self):
        from aineko.messaging_dedupe import is_duplicate

        assert is_duplicate("hello world foo bar", ["hello world foo bar"]) is True

    def test_substring_a_in_b(self):
        """Sent text is a substring of new text → duplicate."""
        from aineko.messaging_dedupe import is_duplicate

        assert (
            is_duplicate(
                "awesome, 387 docs indexed. semantic search is live.",
                ["387 docs indexed. semantic search is live."],
            )
            is True
        )

    def test_substring_b_in_a(self):
        """New text is a substring of sent text → duplicate."""
        from aineko.messaging_dedupe import is_duplicate

        assert (
            is_duplicate(
                "387 docs indexed",
                ["awesome, 387 docs indexed. semantic search is live."],
            )
            is True
        )

    def test_no_match(self):
        from aineko.messaging_dedupe import is_duplicate

        assert (
            is_duplicate(
                "completely different text here",
                ["hello world foo bar baz"],
            )
            is False
        )

    def test_short_text_ignored(self):
        """Texts shorter than 10 chars (normalized) are never duplicates."""
        from aineko.messaging_dedupe import is_duplicate

        assert is_duplicate("hi", ["hi"]) is False
        assert is_duplicate("short", ["short"]) is False

    def test_short_sent_text_ignored(self):
        from aineko.messaging_dedupe import is_duplicate

        assert is_duplicate("this is a long message", ["ok"]) is False

    def test_normalization_applied(self):
        """Comparison uses normalized form — case/emoji/whitespace don't matter."""
        from aineko.messaging_dedupe import is_duplicate

        assert (
            is_duplicate(
                "Hello 🎉 World this is great",
                ["hello world this is great"],
            )
            is True
        )

    def test_multiple_sent_texts(self):
        from aineko.messaging_dedupe import is_duplicate

        assert (
            is_duplicate(
                "the quick brown fox jumps",
                ["something else entirely", "the quick brown fox jumps"],
            )
            is True
        )

    def test_empty_new_text(self):
        from aineko.messaging_dedupe import is_duplicate

        assert is_duplicate("", ["hello world foo bar"]) is False


# --- Integration tests: MessageDeduplicator ---


class TestMessageDeduplicator:
    def test_track_and_check(self):
        from aineko.messaging_dedupe import MessageDeduplicator

        d = MessageDeduplicator()
        d.track_sent("hello world this is a test message")
        assert d.is_duplicate("hello world this is a test message") is True
        assert d.is_duplicate("something totally different here") is False

    def test_track_failed_not_committed(self):
        """Failed sends should not be tracked."""
        from aineko.messaging_dedupe import MessageDeduplicator

        d = MessageDeduplicator()
        d.track_pending("id1", "hello world this is a test")
        # Not committed — should not suppress
        assert d.is_duplicate("hello world this is a test") is False
        # Now commit
        d.commit_pending("id1")
        assert d.is_duplicate("hello world this is a test") is True

    def test_discard_pending(self):
        from aineko.messaging_dedupe import MessageDeduplicator

        d = MessageDeduplicator()
        d.track_pending("id1", "hello world this is a test")
        d.discard_pending("id1")
        d.commit_pending("id1")  # no-op since discarded
        assert d.is_duplicate("hello world this is a test") is False

    def test_max_tracked_texts(self):
        """Should not grow unbounded."""
        from aineko.messaging_dedupe import MessageDeduplicator

        d = MessageDeduplicator(max_tracked=5)
        for i in range(10):
            d.track_sent(f"message number {i} is long enough")
        assert len(d._sent_normalized) <= 5

    def test_oldest_evicted(self):
        from aineko.messaging_dedupe import MessageDeduplicator

        d = MessageDeduplicator(max_tracked=3)
        d.track_sent("first message that is long enough")
        d.track_sent("second message that is long enough")
        d.track_sent("third message that is long enough")
        d.track_sent("fourth message that is long enough")
        # First should be evicted
        assert d.is_duplicate("first message that is long enough") is False
        assert d.is_duplicate("fourth message that is long enough") is True


# --- Integration tests: chat_loop deduplication ---


class TestChatLoopDedup:
    """Test that the chat_loop properly deduplicates messages."""

    @pytest.mark.asyncio
    async def test_intermediate_skipped_when_send_message_has_same_text(self):
        """When model returns content + send_message with same text,
        only one message should be sent."""
        from unittest.mock import MagicMock
        from aineko.kimi.client import KimiClient, ChatResponse, ToolCall
        from aineko.tools.registry import ToolDef, ToolRegistry

        client = KimiClient(
            MagicMock(
                api_key="k",
                base_url="http://x",
                model="k2p5",
                user_agent="t",
                max_context_tokens=1000,
                thinking=False,
                temperature=1.0,
                top_p=0.95,
                max_tokens=100,
            )
        )

        call_count = 0

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="hello world this is a progress update",
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="send_message",
                            arguments={
                                "message": "hello world this is a progress update"
                            },
                        ),
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done", finish_reason="end_turn")

        client.chat = mock_chat

        send_count = 0

        async def fake_send(message=""):
            nonlocal send_count
            send_count += 1
            return "sent"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="send_message",
                description="send",
                parameters={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                handler=fake_send,
            )
        )

        messages = [{"role": "user", "content": "hi"}]
        await client.chat_loop(messages, registry)

        # Should only send once, not twice
        assert send_count == 1

    @pytest.mark.asyncio
    async def test_multiple_send_message_calls_deduplicated(self):
        """When model returns two send_message calls with same text,
        only one should actually send."""
        from unittest.mock import MagicMock
        from aineko.kimi.client import KimiClient, ChatResponse, ToolCall
        from aineko.tools.registry import ToolDef, ToolRegistry

        client = KimiClient(
            MagicMock(
                api_key="k",
                base_url="http://x",
                model="k2p5",
                user_agent="t",
                max_context_tokens=1000,
                thinking=False,
                temperature=1.0,
                top_p=0.95,
                max_tokens=100,
            )
        )

        call_count = 0

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="progress update for the user",
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="send_message",
                            arguments={"message": "progress update for the user"},
                        ),
                        ToolCall(
                            id="t2",
                            name="send_message",
                            arguments={"message": "progress update for the user"},
                        ),
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done", finish_reason="end_turn")

        client.chat = mock_chat

        send_count = 0

        async def fake_send(message=""):
            nonlocal send_count
            send_count += 1
            return "sent"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="send_message",
                description="send",
                parameters={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                handler=fake_send,
            )
        )

        messages = [{"role": "user", "content": "hi"}]
        await client.chat_loop(messages, registry)

        assert send_count == 1

    @pytest.mark.asyncio
    async def test_different_send_messages_not_deduplicated(self):
        """Two send_message calls with genuinely different content
        should both go through."""
        from unittest.mock import MagicMock
        from aineko.kimi.client import KimiClient, ChatResponse, ToolCall
        from aineko.tools.registry import ToolDef, ToolRegistry

        client = KimiClient(
            MagicMock(
                api_key="k",
                base_url="http://x",
                model="k2p5",
                user_agent="t",
                max_context_tokens=1000,
                thinking=False,
                temperature=1.0,
                top_p=0.95,
                max_tokens=100,
            )
        )

        call_count = 0

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="send_message",
                            arguments={
                                "message": "first message about topic A in detail"
                            },
                        ),
                        ToolCall(
                            id="t2",
                            name="send_message",
                            arguments={
                                "message": "second message about topic B in detail"
                            },
                        ),
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done", finish_reason="end_turn")

        client.chat = mock_chat

        send_count = 0

        async def fake_send(message=""):
            nonlocal send_count
            send_count += 1
            return "sent"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="send_message",
                description="send",
                parameters={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                handler=fake_send,
            )
        )

        messages = [{"role": "user", "content": "hi"}]
        await client.chat_loop(messages, registry)

        assert send_count == 2

    @pytest.mark.asyncio
    async def test_intermediate_sent_when_no_send_message_tool(self):
        """When model returns content + non-send tools, intermediate
        content should still be sent."""
        from unittest.mock import MagicMock
        from aineko.kimi.client import KimiClient, ChatResponse, ToolCall
        from aineko.tools.registry import ToolDef, ToolRegistry

        client = KimiClient(
            MagicMock(
                api_key="k",
                base_url="http://x",
                model="k2p5",
                user_agent="t",
                max_context_tokens=1000,
                thinking=False,
                temperature=1.0,
                top_p=0.95,
                max_tokens=100,
            )
        )

        call_count = 0

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="let me check that for you right now",
                    tool_calls=[
                        ToolCall(id="t1", name="bash", arguments={"command": "ls"}),
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done", finish_reason="end_turn")

        client.chat = mock_chat

        send_count = 0

        async def fake_send(message=""):
            nonlocal send_count
            send_count += 1
            return "sent"

        async def fake_bash(command="", **kw):
            return "file.txt"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="send_message",
                description="send",
                parameters={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                handler=fake_send,
            )
        )
        registry.register(
            ToolDef(
                name="bash",
                description="run",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                handler=fake_bash,
            )
        )

        messages = [{"role": "user", "content": "hi"}]
        await client.chat_loop(messages, registry)

        # Intermediate content should have been sent
        assert send_count == 1
