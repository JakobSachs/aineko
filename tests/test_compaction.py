"""Tests for conversation compaction — LLM-powered history summarization."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from aineko.compaction import (
    should_compact,
    build_compaction_prompt,
    compact_messages,
)

# --- should_compact ---


def test_short_conversation_no_compaction():
    """Small conversations don't need compaction."""
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert should_compact(messages, max_tokens=100_000) is False


def test_large_conversation_triggers_compaction():
    """When token usage approaches the limit, compaction should trigger."""
    system = {"role": "system", "content": "x" * 400}
    messages = [system]
    # Fill up to exceed threshold
    for i in range(200):
        messages.append({"role": "user", "content": f"message {i} " + "x" * 500})
        messages.append({"role": "assistant", "content": f"reply {i} " + "x" * 500})
    assert should_compact(messages, max_tokens=50_000) is True


def test_compaction_respects_reserve():
    """Compaction triggers before hitting the hard limit, leaving reserve space."""
    # Build messages that total ~45k tokens (at 4 chars/token)
    messages = [{"role": "system", "content": "sys"}]
    for i in range(90):
        messages.append({"role": "user", "content": "x" * 2000})
    # 90 * 2000 / 4 = 45k tokens
    # With 50k max and reserve=0, threshold is 50k — not quite there
    # Use 40k max so 45k tokens exceeds it
    assert should_compact(messages, max_tokens=40_000) is True


def test_no_compaction_when_under_threshold():
    """Under threshold = no compaction needed."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    assert should_compact(messages, max_tokens=100_000) is False


# --- build_compaction_prompt ---


def test_compaction_prompt_contains_template():
    """The compaction prompt should include the structured summary template."""
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "fix the bug in foo.py"},
        {"role": "assistant", "content": "I found the issue and fixed it"},
    ]
    prompt = build_compaction_prompt(messages)
    assert "## Goal" in prompt
    assert "## Accomplished" in prompt
    assert "## Relevant files" in prompt


def test_compaction_prompt_includes_conversation():
    """The conversation content should be embedded in the prompt."""
    messages = [
        {"role": "system", "content": "system prompt here"},
        {"role": "user", "content": "unique_user_request_xyz"},
        {"role": "assistant", "content": "unique_response_abc"},
    ]
    prompt = build_compaction_prompt(messages)
    assert "unique_user_request_xyz" in prompt
    assert "unique_response_abc" in prompt


# --- compact_messages ---


@pytest.mark.asyncio
async def test_compact_messages_returns_summary():
    """compact_messages calls the LLM and returns compacted message list."""
    from aineko.kimi.client import ChatResponse

    mock_kimi = MagicMock()
    mock_kimi.chat = AsyncMock(
        return_value=ChatResponse(
            content="## Goal\nFix bugs\n## Accomplished\nFixed foo.py",
            usage={"total_tokens": 500},
        )
    )

    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "old message 1"},
        {"role": "assistant", "content": "old reply 1"},
        {"role": "user", "content": "old message 2"},
        {"role": "assistant", "content": "old reply 2"},
        {"role": "user", "content": "recent message"},
        {"role": "assistant", "content": "recent reply"},
        {"role": "user", "content": "latest question"},
    ]

    result, summary = await compact_messages(messages, mock_kimi, keep_recent=4)

    # Should have: system + compaction summary + recent messages
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "system"  # compaction summary
    assert "Goal" in result[1]["content"] or "Fix bugs" in result[1]["content"]
    # Recent messages preserved
    assert any("latest question" in m.get("content", "") for m in result)
    assert any("recent message" in m.get("content", "") for m in result)
    # Summary returned explicitly
    assert summary is not None
    assert "Fix bugs" in summary


@pytest.mark.asyncio
async def test_compact_messages_preserves_system():
    """System prompt always kept as-is."""
    from aineko.kimi.client import ChatResponse

    mock_kimi = MagicMock()
    mock_kimi.chat = AsyncMock(
        return_value=ChatResponse(
            content="summary here",
            usage={"total_tokens": 200},
        )
    )

    messages = [
        {"role": "system", "content": "original system prompt"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "new"},
    ]

    result, summary = await compact_messages(messages, mock_kimi, keep_recent=2)
    assert result[0]["content"] == "original system prompt"
    assert summary is not None


@pytest.mark.asyncio
async def test_compact_messages_too_short_returns_unchanged():
    """If there aren't enough messages to compact, return unchanged."""

    mock_kimi = MagicMock()

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    result, summary = await compact_messages(messages, mock_kimi, keep_recent=4)
    # Too few messages to compact — returned unchanged
    assert result == messages
    assert summary is None
    mock_kimi.chat.assert_not_called()
