"""Tests for sliding window context trimmer."""

from aineko.context import estimate_tokens, trim_messages


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_estimate_tokens_none():
    assert estimate_tokens(None) == 0


def test_estimate_tokens_list_text_blocks():
    blocks = [
        {"type": "text", "text": "a" * 400},
        {"type": "text", "text": "b" * 400},
    ]
    assert estimate_tokens(blocks) == 200


def test_estimate_tokens_list_image_block():
    blocks = [{"type": "image", "source": {"type": "base64", "data": "..."}}]
    assert estimate_tokens(blocks) == 1000


def test_estimate_tokens_list_mixed():
    blocks = [
        {"type": "text", "text": "a" * 400},
        {"type": "image", "source": {"type": "base64", "data": "..."}},
    ]
    assert estimate_tokens(blocks) == 100 + 1000


def test_estimate_tokens_tool_result_blocks():
    # Tool result blocks use "content" key not "text"
    blocks = [{"type": "tool_result", "tool_use_id": "t1", "content": "a" * 400}]
    assert estimate_tokens(blocks) == 100


def test_trim_handles_list_content():
    """Messages with list content (tool results) are counted correctly."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
    ]
    result = trim_messages(msgs, max_tokens=10000)
    assert len(result) == 3


def test_trim_list_content_counts_toward_budget():
    """Large list-content messages are dropped when over budget."""
    big_block = [{"type": "text", "text": "x" * 4000}]  # ~1000 tokens
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": big_block},
        {"role": "assistant", "content": "recent"},
    ]
    result = trim_messages(msgs, max_tokens=50)
    assert result[0]["role"] == "system"
    # The big block message should be dropped, only recent kept
    contents = [m["content"] for m in result]
    assert "recent" in contents
    assert big_block not in contents


def test_trim_keeps_all_when_under_budget():
    msgs = [
        {"role": "system", "content": "you are a bot"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = trim_messages(msgs, max_tokens=10000)
    assert len(result) == 3


def test_trim_drops_oldest_first():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old message " * 50},
        {"role": "assistant", "content": "old reply " * 50},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "recent reply"},
    ]
    # Budget tight enough to drop the old messages but keep recent
    result = trim_messages(msgs, max_tokens=30)
    assert result[0]["role"] == "system"
    assert result[-1]["content"] == "recent reply"
    assert "old message" not in str(result)


def test_trim_always_keeps_system():
    msgs = [
        {"role": "system", "content": "important system prompt"},
        {"role": "user", "content": "x" * 10000},
    ]
    result = trim_messages(msgs, max_tokens=50)
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "important system prompt"


def test_trim_empty_list():
    assert trim_messages([], max_tokens=1000) == []
