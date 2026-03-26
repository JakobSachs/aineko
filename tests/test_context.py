"""Tests for sliding window context trimmer."""

import pytest

from aineko.context import estimate_tokens, trim_messages


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


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
