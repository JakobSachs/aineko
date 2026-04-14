"""Tests for reply-body reformatting passed to the LLM."""

from __future__ import annotations

from types import SimpleNamespace

from aineko.matrix.client import _format_reply_body


def _make_event(body: str, in_reply_to: str | None) -> SimpleNamespace:
    content: dict = {"body": body, "msgtype": "m.text"}
    if in_reply_to is not None:
        content["m.relates_to"] = {"m.in_reply_to": {"event_id": in_reply_to}}
    return SimpleNamespace(
        body=body, sender="@test:example.com", source={"content": content}
    )


def test_non_reply_body_unchanged() -> None:
    event = _make_event("just a regular message", in_reply_to=None)
    assert _format_reply_body(event) == "just a regular message"


def test_reply_with_fallback_is_reformatted() -> None:
    body = (
        "> <@alice:example.com> hey whats the plan for tomorrow\n"
        "\n"
        "lets meet at 10"
    )
    event = _make_event(body, in_reply_to="$orig")
    out = _format_reply_body(event)
    assert out == (
        '[in reply to @alice:example.com: "hey whats the plan for tomorrow"]\n'
        "lets meet at 10"
    )


def test_reply_with_multiline_original() -> None:
    body = "> <@bob:example.com> first line\n" "> second line\n" "\n" "gotcha"
    event = _make_event(body, in_reply_to="$orig")
    out = _format_reply_body(event)
    assert out == (
        '[in reply to @bob:example.com: "first line\nsecond line"]\n' "gotcha"
    )


def test_reply_without_sender_angle_brackets() -> None:
    body = "> plain quoted line\n\nmy reply"
    event = _make_event(body, in_reply_to="$orig")
    out = _format_reply_body(event)
    assert out == '[in reply to: "plain quoted line"]\nmy reply'


def test_reply_fallback_missing_but_flagged_as_reply() -> None:
    # Defensive: m.in_reply_to is set but body has no quote block.
    # We leave the body alone rather than produce garbage.
    event = _make_event("no fallback here", in_reply_to="$orig")
    assert _format_reply_body(event) == "no fallback here"


def test_manual_quote_without_in_reply_to_is_not_reformatted() -> None:
    # User typed `>` manually — not an actual Matrix reply. Leave as-is.
    body = "> some quote\n\nmy text"
    event = _make_event(body, in_reply_to=None)
    assert _format_reply_body(event) == body
