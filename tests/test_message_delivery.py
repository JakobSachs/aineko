"""Tests for message delivery logic — send_message tool + auto-send.

Regression: when the model called send_message as a progress update during a
tool chain, the final response was never delivered because the auto-send path
was skipped. The fix: always auto-send the final response regardless of whether
send_message was called during the loop.
"""

import pytest


# Minimal reproduction of the delivery logic from app.py handle_message
async def simulate_delivery(
    tool_calls_sequence: list[list[dict]],
    final_content: str,
) -> dict:
    """Simulate the message delivery flow.

    Args:
        tool_calls_sequence: list of rounds. Each round is a list of
            tool calls the model makes: [{"name": ..., "args": ...}, ...]
        final_content: the model's final text response after all tool rounds.

    Returns:
        {"sent_via_tool": [...], "auto_sent": [...], "saved_history": str}
    """
    sent_via_tool: list[str] = []
    auto_sent: list[str] = []

    async def _send_message_tool(message: str, room: str = "") -> str:
        sent_via_tool.append(message)
        return "sent"

    # Simulate tool execution (just track send_message calls)
    for round_calls in tool_calls_sequence:
        for call in round_calls:
            if call["name"] == "send_message":
                await _send_message_tool(**call["args"])

    # Auto-send final response
    if final_content:
        auto_sent.append(final_content)

    # Build saved history (the fixed logic)
    all_visible = sent_via_tool.copy()
    if final_content:
        all_visible.append(final_content)
    visible = "\n".join(all_visible) if all_visible else ""

    return {
        "sent_via_tool": sent_via_tool,
        "auto_sent": auto_sent,
        "saved_history": visible,
    }


@pytest.mark.asyncio
async def test_final_response_sent_even_after_send_message_tool():
    """Regression: model sends progress update via tool, then has final content.
    Both should reach the user."""
    result = await simulate_delivery(
        tool_calls_sequence=[
            [{"name": "send_message", "args": {"message": "looking into it"}}],
            # web_search would happen here, but we skip non-send_message tools
        ],
        final_content="here are the results: fungi connect trees underground",
    )

    assert result["sent_via_tool"] == ["looking into it"]
    assert result["auto_sent"] == [
        "here are the results: fungi connect trees underground"
    ]
    assert "looking into it" in result["saved_history"]
    assert "fungi connect trees" in result["saved_history"]


@pytest.mark.asyncio
async def test_auto_send_works_without_send_message_tool():
    """Normal flow: model never calls send_message, response is auto-sent."""
    result = await simulate_delivery(
        tool_calls_sequence=[],
        final_content="just a normal reply",
    )

    assert result["sent_via_tool"] == []
    assert result["auto_sent"] == ["just a normal reply"]
    assert result["saved_history"] == "just a normal reply"


@pytest.mark.asyncio
async def test_no_content_no_send():
    """Empty final response sends nothing."""
    result = await simulate_delivery(
        tool_calls_sequence=[],
        final_content="",
    )

    assert result["sent_via_tool"] == []
    assert result["auto_sent"] == []
    assert result["saved_history"] == ""


@pytest.mark.asyncio
async def test_multiple_progress_updates_plus_final():
    """Model sends multiple progress updates then a final answer."""
    result = await simulate_delivery(
        tool_calls_sequence=[
            [{"name": "send_message", "args": {"message": "searching..."}}],
            [
                {
                    "name": "send_message",
                    "args": {"message": "found 5 results, analyzing"},
                }
            ],
        ],
        final_content="here is the summary",
    )

    assert len(result["sent_via_tool"]) == 2
    assert result["auto_sent"] == ["here is the summary"]
    # History contains everything
    assert "searching..." in result["saved_history"]
    assert "found 5 results" in result["saved_history"]
    assert "here is the summary" in result["saved_history"]
