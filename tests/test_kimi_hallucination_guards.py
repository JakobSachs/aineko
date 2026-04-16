"""Behavior tests for the guards that prevent hallucinated pre-tool content.

These tests describe observable behavior — they would fail if the guards were
removed or regressed, regardless of the internal implementation.

Context (2026-04-16): Kimi k2p5 was observed returning `text + tool_use` in the
same response. The text was a fabricated tool-result-shaped preamble written
*before* the tool ran. Two leaks:
  1. The text was streamed to the user via `on_intermediate`.
  2. The text was fed back into the conversation as the assistant message,
     teaching the model (in-context) that this pattern is correct.

Fix: drop pre-tool text from both channels. Final content (no tool_use) is
still delivered normally.

Separate fix: thinking blocks must be replayed verbatim with their `signature`
field. The previous code concatenated `thinking` strings and dropped signatures,
which breaks the Anthropic API contract for multi-turn thinking.
"""

from unittest.mock import MagicMock

import pytest

from aineko.kimi.client import ChatResponse, KimiClient, ToolCall
from aineko.tools.registry import ToolDef, ToolRegistry


def _settings() -> MagicMock:
    return MagicMock(
        api_key="test",
        base_url="https://api.kimi.com/coding/v1",
        model="k2p5",
        user_agent="test",
        max_context_tokens=262144,
        thinking=True,
        temperature=1.0,
        top_p=0.95,
        max_tokens=32000,
    )


def _registry() -> ToolRegistry:
    async def fake(**kw: object) -> str:
        return "real tool output: nothing here"

    reg = ToolRegistry()
    reg.register(
        ToolDef(
            name="glob",
            description="",
            parameters={
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": [],
            },
            handler=fake,
        )
    )
    return reg


class TestHallucinatedPreambleIsNotStreamed:
    """When the model emits text alongside tool_use, the user must not see it.

    The text is generated before the tool ran; it's a guess, not a result.
    Streaming it to Matrix leaks fabrications to the user.
    """

    @pytest.mark.asyncio
    async def test_pretool_content_does_not_reach_on_intermediate(self) -> None:
        client = KimiClient(_settings())
        seen: list[str] = []

        call_count = 0

        async def fake_chat(msgs, tools=None):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Mimics the observed bug: confident fake listing + tool call.
                return ChatResponse(
                    content="found 4 files: fake1.md, fake2.md, fake3.md, fake4.md",
                    tool_calls=[
                        ToolCall(id="t1", name="glob", arguments={"pattern": "*.md"})
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(
                content="real answer from tool", finish_reason="end_turn"
            )

        client.chat = fake_chat  # type: ignore[assignment]

        async def on_intermediate(text: str) -> None:
            seen.append(text)

        resp = await client.chat_loop(
            [{"role": "user", "content": "list files"}],
            _registry(),
            on_intermediate=on_intermediate,
        )

        assert "fake1.md" not in "".join(
            seen
        ), "Hallucinated pre-tool content leaked to user via on_intermediate"
        assert resp.content == "real answer from tool"

    @pytest.mark.asyncio
    async def test_final_content_still_delivered(self) -> None:
        """Content in a response *without* tool_use is the real answer — deliver it."""
        client = KimiClient(_settings())

        async def fake_chat(msgs, tools=None):  # type: ignore[no-untyped-def]
            return ChatResponse(content="the real answer", finish_reason="end_turn")

        client.chat = fake_chat  # type: ignore[assignment]

        seen: list[str] = []

        async def on_intermediate(text: str) -> None:
            seen.append(text)

        resp = await client.chat_loop(
            [{"role": "user", "content": "hi"}],
            _registry(),
            on_intermediate=on_intermediate,
        )

        assert resp.content == "the real answer"


class TestHallucinatedPreambleIsNotFedBack:
    """When text arrives with tool_use, it must not enter the replayed history.

    Keeping it in the assistant history teaches the model (via in-context
    pattern-matching) that confidently narrating fake results before tools
    run is correct. That's exactly how the bug amplifies across turns.
    """

    @pytest.mark.asyncio
    async def test_pretool_text_not_in_assistant_history(self) -> None:
        client = KimiClient(_settings())
        messages: list[dict] = [{"role": "user", "content": "list files"}]

        call_count = 0

        async def fake_chat(msgs, tools=None):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="HALLUCINATED_PREAMBLE_12345",
                    tool_calls=[
                        ToolCall(id="t1", name="glob", arguments={"pattern": "*"})
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done", finish_reason="end_turn")

        client.chat = fake_chat  # type: ignore[assignment]
        await client.chat_loop(messages, _registry())

        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        for m in assistant_msgs:
            content = m["content"]
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") == "text":
                    assert "HALLUCINATED_PREAMBLE_12345" not in block.get(
                        "text", ""
                    ), "Pre-tool text leaked into the replayed assistant history"

    @pytest.mark.asyncio
    async def test_tool_use_block_still_replayed(self) -> None:
        """The fix must not drop the tool_use block itself."""
        client = KimiClient(_settings())
        messages: list[dict] = [{"role": "user", "content": "list files"}]

        call_count = 0

        async def fake_chat(msgs, tools=None):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="preamble",
                    tool_calls=[
                        ToolCall(id="t1", name="glob", arguments={"pattern": "*"})
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done", finish_reason="end_turn")

        client.chat = fake_chat  # type: ignore[assignment]
        await client.chat_loop(messages, _registry())

        assistant_with_tool = next(
            m
            for m in messages
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        )
        tool_use_block = next(
            b for b in assistant_with_tool["content"] if b["type"] == "tool_use"
        )
        assert tool_use_block["id"] == "t1"
        assert tool_use_block["name"] == "glob"


class TestThinkingSignatureReplay:
    """Thinking blocks from the API must be replayed verbatim with signatures.

    Anthropic's multi-turn thinking requires the `signature` field to round-trip
    unchanged. Stripping or recomputing it breaks the next turn.
    """

    @pytest.mark.asyncio
    async def test_thinking_signature_preserved_in_parsed_response(self) -> None:
        client = KimiClient(_settings())

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "reasoning text",
                            "signature": "sig-abc-123",
                        },
                        {"type": "text", "text": "answer"},
                    ],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }

        async def fake_post(url, **kw):  # type: ignore[no-untyped-def]
            return FakeResp()

        client._http.post = fake_post  # type: ignore[assignment]
        resp = await client.chat([{"role": "user", "content": "hi"}])

        assert len(resp.thinking_blocks) == 1
        block = resp.thinking_blocks[0]
        assert block["type"] == "thinking"
        assert block["thinking"] == "reasoning text"
        assert (
            block.get("signature") == "sig-abc-123"
        ), "Thinking signature was stripped — API will reject the replay"

    @pytest.mark.asyncio
    async def test_thinking_signature_replayed_verbatim_in_next_turn(self) -> None:
        """After a tool round-trip, the assistant message must carry the original signature."""
        client = KimiClient(_settings())
        messages: list[dict] = [{"role": "user", "content": "hi"}]

        call_count = 0

        async def fake_chat(msgs, tools=None):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    thinking_blocks=[
                        {
                            "type": "thinking",
                            "thinking": "hmm let me check",
                            "signature": "SIG_DEADBEEF",
                        }
                    ],
                    tool_calls=[
                        ToolCall(id="t1", name="glob", arguments={"pattern": "*"})
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="ok", finish_reason="end_turn")

        client.chat = fake_chat  # type: ignore[assignment]
        await client.chat_loop(messages, _registry())

        assistant = next(
            m
            for m in messages
            if m.get("role") == "assistant" and isinstance(m.get("content"), list)
        )
        thinking = next(b for b in assistant["content"] if b.get("type") == "thinking")
        assert thinking.get("signature") == "SIG_DEADBEEF"
        assert thinking.get("thinking") == "hmm let me check"
