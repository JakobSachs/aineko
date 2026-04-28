"""Tests for Anthropic Messages API format (Kimi Coding endpoint)."""

import asyncio

import pytest
from hypothesis import given, settings as hsettings
from hypothesis import strategies as st
from unittest.mock import MagicMock

from aineko.kimi.client import KimiClient, ChatResponse, ToolCall
from aineko.tools.registry import ToolDef, ToolRegistry


def _make_settings(**overrides):
    defaults = {
        "api_key": "test-key",
        "base_url": "https://api.kimi.com/coding/v1",
        "model": "k2p5",
        "user_agent": "claude-code/0.1.0",
        "max_context_tokens": 262144,
        "thinking": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 32000,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


# --- Request format ---


class TestRequestFormat:
    """Verify we send Anthropic Messages API format."""

    @pytest.mark.asyncio
    async def test_posts_to_messages_endpoint(self):
        """Should POST to /messages, not /chat/completions."""
        client = KimiClient(_make_settings())
        posted_url = None

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }

        async def capture_post(url, **kwargs):
            nonlocal posted_url
            posted_url = url
            return FakeResp()

        client._http.post = capture_post
        await client.chat([{"role": "user", "content": "hi"}])
        assert posted_url == "/messages"

    @pytest.mark.asyncio
    async def test_system_prompt_extracted_to_top_level(self):
        """System messages should become the top-level 'system' field."""
        client = KimiClient(_make_settings())
        captured_payload = {}

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }

        async def capture_post(url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeResp()

        client._http.post = capture_post
        await client.chat(
            [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hello"},
            ]
        )

        # System should be top-level, not in messages
        assert captured_payload.get("system") == "you are helpful"
        msg_roles = [m["role"] for m in captured_payload["messages"]]
        assert "system" not in msg_roles
        assert "user" in msg_roles

    @pytest.mark.asyncio
    async def test_thinking_includes_budget_tokens(self):
        """Thinking config should include budget_tokens."""
        client = KimiClient(_make_settings(thinking=True))
        captured_payload = {}

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }

        async def capture_post(url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeResp()

        client._http.post = capture_post
        await client.chat([{"role": "user", "content": "hi"}])

        thinking = captured_payload.get("thinking", {})
        assert thinking["type"] == "enabled"
        assert "budget_tokens" in thinking
        assert thinking["budget_tokens"] == 16000

    @pytest.mark.asyncio
    async def test_tools_use_input_schema(self):
        """Tool definitions should use 'input_schema', not 'parameters'."""
        client = KimiClient(_make_settings())
        captured_payload = {}

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }

        async def capture_post(url, json=None, **kwargs):
            captured_payload.update(json or {})
            return FakeResp()

        client._http.post = capture_post

        registry = ToolRegistry()

        async def fake(**kw):
            return "ok"

        registry.register(
            ToolDef(
                name="bash",
                description="run commands",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                handler=fake,
            )
        )

        await client.chat([{"role": "user", "content": "hi"}], tools=registry)

        tools = captured_payload.get("tools", [])
        assert len(tools) == 1
        assert "input_schema" in tools[0]
        assert "parameters" not in tools[0]
        assert tools[0]["name"] == "bash"


# --- Response parsing ---


class TestResponseParsing:
    """Verify we parse Anthropic response format correctly."""

    @pytest.mark.asyncio
    async def test_parse_text_response(self):
        client = KimiClient(_make_settings())

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello world"}],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }

        async def fake_post(url, **kw):
            return FakeResp()

        client._http.post = fake_post

        result = await client.chat([{"role": "user", "content": "hi"}])
        assert result.content == "hello world"
        assert result.finish_reason == "end_turn"
        assert len(result.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_parse_tool_use_response(self):
        client = KimiClient(_make_settings())

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "let me check"},
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "bash",
                            "input": {"command": "ls"},
                        },
                    ],
                    "model": "k2p5",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 20, "output_tokens": 15},
                }

        async def fake_post(url, **kw):
            return FakeResp()

        client._http.post = fake_post

        result = await client.chat([{"role": "user", "content": "list files"}])
        assert result.content == "let me check"
        assert result.finish_reason == "tool_use"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_123"
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[0].arguments == {"command": "ls"}

    @pytest.mark.asyncio
    async def test_parse_thinking_response(self):
        client = KimiClient(_make_settings())

        class FakeResp:
            status_code = 200
            text = ""

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I should analyze this..."},
                        {"type": "text", "text": "here's my answer"},
                    ],
                    "model": "k2p5",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                }

        async def fake_post(url, **kw):
            return FakeResp()

        client._http.post = fake_post

        result = await client.chat([{"role": "user", "content": "think about this"}])
        assert result.content == "here's my answer"
        assert result.reasoning_content == "I should analyze this..."


# --- Tool result format in chat_loop ---


class TestToolResultFormat:
    """Verify tool results are sent as Anthropic tool_result blocks."""

    @pytest.mark.asyncio
    async def test_tool_result_sent_as_content_block(self):
        """Tool results should be user messages with tool_result content blocks."""
        client = KimiClient(_make_settings())
        messages: list[dict] = [{"role": "user", "content": "list files"}]

        call_count = 0

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="toolu_1", name="bash", arguments={"command": "ls"})
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done", finish_reason="end_turn")

        client.chat = mock_chat

        async def fake_bash(**kw):
            return "file.txt"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="bash",
                description="run",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=fake_bash,
            )
        )

        await client.chat_loop(messages, registry)

        # Find the tool result message
        tool_result_msgs = [
            m
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), list)
        ]
        assert len(tool_result_msgs) >= 1
        content = tool_result_msgs[0]["content"]
        assert any(
            block.get("type") == "tool_result" and block.get("tool_use_id") == "toolu_1"
            for block in content
        )

    @pytest.mark.asyncio
    async def test_assistant_tool_use_preserved_in_messages(self):
        """Assistant messages with tool_use should preserve content blocks."""
        client = KimiClient(_make_settings())
        messages: list[dict] = [{"role": "user", "content": "hi"}]

        call_count = 0

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="thinking...",
                    tool_calls=[
                        ToolCall(id="toolu_1", name="bash", arguments={"command": "ls"})
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done")

        client.chat = mock_chat

        async def fake_bash(**kw):
            return "ok"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="bash",
                description="",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=fake_bash,
            )
        )

        await client.chat_loop(messages, registry)

        # Find assistant message — should have content blocks, not just a string
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) >= 1
        content = assistant_msgs[0]["content"]
        assert isinstance(content, list)
        types = [block["type"] for block in content]
        assert "tool_use" in types


# --- Interjection during final (end_turn) response ---


class TestInterjectionRace:
    """Regression: a user message that lands in the interject queue while the
    LLM is producing its final (no-tool-calls) response must not be silently
    dropped on the floor when chat_loop returns.

    Real-world incident (2026-04-28 09:59:56): user sent a message ~100ms
    before the final end_turn arrived; queue.py logged `interject_enqueue`,
    but the agent never addressed it because chat_loop only drains the queue
    at the *start* of each round.
    """

    @pytest.mark.asyncio
    async def test_interject_during_final_turn_is_not_dropped(self):
        client = KimiClient(_make_settings())
        messages: list[dict] = [{"role": "user", "content": "thanks"}]

        interject_queue: asyncio.Queue[str] = asyncio.Queue()
        call_count = 0

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate the LLM producing a final response while a user
                # message arrives in the interject queue mid-flight.
                interject_queue.put_nowait("Can ya help me make a list")
                # tiny await so the put_nowait settles like a real race
                await asyncio.sleep(0)
                return ChatResponse(
                    content="safe travels tomorrow",
                    finish_reason="end_turn",
                )
            # If the loop correctly re-runs after seeing the interjection,
            # this second call is what addresses the new user message.
            return ChatResponse(
                content="sure, what are you packing?",
                finish_reason="end_turn",
            )

        client.chat = mock_chat
        registry = ToolRegistry()

        await client.chat_loop(messages, registry, interject_queue=interject_queue)

        # The interjection must have been consumed (not stranded in the queue).
        assert (
            interject_queue.empty()
        ), "interjection left in queue — message was dropped on end_turn"

        # And the LLM must have been given a chance to actually respond to it:
        # either by re-running the loop, or by some equivalent mechanism that
        # surfaces the user's text into the conversation.
        all_user_text = " ".join(
            (m["content"] if isinstance(m["content"], str) else "")
            for m in messages
            if m.get("role") == "user"
        )
        assert "Can ya help me make a list" in all_user_text or call_count >= 2, (
            "interjected message was never injected into history nor triggered "
            "a follow-up LLM round"
        )

    @pytest.mark.asyncio
    async def test_interject_during_max_rounds_final_call_is_not_dropped(self):
        """Same race, but on the max_rounds exit path: the forced final
        `chat(...)` at the bottom of the loop also runs while interjections
        can land. That exit must go through the same gate."""
        client = KimiClient(_make_settings())
        messages: list[dict] = [{"role": "user", "content": "do stuff"}]
        interject_queue: asyncio.Queue[str] = asyncio.Queue()

        call_count = 0
        forced_final_round = False
        interjected_once = False

        async def mock_chat(msgs, tools=None):
            nonlocal call_count, forced_final_round, interjected_once
            call_count += 1
            # Keep emitting tool calls until we hit max_rounds → forces the
            # "summarize now" path. tools=None signals the forced final call.
            if tools is None:
                forced_final_round = True
                if not interjected_once:
                    interject_queue.put_nowait("wait one more thing")
                    interjected_once = True
                    await asyncio.sleep(0)
                return ChatResponse(content="summary", finish_reason="end_turn")
            return ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"t{call_count}",
                        name="bash",
                        arguments={"command": f"echo {call_count}"},
                    )
                ],
                finish_reason="tool_use",
            )

        client.chat = mock_chat

        async def fake_bash(**kw):
            return "ok"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="bash",
                description="",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=fake_bash,
            )
        )

        await client.chat_loop(
            messages,
            registry,
            max_rounds=2,
            interject_queue=interject_queue,
        )

        assert forced_final_round, "test setup failed — never hit max_rounds path"
        assert (
            interject_queue.empty()
        ), "interjection during max_rounds final chat() was dropped"


# ---------------------------------------------------------------------------
# Property tests: random interjection schedules across random round counts
# ---------------------------------------------------------------------------


def _collect_user_text(messages: list[dict]) -> str:
    """Concatenate all human-readable user text in messages, including
    text blocks merged into tool_result-bearing user messages."""
    parts: list[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts)


def _assert_no_consecutive_user(messages: list[dict]) -> None:
    prev_role: str | None = None
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "user" and prev_role == "user":
            raise AssertionError(
                f"consecutive user-role messages at index {i - 1}, {i}: "
                f"{messages[i - 1]} -> {m}"
            )
        prev_role = role


def _assert_tool_use_paired(messages: list[dict]) -> None:
    """Every assistant message with tool_use blocks must be immediately followed
    by a user message containing tool_result blocks for each tool_use id."""
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        tool_use_ids = [
            b["id"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        if not tool_use_ids:
            continue
        assert i + 1 < len(messages), f"trailing tool_use with no result at {i}"
        nxt = messages[i + 1]
        assert nxt.get("role") == "user", f"tool_use at {i} not followed by user"
        nxt_content = nxt.get("content")
        assert isinstance(nxt_content, list), "tool_result message must use blocks"
        result_ids = {
            b.get("tool_use_id")
            for b in nxt_content
            if isinstance(b, dict) and b.get("type") == "tool_result"
        }
        for tid in tool_use_ids:
            assert tid in result_ids, (
                f"tool_use id {tid} at message {i} has no matching tool_result "
                f"in next message"
            )


class TestChatLoopProperties:
    """Fuzz interjection timing across random round counts. Each test enqueues
    interjections at random points during the run and asserts the loop's
    invariants hold regardless of when they land."""

    @given(
        rounds_to_tool=st.integers(min_value=0, max_value=4),
        interject_targets=st.lists(
            st.integers(min_value=0, max_value=8), min_size=0, max_size=6
        ),
    )
    @hsettings(max_examples=40, deadline=None)
    @pytest.mark.asyncio
    async def test_interjections_preserved_and_history_well_formed(
        self, rounds_to_tool: int, interject_targets: list[int]
    ):
        """For an arbitrary mix of tool-using rounds and interjection arrivals,
        chat_loop must (a) never strand a message in the queue, (b) inject
        every interjection into the conversation, (c) never produce two
        consecutive user-role messages, (d) keep every tool_use paired with a
        tool_result."""
        client = KimiClient(_make_settings())
        messages: list[dict] = [{"role": "user", "content": "kickoff"}]
        interject_queue: asyncio.Queue[str] = asyncio.Queue()

        # Map each scheduled interjection to a clipped target call index in
        # [0, rounds_to_tool] so it actually gets injected before the loop
        # decides to terminate.
        clipped_targets = [min(t, rounds_to_tool) for t in interject_targets]
        bodies = [f"inj_{i}" for i in range(len(clipped_targets))]
        remaining = list(range(len(clipped_targets)))

        call_count = 0

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            this_call = call_count
            call_count += 1
            # Drop scheduled interjections onto the queue *during* this call,
            # simulating user messages arriving while the LLM await is in flight.
            for idx in list(remaining):
                if clipped_targets[idx] == this_call:
                    interject_queue.put_nowait(bodies[idx])
                    remaining.remove(idx)
            await asyncio.sleep(0)
            # Hard guard: never tool-call once tools is None (forced summary).
            if tools is None:
                return ChatResponse(content="forced", finish_reason="end_turn")
            if this_call < rounds_to_tool:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"t{this_call}",
                            name="bash",
                            arguments={"i": this_call},
                        )
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done", finish_reason="end_turn")

        client.chat = mock_chat

        async def fake_bash(**kw):
            return "ok"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="bash",
                description="",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=fake_bash,
            )
        )

        await client.chat_loop(
            messages,
            registry,
            max_rounds=20,
            interject_queue=interject_queue,
        )

        # (a) no interjection stranded
        assert interject_queue.empty(), "interjection left in queue at exit"
        assert remaining == [], "test setup: not all scheduled bodies were injected"

        # (b) every body present somewhere in the user-side text
        all_user = _collect_user_text(messages)
        for body in bodies:
            assert body in all_user, f"interjection {body!r} never reached messages"

        # (c) no consecutive user-role messages (API rejects this)
        _assert_no_consecutive_user(messages)

        # (d) tool_use / tool_result pairing intact
        _assert_tool_use_paired(messages)

    @pytest.mark.asyncio
    async def test_pure_interjection_survives_persistence_filter(self):
        """BUG: a user message containing only an interjection (no tool_result
        blocks) is in `pending.new_messages` but `persist_response` skips it
        because its filter requires `has_tool_result`. The user's text never
        reaches the DB — next turn the model has no record the user said it.

        Repro: model returns end_turn on the first call, an interjection lands
        during that call, exit gate re-enters, _drain_interjections appends a
        NEW user message (trailing message is now assistant). That new user
        message has only text blocks — persistence drops it on the floor."""
        client = KimiClient(_make_settings())
        messages: list[dict] = [{"role": "user", "content": "kickoff"}]
        interject_queue: asyncio.Queue[str] = asyncio.Queue()

        call_count = 0

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                interject_queue.put_nowait("URGENT_USER_TEXT")
                await asyncio.sleep(0)
                return ChatResponse(content="first reply", finish_reason="end_turn")
            return ChatResponse(content="second reply", finish_reason="end_turn")

        client.chat = mock_chat
        registry = ToolRegistry()

        response = await client.chat_loop(
            messages, registry, interject_queue=interject_queue
        )

        # Replicate persist_response's filter after the fix (handler.py):
        # list-format user messages are persisted (both tool_result-bearing and
        # pure-text interjections); string-content messages (checkpoint prompts)
        # are transient and skipped.
        persisted_user_text: list[str] = []
        for m in response.new_messages:
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue  # string-content: checkpoint / max-rounds prompt, transient
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    persisted_user_text.append(b.get("text", ""))

        joined = "\n".join(persisted_user_text)
        assert "URGENT_USER_TEXT" in joined, (
            "interjection text was not found in list-format user messages — "
            "_drain_interjections may have reverted to string content, or "
            "the message was not included in new_messages."
        )

    @pytest.mark.asyncio
    async def test_doom_loop_counter_does_not_persist_across_reentry(self):
        """BUG: `recent_calls` accumulates across the entire chat_loop, including
        across exit-gate re-entries. If the model called X(args) DOOM_LOOP_THRESHOLD-1
        times pre-interjection, then the user interjects, then the model legitimately
        calls X(args) once more to start fresh work, the doom-loop guard trips
        and the call is rejected — even though the user's interjection makes
        this a new context, not a stuck loop."""
        from aineko.kimi.client import DOOM_LOOP_THRESHOLD

        client = KimiClient(_make_settings())
        messages: list[dict] = [{"role": "user", "content": "go"}]
        interject_queue: asyncio.Queue[str] = asyncio.Queue()

        call_count = 0
        bash_invocations: list[dict] = []

        async def mock_chat(msgs, tools=None):
            nonlocal call_count
            call_count += 1
            # First (THRESHOLD-1) tool calls all repeat the same args, building
            # up the doom-loop counter just under the trip threshold.
            if call_count <= DOOM_LOOP_THRESHOLD - 1:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"t{call_count}", name="bash", arguments={"cmd": "x"}
                        )
                    ],
                    finish_reason="tool_use",
                )
            # Then end_turn, but an interjection lands mid-flight.
            if call_count == DOOM_LOOP_THRESHOLD:
                interject_queue.put_nowait("now do x for real this time")
                await asyncio.sleep(0)
                return ChatResponse(content="ok", finish_reason="end_turn")
            # After re-entry: model addresses the new context with the same
            # tool. A *fresh* call following user input is not a doom loop —
            # but the leftover counter says otherwise.
            if call_count == DOOM_LOOP_THRESHOLD + 1:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"t{call_count}", name="bash", arguments={"cmd": "x"}
                        )
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done", finish_reason="end_turn")

        client.chat = mock_chat

        async def fake_bash(**kw):
            bash_invocations.append(kw)
            return "ok"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="bash",
                description="",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=fake_bash,
            )
        )

        await client.chat_loop(
            messages,
            registry,
            max_rounds=20,
            interject_queue=interject_queue,
        )

        # The post-interjection bash call should have actually executed. If the
        # doom-loop guard trips on it, fake_bash never runs → bug.
        post_interject_calls = len(bash_invocations) - (DOOM_LOOP_THRESHOLD - 1)
        assert post_interject_calls >= 1, (
            "doom-loop counter persisted across exit-gate re-entry: model's "
            "first tool call after the user's interjection was rejected as a "
            "loop. recent_calls should reset (or decay) when an interjection "
            "establishes new context."
        )

    @given(
        burst_size=st.integers(min_value=1, max_value=5),
        round_count=st.integers(min_value=1, max_value=4),
    )
    @hsettings(max_examples=20, deadline=None)
    @pytest.mark.asyncio
    async def test_burst_of_interjections_on_final_turn_all_addressed(
        self, burst_size: int, round_count: int
    ):
        """Simultaneous burst of interjections during the final (end_turn) call:
        the loop should re-enter, drain them all in one round, and then exit
        cleanly with the queue empty."""
        client = KimiClient(_make_settings())
        messages: list[dict] = [{"role": "user", "content": "go"}]
        interject_queue: asyncio.Queue[str] = asyncio.Queue()

        call_count = 0
        burst_dropped = False

        async def mock_chat(msgs, tools=None):
            nonlocal call_count, burst_dropped
            this_call = call_count
            call_count += 1
            # On the call that's about to return end_turn, dump the whole burst.
            if this_call == round_count and not burst_dropped:
                for i in range(burst_size):
                    interject_queue.put_nowait(f"burst_{i}")
                burst_dropped = True
                await asyncio.sleep(0)
            if tools is None:
                return ChatResponse(content="forced", finish_reason="end_turn")
            if this_call < round_count:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"t{this_call}", name="bash", arguments={"i": this_call}
                        )
                    ],
                    finish_reason="tool_use",
                )
            return ChatResponse(content="done", finish_reason="end_turn")

        client.chat = mock_chat

        async def fake_bash(**kw):
            return "ok"

        registry = ToolRegistry()
        registry.register(
            ToolDef(
                name="bash",
                description="",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=fake_bash,
            )
        )

        await client.chat_loop(
            messages,
            registry,
            max_rounds=20,
            interject_queue=interject_queue,
        )

        assert interject_queue.empty()
        all_user = _collect_user_text(messages)
        for i in range(burst_size):
            assert f"burst_{i}" in all_user, f"burst_{i} dropped"
        _assert_no_consecutive_user(messages)
        _assert_tool_use_paired(messages)
