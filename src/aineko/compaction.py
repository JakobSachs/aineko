"""Conversation compaction — LLM-powered history summarization.

When conversations grow too long, summarize older messages into a structured
summary and keep only recent messages. Inspired by OpenClaw's approach.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from aineko.context import estimate_tokens
from aineko.time import now_local

logger = logging.getLogger(__name__)

# Reserve buffer: trigger compaction this many tokens before the hard limit
COMPACTION_RESERVE = 0

# Minimum messages in conversation before compaction is even considered
# (system + at least 6 conversation messages)
MIN_MESSAGES_FOR_COMPACTION = 7

COMPACTION_TEMPLATE = """\
## Goal
[What goal(s) is the user trying to accomplish?]

## Instructions
[What important instructions did the user give that are still relevant?]

## Discoveries
[What notable things were learned during this conversation?]

## Accomplished
[What work has been completed, what is still in progress, what is left?]

## Relevant files / directories
[List of relevant files that have been read, edited, or created]
"""


def _clean_content(raw: str) -> str:
    """Extract human-readable text from a message, stripping tool-call JSON."""
    if not raw:
        return ""
    try:
        blocks = json.loads(raw)
        if isinstance(blocks, list):
            parts = []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"].strip())
            return "\n".join(p for p in parts if p)
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    cleaned = re.sub(r"<\|tool_call[^>]*\|>", "", raw)
    return cleaned.strip()


SILENT_REPLY_TOKEN = "[SILENT_REPLY_TOKEN]"

MEMORY_FLUSH_SYSTEM_PROMPT = """\
Pre-compaction memory flush turn.
The session is near auto-compaction; capture durable memories to disk.
Store durable memories only in memory/YYYY-MM-DD.md via the memory tool's
store action (it will append; never overwrite).
Treat workspace bootstrap/reference files such as MEMORY.md, SOUL.md, and
AGENTS.md as read-only during this flush; never overwrite, replace, or edit them.
You may reply, but usually [SILENT_REPLY_TOKEN] is correct.
"""


def _memory_flush_user_prompt() -> str:
    return (
        "Pre-compaction memory flush.\n"
        "Store durable memories only in memory/YYYY-MM-DD.md via the memory "
        "tool's store action (create the file by appending if it does not exist).\n"
        "Treat workspace bootstrap/reference files such as MEMORY.md, SOUL.md, "
        "and AGENTS.md as read-only during this flush; never overwrite, replace, "
        "or edit them.\n"
        "If today's log already exists, APPEND new content only; do not "
        "overwrite existing entries.\n"
        "Do NOT create timestamped variant files (e.g., YYYY-MM-DD-HHMM.md); "
        "always use the canonical YYYY-MM-DD.md filename.\n"
        f"If nothing to store, reply with {SILENT_REPLY_TOKEN}.\n"
        f"Current time: {now_local().isoformat(timespec='seconds')}\n"
    )


async def run_memory_flush(
    messages: list[dict[str, Any]],
    kimi_client: Any,
    tools: Any,
) -> None:
    """Run a dedicated memory-flush turn right before compaction.

    Replays the pre-compaction transcript with a system prompt instructing
    the model to persist durable memories via the memory tool. Output is
    discarded — only side effects (memory tool calls) matter.
    """
    if not messages:
        return

    conversation = (
        messages[1:] if messages[0].get("role") == "system" else list(messages)
    )
    if not conversation:
        return

    flush_messages: list[dict[str, Any]] = [
        {"role": "system", "content": MEMORY_FLUSH_SYSTEM_PROMPT},
        *conversation,
        {"role": "user", "content": _memory_flush_user_prompt()},
    ]

    logger.info(
        "running memory flush turn before compaction",
        extra={"event": "memory_flush_start", "msg_count": len(conversation)},
    )
    try:
        await kimi_client.chat_loop(flush_messages, tools)
    except Exception:
        logger.exception("memory flush turn failed; continuing with compaction")


def should_compact(messages: list[dict[str, Any]], max_tokens: int) -> bool:
    """Check if conversation needs compaction."""
    if len(messages) < MIN_MESSAGES_FOR_COMPACTION:
        return False
    total = sum(estimate_tokens(m.get("content", "") or "") for m in messages)
    threshold = max_tokens - COMPACTION_RESERVE
    return total >= threshold


def build_compaction_prompt(messages: list[dict[str, Any]]) -> str:
    """Build the prompt sent to the LLM to summarize the conversation."""
    # Format conversation for the summarizer
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content", "") or ""
        if isinstance(content, list):
            # Multimodal content — extract text parts
            content = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if role == "system":
            continue  # Don't include system prompt in what gets summarized
        lines.append(f"[{role}]: {content}")

    conversation = "\n\n".join(lines)

    return (
        f"Summarize the following conversation into a structured summary.\n"
        f"Focus on information needed to continue the conversation: what was done, "
        f"what's being worked on, which files are involved, and what's next.\n"
        f"The summary will replace old messages so another turn can continue the work.\n\n"
        f"Use this template:\n---\n{COMPACTION_TEMPLATE}---\n\n"
        f"Conversation to summarize:\n\n{conversation}"
    )


async def compact_messages(
    messages: list[dict[str, Any]],
    kimi_client: Any,
    keep_recent: int = 4,
) -> tuple[list[dict[str, Any]], str | None]:
    """Compact old messages into a summary, keeping recent ones intact.

    Args:
        messages: Full message list (system + conversation)
        kimi_client: KimiClient instance for LLM calls
        keep_recent: Number of recent conversation messages to preserve

    Returns:
        (compacted_messages, summary_text) — summary_text is None if no
        compaction was performed.
    """
    if not messages:
        return messages, None

    # Split system prompt from conversation
    system_msg = messages[0] if messages[0].get("role") == "system" else None
    conversation = messages[1:] if system_msg else messages

    # Not enough to compact
    if len(conversation) <= keep_recent:
        return messages, None

    # Split into old (to summarize) and recent (to keep)
    old_messages = conversation[:-keep_recent]
    recent_messages = conversation[-keep_recent:]

    if not old_messages:
        return messages, None

    # Build prompt and call LLM for summary
    all_for_summary = ([system_msg] if system_msg else []) + old_messages
    prompt = build_compaction_prompt(all_for_summary)

    logger.info(
        "compacting conversation",
        extra={
            "event": "compaction_start",
            "old_messages": len(old_messages),
            "recent_messages": len(recent_messages),
        },
    )

    summary_response = await kimi_client.chat(
        messages=[{"role": "user", "content": prompt}],
        tools=None,
    )

    summary: str = summary_response.content or "Conversation summary unavailable."

    logger.info(
        "compaction complete",
        extra={
            "event": "compaction_done",
            "summary_len": len(summary),
            "old_messages_removed": len(old_messages),
        },
    )

    # Rebuild: system + summary as system note + recent messages
    summary_content: str = (
        f"[Conversation summary — older messages were compacted]\n\n{summary}"
    )
    result: list[dict[str, Any]] = []
    if system_msg:
        result.append(system_msg)
    result.append({"role": "system", "content": summary_content})
    result.extend(recent_messages)

    return result, summary_content
