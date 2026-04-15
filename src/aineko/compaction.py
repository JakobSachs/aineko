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


def ingest_before_compaction(
    messages: list[dict[str, Any]],
    session_id: int,
    room_id: str,
    keep_recent: int = 4,
) -> None:
    """Store messages about to be compacted into ChromaDB long-term memory.

    Called right before compact_messages(). Only ingests the old messages
    that will be replaced by the summary. The MemoryStore's built-in dedup
    (0.95 cosine threshold) prevents storing content that's already in memory.
    """
    from aineko.memory.store import MemoryStore

    conversation = (
        messages[1:] if messages and messages[0].get("role") == "system" else messages
    )
    if len(conversation) <= keep_recent:
        return

    old_messages = conversation[:-keep_recent]

    # Build exchange pairs from old messages
    exchanges: list[str] = []
    pending_user: str | None = None
    for m in old_messages:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            text = _clean_content(content)
            if text:
                exchanges.append(f"> {pending_user}\n{text}")
            pending_user = None

    if not exchanges:
        return

    try:
        store = MemoryStore(persist_dir="/data/memory/chromadb")
        text = "\n\n---\n\n".join(exchanges)
        tags = {"type": "conversation", "session": str(session_id), "room": room_id}
        ids = store.add(text, source=f"conversation:{session_id}", tags=tags)
        if ids:
            logger.info(
                "ingested messages into long-term memory before compaction",
                extra={"chunks_stored": len(ids), "session_id": session_id},
            )
    except Exception:
        logger.exception("failed to ingest messages before compaction")


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
