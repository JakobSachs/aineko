"""Conversation compaction — LLM-powered history summarization.

When conversations grow too long, summarize older messages into a structured
summary and keep only recent messages. Inspired by OpenClaw's approach.
"""

import logging
from typing import Any

from aineko.context import estimate_tokens

logger = logging.getLogger(__name__)

# Reserve buffer: trigger compaction this many tokens before the hard limit
COMPACTION_RESERVE = 20_000

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
) -> list[dict[str, Any]]:
    """Compact old messages into a summary, keeping recent ones intact.

    Args:
        messages: Full message list (system + conversation)
        kimi_client: KimiClient instance for LLM calls
        keep_recent: Number of recent conversation messages to preserve

    Returns:
        Compacted message list: [system, summary, ...recent_messages]
    """
    if not messages:
        return messages

    # Split system prompt from conversation
    system_msg = messages[0] if messages[0].get("role") == "system" else None
    conversation = messages[1:] if system_msg else messages

    # Not enough to compact
    if len(conversation) <= keep_recent:
        return messages

    # Split into old (to summarize) and recent (to keep)
    old_messages = conversation[:-keep_recent]
    recent_messages = conversation[-keep_recent:]

    if not old_messages:
        return messages

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

    summary = summary_response.content or "Conversation summary unavailable."

    logger.info(
        "compaction complete",
        extra={
            "event": "compaction_done",
            "summary_len": len(summary),
            "old_messages_removed": len(old_messages),
        },
    )

    # Rebuild: system + summary as system note + recent messages
    result: list[dict[str, Any]] = []
    if system_msg:
        result.append(system_msg)
    result.append(
        {
            "role": "system",
            "content": f"[Conversation summary — older messages were compacted]\n\n{summary}",
        }
    )
    result.extend(recent_messages)

    return result
