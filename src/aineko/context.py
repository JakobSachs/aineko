"""Context window management — sliding window trimmer."""

import logging

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def trim_messages(
    messages: list[dict[str, str]],
    max_tokens: int,
) -> list[dict[str, str]]:
    """Keep system prompt + as many recent messages as fit within the token budget.

    Returns a new list with the system message first, then the most recent
    messages that fit. Oldest messages are dropped first.
    """
    if not messages:
        return messages

    # System message is always kept
    system_msg = messages[0] if messages[0]["role"] == "system" else None
    conversation = messages[1:] if system_msg else messages

    budget = max_tokens
    if system_msg:
        budget -= estimate_tokens(system_msg["content"])

    # Walk backwards (newest first), accumulate until we exceed budget
    kept: list[dict[str, str]] = []
    used = 0
    for msg in reversed(conversation):
        cost = estimate_tokens(msg.get("content", "") or "")
        if used + cost > budget:
            break
        kept.append(msg)
        used += cost

    kept.reverse()

    dropped = len(conversation) - len(kept)
    if dropped > 0:
        logger.info(
            "Context: trimmed %d old messages (%d kept, ~%d tokens)",
            dropped,
            len(kept),
            used,
        )

    result = []
    if system_msg:
        result.append(system_msg)
    result.extend(kept)
    return result
