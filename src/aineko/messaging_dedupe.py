"""Message deduplication for intermediate content vs send_message tool calls.

Ported from OpenClaw's messaging-dedupe.ts — prevents the same message
from being delivered twice when the LLM returns both response content
and an explicit send_message tool call.
"""

from __future__ import annotations

import re
from collections import deque

MIN_DUPLICATE_LENGTH = 10

# Regex for emoji: covers most common emoji ranges
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001fa6f"  # chess symbols
    "\U0001fa70-\U0001faff"  # symbols extended-A
    "\U00002702-\U000027b0"  # dingbats
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"  # ZWJ
    "\U000020e3"  # combining enclosing keycap
    "\U00002600-\U000026ff"  # misc symbols
    "\U00002b50-\U00002b55"  # stars
    "\U0000231a-\U0000231b"  # watch/hourglass
    "\U00002934-\U00002935"  # arrows
    "\U000025aa-\U000025ab"  # squares
    "\U000025fb-\U000025fe"  # squares
    "\U00003030\U0000303d"  # wavy dash, part alternation mark
    "\U0001f004\U0001f0cf"  # mahjong, joker
    "]+",
    flags=re.UNICODE,
)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for duplicate comparison.

    - Trims whitespace
    - Lowercases
    - Strips emoji
    - Collapses multiple spaces to single space
    """
    text = text.strip().lower()
    text = _EMOJI_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def is_duplicate(text: str, sent_texts: list[str]) -> bool:
    """Check if text is a duplicate of any previously sent text.

    Uses bidirectional substring matching on normalized text.
    Texts shorter than MIN_DUPLICATE_LENGTH are never considered duplicates.
    """
    if not sent_texts:
        return False
    normalized = normalize_text(text)
    if not normalized or len(normalized) < MIN_DUPLICATE_LENGTH:
        return False
    for sent in sent_texts:
        norm_sent = normalize_text(sent)
        if not norm_sent or len(norm_sent) < MIN_DUPLICATE_LENGTH:
            continue
        if normalized in norm_sent or norm_sent in normalized:
            return True
    return False


class MessageDeduplicator:
    """Tracks sent messages and detects duplicates.

    Supports pending/commit flow so failed sends don't suppress retries.
    """

    def __init__(self, max_tracked: int = 200):
        self._max_tracked = max_tracked
        self._sent_normalized: deque[str] = deque(maxlen=max_tracked)
        self._pending: dict[str, str] = {}  # id -> raw text

    def track_sent(self, text: str) -> None:
        """Track a successfully sent message."""
        normalized = normalize_text(text)
        if normalized and len(normalized) >= MIN_DUPLICATE_LENGTH:
            self._sent_normalized.append(normalized)

    def track_pending(self, call_id: str, text: str) -> None:
        """Track a pending (not yet confirmed) send."""
        self._pending[call_id] = text

    def commit_pending(self, call_id: str) -> None:
        """Mark a pending send as successful — now counts for dedup."""
        text = self._pending.pop(call_id, None)
        if text is not None:
            self.track_sent(text)

    def discard_pending(self, call_id: str) -> None:
        """Discard a pending send (failed or cancelled)."""
        self._pending.pop(call_id, None)

    def is_duplicate(self, text: str) -> bool:
        """Check if text would be a duplicate of a committed sent message."""
        normalized = normalize_text(text)
        if not normalized or len(normalized) < MIN_DUPLICATE_LENGTH:
            return False
        return any(
            normalized in sent or sent in normalized for sent in self._sent_normalized
        )
