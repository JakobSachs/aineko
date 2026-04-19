"""RSS feed poller — runs as a background task, injects new items into the agent queue."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import feedparser
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

import aineko.db as _db
from aineko.models.rss import RssSeenItem

if TYPE_CHECKING:
    from aineko.config import RssFeedConfig
    from aineko.matrix.client import MatrixConnector

logger = logging.getLogger(__name__)

_DESCRIPTION_MAX = 600


def _entry_guid(entry: feedparser.FeedParserDict) -> str:
    return entry.get("id") or entry.get("link") or entry.get("title") or str(id(entry))


def _entry_description(entry: feedparser.FeedParserDict) -> str:
    raw = entry.get("summary") or entry.get("content", [{}])[0].get("value", "")
    return raw[:_DESCRIPTION_MAX].strip()


def _build_prompt(feed_name: str, entry: feedparser.FeedParserDict) -> str:
    title = entry.get("title", "(no title)")
    link = entry.get("link", "")
    description = _entry_description(entry)

    lines = [
        f"[RSS] New item from **{feed_name}**",
        "",
        f"**{title}**",
    ]
    if link:
        lines.append(link)
    if description:
        lines += ["", description]
    lines += [
        "",
        "---",
        "Based on what you know about my interests and our conversation history, "
        "decide whether this item is worth bringing to my attention. "
        "If yes, forward it using send_message. "
        "If no, stay completely silent — do not send any message.",
    ]
    return "\n".join(lines)


async def _fetch_feed(url: str) -> feedparser.FeedParserDict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, feedparser.parse, url)


async def _is_seen(feed_url: str, guid: str) -> bool:
    assert _db.async_session_factory is not None
    async with _db.async_session_factory() as db:
        result = await db.execute(
            select(RssSeenItem).where(
                RssSeenItem.feed_url == feed_url,
                RssSeenItem.guid == guid,
            )
        )
        return result.scalar_one_or_none() is not None


async def _mark_seen_bulk(
    feed_url: str, entries: list[feedparser.FeedParserDict]
) -> None:
    """Insert all entries as seen, ignoring conflicts."""
    assert _db.async_session_factory is not None
    rows = [
        {"feed_url": feed_url, "guid": _entry_guid(e), "title": e.get("title", "")}
        for e in entries
    ]
    if not rows:
        return
    async with _db.async_session_factory() as db:
        stmt = (
            pg_insert(RssSeenItem)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["feed_url", "guid"])
        )
        await db.execute(stmt)
        await db.commit()


async def _seed_feed(feed: RssFeedConfig) -> None:
    """Mark all current entries as seen without notifying. Called once on startup."""
    try:
        parsed = await _fetch_feed(feed.url)
        await _mark_seen_bulk(feed.url, parsed.entries)
        logger.info("RSS seeded %d entries from %s", len(parsed.entries), feed.name)
    except Exception:
        logger.exception("RSS seed failed for %s", feed.url)


async def _check_feed(
    matrix: MatrixConnector,
    room_id: str,
    feed: RssFeedConfig,
) -> None:
    parsed = await _fetch_feed(feed.url)
    for entry in parsed.entries:
        guid = _entry_guid(entry)
        if await _is_seen(feed.url, guid):
            continue
        # Mark seen before injecting to avoid double-firing on slow agent runs
        await _mark_seen_bulk(feed.url, [entry])
        prompt = _build_prompt(feed.name, entry)
        logger.info("RSS new item from %s: %s", feed.name, entry.get("title", guid))
        await matrix.inject_message(
            room_id,
            prompt,
            sender="rss",
            suppress_text_response=True,
        )


_CLEANUP_INTERVAL_SECONDS = 24 * 3600
_CLEANUP_AGE_DAYS = 90


async def rss_cleanup_loop() -> None:
    """Periodically delete rss_seen_items rows older than 90 days."""
    assert _db.async_session_factory is not None
    while True:
        try:
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
                days=_CLEANUP_AGE_DAYS
            )
            async with _db.async_session_factory() as db:
                result = await db.execute(
                    delete(RssSeenItem).where(RssSeenItem.seen_at < cutoff)
                )
                await db.commit()
                logger.info("RSS cleanup: deleted %d old rows", result.rowcount or 0)
        except Exception:
            logger.exception("RSS cleanup failed")
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)


async def rss_poll_loop(
    matrix: MatrixConnector,
    room_id: str,
    feeds: list[RssFeedConfig],
    interval: int,
) -> None:
    """Seed all feeds on startup, then poll forever. Designed for asyncio.create_task."""
    for feed in feeds:
        await _seed_feed(feed)

    while True:
        await asyncio.sleep(interval)
        for feed in feeds:
            try:
                await _check_feed(matrix, room_id, feed)
            except Exception:
                logger.exception("RSS poll failed for %s", feed.url)
