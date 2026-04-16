"""Tool to query seen RSS items from the database."""

from sqlalchemy import select

from aineko.db import get_session
from aineko.models.rss import RssSeenItem
from aineko.tools.registry import ToolDef


async def _query_rss(
    keyword: str = "",
    feed: str = "",
    limit: int = 20,
) -> str:
    """Query seen RSS items, optionally filtered by keyword and/or feed name."""
    limit = min(limit, 100)

    async for db in get_session():
        query = select(RssSeenItem).order_by(RssSeenItem.seen_at.desc())

        if keyword:
            query = query.where(RssSeenItem.title.ilike(f"%{keyword}%"))

        if feed:
            # Match against the feed_url column; also try a loose substring match
            # so the user can pass a domain or short name instead of the full URL
            query = query.where(RssSeenItem.feed_url.ilike(f"%{feed}%"))

        query = query.limit(limit)
        result = await db.execute(query)
        items = result.scalars().all()

    if not items:
        return "No RSS items found matching your query."

    lines: list[str] = []
    for item in reversed(items):  # chronological order
        lines.append(f"[{item.seen_at:%Y-%m-%d %H:%M}] {item.feed_url}\n  {item.title}")

    return "\n\n".join(lines)


query_rss_tool = ToolDef(
    name="query_rss",
    description=(
        "Query previously seen RSS feed items stored in the database. "
        "Use this to look up past articles, check what was posted by a feed, "
        "or search for items by title keyword. "
        "Results are returned in chronological order (oldest first within the limit)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "Filter by title keyword (case-insensitive substring match). Leave empty for no filter.",
            },
            "feed": {
                "type": "string",
                "description": "Filter by feed URL substring (e.g. a domain name or feed name). Leave empty for all feeds.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 20, max 100).",
            },
        },
        "required": [],
    },
    handler=_query_rss,
)
