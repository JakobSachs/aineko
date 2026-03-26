"""Web search tool via Brave Search API."""

import logging

import httpx

from aineko.tools.registry import ToolDef

logger = logging.getLogger(__name__)

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

# Loaded once at import; overridden by app.py before tools are used
brave_api_key: str = ""


async def web_search(query: str, max_results: int = 5) -> str:
    if not brave_api_key:
        logger.warning("brave api key not configured", extra={"event": "search_error"})
        return "Error: BRAVE_API_KEY not configured"

    logger.info("web search", extra={"event": "search_start", "query": query})
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(
                BRAVE_URL,
                params={"q": query, "count": max_results},
                headers={"X-Subscription-Token": brave_api_key},
            )
            resp.raise_for_status()
            data = resp.json()

            results: list[str] = []
            for item in data.get("web", {}).get("results", [])[:max_results]:
                title = item.get("title", "")
                url = item.get("url", "")
                desc = item.get("description", "")
                results.append(f"{title}\n{url}\n{desc}")

            if not results:
                logger.info("no results", extra={"event": "search_done", "query": query, "hits": 0})
                return f"No results found for: {query}"

            logger.info("search done", extra={"event": "search_done", "query": query, "hits": len(results)})
            return "\n\n".join(results)
        except Exception as e:
            logger.error("search failed", extra={"event": "search_error", "query": query, "error": str(e)})
            return f"Search error: {e}"


web_search_tool = ToolDef(
    name="web_search",
    description="Search the web via Brave Search. Returns titles, URLs, and descriptions.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    handler=web_search,
)
