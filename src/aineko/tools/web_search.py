"""Web search tool via DuckDuckGo (no API key needed)."""

import httpx

from aineko.tools.registry import ToolDef

DDG_URL = "https://api.duckduckgo.com/"


async def web_search(query: str, max_results: int = 5) -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            # DuckDuckGo instant answer API
            resp = await client.get(
                DDG_URL,
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            )
            resp.raise_for_status()
            data = resp.json()

            results: list[str] = []

            # Abstract (main answer)
            if data.get("AbstractText"):
                results.append(f"**{data.get('AbstractSource', 'Answer')}**: {data['AbstractText']}")

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if "Text" in topic:
                    url = topic.get("FirstURL", "")
                    results.append(f"- {topic['Text']}" + (f" ({url})" if url else ""))

            if not results:
                return f"No results found for: {query}"

            return "\n".join(results)
        except Exception as e:
            return f"Search error: {e}"


web_search_tool = ToolDef(
    name="web_search",
    description="Search the web via DuckDuckGo. Returns a summary and related topics.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum related topics to return (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    handler=web_search,
)
