"""Web fetch tool — retrieves a URL and returns content as text or an image block."""

import asyncio
import base64
import io
import logging

import httpx

from aineko.tools.registry import ToolDef, ToolResult

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0 aineko/1.0"
)
_PLAIN_UA = "curl/8.7.1"
_MAX_SIZE = 5_000_000  # 5 MB
_MAX_OUTPUT = 100_000  # chars
_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def _is_cloudflare_block(resp: httpx.Response) -> bool:
    if "cf-mitigated" in resp.headers:
        return True
    if resp.status_code in (403, 503):
        body = resp.text.lower()
        return "just a moment" in body or "cloudflare" in body
    return False


async def _get(url: str, ua: str, timeout: int) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        return await client.get(url, headers={"User-Agent": ua})


def _to_markdown(html_bytes: bytes) -> str:
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert_stream(io.BytesIO(html_bytes), file_extension=".html")
    return result.text_content


def _strip_html(html_bytes: bytes) -> str:
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self._parts: list[str] = []
            self._skip = False

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag in ("script", "style"):
                self._skip = True

        def handle_endtag(self, tag: str) -> None:
            if tag in ("script", "style"):
                self._skip = False

        def handle_data(self, data: str) -> None:
            if not self._skip:
                self._parts.append(data)

        def get_text(self) -> str:
            return " ".join(self._parts)

    s = _Stripper()
    s.feed(html_bytes.decode("utf-8", errors="replace"))
    return s.get_text()


async def web_fetch(
    url: str,
    format: str = "markdown",
    timeout: int = 30,
) -> ToolResult:
    timeout = min(timeout, 120)

    logger.info("web fetch", extra={"event": "web_fetch_start", "url": url})

    try:
        resp = await _get(url, _BROWSER_UA, timeout)
    except httpx.TimeoutException:
        return f"Error: request timed out after {timeout}s"
    except Exception as e:
        return f"Error fetching {url}: {e}"

    if _is_cloudflare_block(resp):
        logger.info(
            "cloudflare block detected, retrying with plain UA",
            extra={"event": "web_fetch_cf_retry", "url": url},
        )
        try:
            resp = await _get(url, _PLAIN_UA, timeout)
        except Exception as e:
            return f"Error fetching {url}: {e}"

    if len(resp.content) > _MAX_SIZE:
        return (
            f"Error: response too large ({len(resp.content) // 1_000_000}MB, limit 5MB)"
        )

    ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()

    # Images — return as vision content block
    if ct in _IMAGE_TYPES:
        data = base64.b64encode(resp.content).decode()
        logger.info(
            "web fetch image",
            extra={"event": "web_fetch_done", "url": url, "type": "image", "mime": ct},
        )
        return [
            {"type": "text", "text": f"Image from {url} ({ct}):"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": ct, "data": data},
            },
        ]

    # HTML — convert to markdown or plain text
    if "text/html" in ct:
        if format == "text":
            text = _strip_html(resp.content)
        else:
            text = await asyncio.to_thread(_to_markdown, resp.content)
        logger.info(
            "web fetch html",
            extra={
                "event": "web_fetch_done",
                "url": url,
                "type": "html",
                "format": format,
            },
        )
        return text[:_MAX_OUTPUT]

    # Everything else (plain text, JSON, XML, SVG, …)
    logger.info(
        "web fetch raw",
        extra={"event": "web_fetch_done", "url": url, "type": ct},
    )
    return resp.text[:_MAX_OUTPUT]


web_fetch_tool = ToolDef(
    name="web_fetch",
    description=(
        "Fetch the content of a URL. "
        "HTML pages are converted to markdown by default (format='markdown') "
        "or stripped to plain text (format='text'). "
        "Images are returned as vision attachments. "
        "Other content (JSON, plain text, SVG, …) is returned as-is. "
        "Max 5MB response, timeout capped at 120s."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "format": {
                "type": "string",
                "enum": ["markdown", "text"],
                "description": "Output format for HTML pages (default: markdown)",
                "default": "markdown",
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds (default 30, max 120)",
                "default": 30,
            },
        },
        "required": ["url"],
    },
    handler=web_fetch,
)
