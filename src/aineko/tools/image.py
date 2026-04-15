"""Image description tool — delegates vision to the configured model."""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from aineko.tools.registry import ToolDef

logger = logging.getLogger(__name__)

# Injected by app.py before any tool call
_settings: Any = None  # KimiSettings

_BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"
_MAX_IMAGES = 20
_MAX_IMAGE_BYTES = 5_000_000


async def _load_image(url: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for any supported URL scheme."""
    if url.startswith("data:"):
        # data:<mime>;base64,<data>
        header, data = url.split(",", 1)
        media_type = header.split(":")[1].split(";")[0]
        return data, media_type

    if url.startswith("file://"):
        path = Path(url[7:])
        raw = path.read_bytes()
        media_type = mimetypes.guess_type(str(path))[0] or "image/png"
        return base64.b64encode(raw).decode(), media_type

    # http(s)://
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": _BROWSER_UA})
        resp.raise_for_status()
        if len(resp.content) > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"image too large ({len(resp.content) // 1_000_000}MB, limit 5MB)"
            )
        media_type = resp.headers.get("content-type", "image/png").split(";")[0].strip()
        return base64.b64encode(resp.content).decode(), media_type


async def describe_images(
    urls: list[str],
    prompt: str = "Describe the image.",
) -> str:
    if not _settings:
        return "Error: image tool not configured"
    if not urls:
        return "Error: no URLs provided"
    if len(urls) > _MAX_IMAGES:
        return f"Error: maximum {_MAX_IMAGES} images per call"

    from aineko.kimi.client import KimiClient

    # Disable thinking for lightweight vision sub-calls
    vision_settings = _settings.model_copy(update={"thinking": False})
    client = KimiClient(vision_settings)

    content: list[dict] = []
    for url in urls:
        try:
            data, media_type = await _load_image(url)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
            )
            logger.info(
                "image loaded",
                extra={"event": "image_load", "url": url[:100], "mime": media_type},
            )
        except Exception as e:
            logger.warning(
                "failed to load image", extra={"url": url[:100], "error": str(e)}
            )
            content.append({"type": "text", "text": f"[could not load {url}: {e}]"})

    content.append({"type": "text", "text": prompt})

    response = await client.chat([{"role": "user", "content": content}])
    return response.content or "No description returned."


image_tool = ToolDef(
    name="describe_images",
    description=(
        "Describe one or more images by URL. "
        "Accepts http(s)://, file://, and data: URLs (up to 20). "
        "Calls the vision model with an optional prompt and returns a text description. "
        "Since the primary model already has vision, only use this for images that were "
        "not directly provided in the conversation (e.g. URLs discovered while browsing)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Image URLs to describe (http(s)://, file://, or data:)",
                "maxItems": _MAX_IMAGES,
            },
            "prompt": {
                "type": "string",
                "description": "Instruction for the vision model (default: 'Describe the image.')",
                "default": "Describe the image.",
            },
        },
        "required": ["urls"],
    },
    handler=describe_images,
)
