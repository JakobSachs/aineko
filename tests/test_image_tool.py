"""Tests for the describe_images tool."""

import base64
from pathlib import Path

import httpx
import pytest

from aineko.tools import image as image_mod
from aineko.tools.image import _load_image, describe_images

# Minimal valid PNG (4x4 red pixels)
_RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAFUlEQVQI12P8z8BQ"
    "DwADhQGAWjR9awAAAABJRU5ErkJggg=="
)
RED_PNG = base64.b64decode(_RED_PNG_B64)


@pytest.fixture(autouse=True)
def _inject_settings(monkeypatch):
    """Provide a minimal KimiSettings-like object."""
    from aineko.config import KimiSettings

    monkeypatch.setattr(image_mod, "_settings", KimiSettings(api_key="test-key"))


# --- _load_image ---


@pytest.mark.asyncio
async def test_load_data_url():
    data_url = f"data:image/png;base64,{_RED_PNG_B64}"
    data, mime = await _load_image(data_url)
    assert mime == "image/png"
    assert base64.b64decode(data) == RED_PNG


@pytest.mark.asyncio
async def test_load_file_url(tmp_path: Path):
    img_file = tmp_path / "test.png"
    img_file.write_bytes(RED_PNG)
    data, mime = await _load_image(f"file://{img_file}")
    assert mime == "image/png"
    assert base64.b64decode(data) == RED_PNG


@pytest.mark.asyncio
async def test_load_http_url(httpx_mock):
    httpx_mock.add_response(
        content=RED_PNG,
        headers={"content-type": "image/png"},
    )
    data, mime = await _load_image("https://example.com/img.png")
    assert mime == "image/png"
    assert base64.b64decode(data) == RED_PNG


@pytest.mark.asyncio
async def test_load_http_too_large(httpx_mock):
    httpx_mock.add_response(
        content=b"x" * 6_000_000,
        headers={"content-type": "image/png"},
    )
    with pytest.raises(ValueError, match="too large"):
        await _load_image("https://example.com/big.png")


# --- describe_images ---


@pytest.mark.asyncio
async def test_describe_images_calls_model(httpx_mock):
    httpx_mock.add_response(
        content=RED_PNG,
        headers={"content-type": "image/png"},
    )
    httpx_mock.add_response(
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "A red square."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    result = await describe_images(["https://example.com/img.png"])
    assert result == "A red square."


@pytest.mark.asyncio
async def test_describe_images_thinking_disabled(httpx_mock):
    """Vision sub-call must not include thinking payload."""
    httpx_mock.add_response(content=RED_PNG, headers={"content-type": "image/png"})
    httpx_mock.add_response(
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Red."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    )
    import json as _json

    await describe_images(["https://example.com/img.png"])
    # Find the /messages request (second call)
    msgs_req = [r for r in httpx_mock.get_requests() if b"/messages" in r.url.raw_path]
    assert msgs_req, "no /messages request found"
    payload = _json.loads(msgs_req[0].content)
    assert "thinking" not in payload


@pytest.mark.asyncio
async def test_describe_images_custom_prompt(httpx_mock):
    httpx_mock.add_response(content=RED_PNG, headers={"content-type": "image/png"})
    httpx_mock.add_response(
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Lots of red."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }
    )
    import json as _json

    await describe_images(
        ["https://example.com/img.png"], prompt="What is the dominant colour?"
    )
    msgs_req = [r for r in httpx_mock.get_requests() if b"/messages" in r.url.raw_path]
    payload = _json.loads(msgs_req[0].content)
    text_blocks = [
        b["text"] for b in payload["messages"][0]["content"] if b["type"] == "text"
    ]
    assert any("dominant colour" in t for t in text_blocks)


@pytest.mark.asyncio
async def test_describe_images_failed_load_included_as_text(httpx_mock):
    """If an image URL fails, an error note is included and the call still proceeds."""
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    httpx_mock.add_response(
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Only text blocks present."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 5},
        }
    )
    import json as _json

    await describe_images(["https://bad.example.com/img.png"])
    msgs_req = [r for r in httpx_mock.get_requests() if b"/messages" in r.url.raw_path]
    payload = _json.loads(msgs_req[0].content)
    text_blocks = [
        b["text"] for b in payload["messages"][0]["content"] if b["type"] == "text"
    ]
    assert any("could not load" in t for t in text_blocks)


@pytest.mark.asyncio
async def test_describe_images_no_settings(monkeypatch):
    monkeypatch.setattr(image_mod, "_settings", None)
    result = await describe_images(["https://example.com/img.png"])
    assert "not configured" in result


@pytest.mark.asyncio
async def test_describe_images_empty_urls():
    result = await describe_images([])
    assert "no urls" in result.lower()


@pytest.mark.asyncio
async def test_describe_images_too_many_urls():
    urls = [f"data:image/png;base64,{_RED_PNG_B64}"] * 21
    result = await describe_images(urls)
    assert "maximum" in result.lower()


@pytest.mark.asyncio
async def test_describe_images_data_url_no_http(httpx_mock):
    """data: URLs should not trigger any HTTP request for loading."""
    httpx_mock.add_response(
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "A small image."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }
    )
    data_url = f"data:image/png;base64,{_RED_PNG_B64}"
    result = await describe_images([data_url])
    assert result == "A small image."
    # Only one HTTP request: the /messages call, no image download
    reqs = httpx_mock.get_requests()
    assert len(reqs) == 1
