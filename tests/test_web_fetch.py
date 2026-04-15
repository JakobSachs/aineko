"""Tests for the web_fetch tool."""

import base64

import httpx
import pytest

from aineko.tools.web_fetch import _is_cloudflare_block, web_fetch

SIMPLE_HTML = b"<html><body><h1>Hello</h1><p>World</p></body></html>"
PLAIN_TEXT = b"just some text"
SVG_CONTENT = (
    b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>'
)

# Minimal 4x4 red PNG (valid)
_RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAFUlEQVQI12P8z8BQ"
    "DwADhQGAWjR9awAAAABJRU5ErkJggg=="
)
RED_PNG = base64.b64decode(_RED_PNG_B64)


# --- _is_cloudflare_block ---


def test_cf_block_detects_cf_mitigated_header():
    resp = httpx.Response(403, headers={"cf-mitigated": "challenge"}, content=b"")
    assert _is_cloudflare_block(resp) is True


def test_cf_block_detects_just_a_moment():
    resp = httpx.Response(503, content=b"<html>Just a moment...</html>")
    assert _is_cloudflare_block(resp) is True


def test_cf_block_normal_403():
    resp = httpx.Response(403, content=b"Forbidden")
    assert _is_cloudflare_block(resp) is False


def test_cf_block_200_ok():
    resp = httpx.Response(200, content=b"Hello")
    assert _is_cloudflare_block(resp) is False


# --- web_fetch: HTML ---


@pytest.mark.asyncio
async def test_html_returns_string(httpx_mock):
    httpx_mock.add_response(
        content=SIMPLE_HTML,
        headers={"content-type": "text/html; charset=utf-8"},
    )
    result = await web_fetch("https://example.com")
    assert isinstance(result, str)
    assert "Hello" in result


@pytest.mark.asyncio
async def test_html_text_format_strips_tags(httpx_mock):
    httpx_mock.add_response(
        content=SIMPLE_HTML,
        headers={"content-type": "text/html"},
    )
    result = await web_fetch("https://example.com", format="text")
    assert isinstance(result, str)
    assert "<h1>" not in result
    assert "Hello" in result


# --- web_fetch: image ---


@pytest.mark.asyncio
async def test_image_returns_content_blocks(httpx_mock):
    httpx_mock.add_response(
        content=RED_PNG,
        headers={"content-type": "image/png"},
    )
    result = await web_fetch("https://example.com/img.png")
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "image"
    assert result[1]["source"]["type"] == "base64"
    assert result[1]["source"]["media_type"] == "image/png"
    # data round-trips correctly
    assert base64.b64decode(result[1]["source"]["data"]) == RED_PNG


# --- web_fetch: plain text / SVG ---


@pytest.mark.asyncio
async def test_plain_text_returned_as_string(httpx_mock):
    httpx_mock.add_response(
        content=PLAIN_TEXT,
        headers={"content-type": "text/plain"},
    )
    result = await web_fetch("https://example.com/robots.txt")
    assert isinstance(result, str)
    assert "just some text" in result


@pytest.mark.asyncio
async def test_svg_returned_as_text_not_image(httpx_mock):
    httpx_mock.add_response(
        content=SVG_CONTENT,
        headers={"content-type": "image/svg+xml"},
    )
    result = await web_fetch("https://example.com/icon.svg")
    assert isinstance(result, str)


# --- web_fetch: size cap ---


@pytest.mark.asyncio
async def test_rejects_oversized_response(httpx_mock):
    httpx_mock.add_response(
        content=b"x" * 6_000_000,
        headers={"content-type": "text/plain"},
    )
    result = await web_fetch("https://example.com/big.txt")
    assert isinstance(result, str)
    assert "too large" in result.lower()


# --- web_fetch: timeout ---


@pytest.mark.asyncio
async def test_timeout_returns_error(httpx_mock):
    httpx_mock.add_exception(httpx.TimeoutException("timed out"))
    result = await web_fetch("https://example.com", timeout=5)
    assert isinstance(result, str)
    assert "timed out" in result.lower()


# --- web_fetch: Cloudflare retry ---


@pytest.mark.asyncio
async def test_cloudflare_retry_succeeds(httpx_mock):
    # First request: CF challenge
    httpx_mock.add_response(
        status_code=503,
        content=b"<html>Just a moment...</html>",
        headers={"content-type": "text/html"},
    )
    # Second request (plain UA): real content
    httpx_mock.add_response(
        content=PLAIN_TEXT,
        headers={"content-type": "text/plain"},
    )
    result = await web_fetch("https://example.com")
    assert isinstance(result, str)
    assert "just some text" in result


# --- web_fetch: network error ---


@pytest.mark.asyncio
async def test_network_error_returns_error_string(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    result = await web_fetch("https://example.com")
    assert isinstance(result, str)
    assert "error" in result.lower()


# --- timeout clamping ---


@pytest.mark.asyncio
async def test_timeout_clamped_to_120(httpx_mock):
    httpx_mock.add_response(
        content=PLAIN_TEXT,
        headers={"content-type": "text/plain"},
    )
    # Should not raise — just clamps internally
    result = await web_fetch("https://example.com", timeout=9999)
    assert isinstance(result, str)
