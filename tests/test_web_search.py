"""Happy path test for Brave web search tool."""

import pytest

from aineko.tools import web_search as web_search_mod
from aineko.tools.web_search import web_search

FAKE_BRAVE_RESPONSE = {
    "web": {
        "results": [
            {
                "title": "Python.org",
                "url": "https://www.python.org/",
                "description": "The official home of the Python Programming Language.",
            },
            {
                "title": "Python Tutorial",
                "url": "https://docs.python.org/3/tutorial/",
                "description": "An informal introduction to Python.",
            },
        ]
    }
}


@pytest.fixture(autouse=True)
def _set_brave_key(monkeypatch):
    monkeypatch.setattr(web_search_mod, "brave_api_key", "test-key-123")


@pytest.mark.asyncio
async def test_web_search_happy_path(httpx_mock):
    """Search returns formatted titles, URLs, and descriptions."""
    httpx_mock.add_response(json=FAKE_BRAVE_RESPONSE)

    result = await web_search("python programming")

    assert "Python.org" in result
    assert "https://www.python.org/" in result
    assert "official home" in result
    assert "Python Tutorial" in result


@pytest.mark.asyncio
async def test_web_search_no_results(httpx_mock):
    """Empty results returns a 'no results' message."""
    httpx_mock.add_response(json={"web": {"results": []}})

    result = await web_search("asdfghjklqwerty")

    assert "No results found" in result


@pytest.mark.asyncio
async def test_web_search_no_api_key(monkeypatch):
    """Missing API key returns an error."""
    monkeypatch.setattr(web_search_mod, "brave_api_key", "")

    result = await web_search("test")

    assert "not configured" in result


@pytest.mark.asyncio
async def test_web_search_http_error(httpx_mock):
    """HTTP errors return an error string, not an exception."""
    httpx_mock.add_response(status_code=500)

    result = await web_search("test query")

    assert "error" in result.lower()


@pytest.mark.asyncio
async def test_web_search_network_error(httpx_mock):
    """Network failures return an error string."""
    import httpx

    httpx_mock.add_exception(httpx.ConnectError("connection refused"))

    result = await web_search("test query")

    assert "error" in result.lower()
