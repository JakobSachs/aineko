"""Tests for Matrix file sending."""

from unittest.mock import AsyncMock, patch
from dataclasses import dataclass

import pytest

from aineko.matrix.client import MatrixConnector, send_file


@dataclass
class FakeUploadResponse:
    content_uri: str


@dataclass
class FakeUploadError:
    message: str


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point DATA_ROOT at a temp directory for send_file."""
    import aineko.matrix.client as mod

    monkeypatch.setattr(mod, "_SEND_FILE_DATA_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def matrix_connector(tmp_path):
    """Create a MatrixConnector with a mocked nio client."""
    from aineko.config import MatrixSettings

    settings = MatrixSettings(
        homeserver="https://matrix.example.com",
        user_id="@bot:example.com",
        password="secret",
    )
    connector = MatrixConnector(settings, tmp_path / "store")
    connector._client = AsyncMock()
    return connector


# --- send_file standalone function ---


@pytest.mark.asyncio
async def test_send_file_text(data_dir, matrix_connector):
    """Send a plain text file."""
    (data_dir / "notes.txt").write_text("hello world")

    matrix_connector._client.upload.return_value = (
        FakeUploadResponse(content_uri="mxc://example.com/abc123"),
        None,
    )
    matrix_connector._client.room_send.return_value = AsyncMock()

    result = await send_file(matrix_connector, "!room:example.com", "notes.txt")

    assert "sent" in result.lower()
    matrix_connector._client.upload.assert_called_once()
    matrix_connector._client.room_send.assert_called_once()

    # Check the room_send content
    call_kwargs = matrix_connector._client.room_send.call_args
    content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content")
    assert content["msgtype"] == "m.file"
    assert content["body"] == "notes.txt"
    assert content["url"] == "mxc://example.com/abc123"
    assert content["info"]["mimetype"] == "text/plain"
    assert content["info"]["size"] == 11


@pytest.mark.asyncio
async def test_send_file_image(data_dir, matrix_connector):
    """Images use m.image msgtype."""
    (data_dir / "photo.png").write_bytes(b"\x89PNG" + b"\x00" * 100)

    matrix_connector._client.upload.return_value = (
        FakeUploadResponse(content_uri="mxc://example.com/img456"),
        None,
    )
    matrix_connector._client.room_send.return_value = AsyncMock()

    result = await send_file(matrix_connector, "!room:example.com", "photo.png")

    assert "sent" in result.lower()
    call_kwargs = matrix_connector._client.room_send.call_args
    content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content")
    assert content["msgtype"] == "m.image"
    assert content["body"] == "photo.png"


@pytest.mark.asyncio
async def test_send_file_custom_filename(data_dir, matrix_connector):
    """Override the displayed filename."""
    (data_dir / "report.csv").write_text("a,b,c")

    matrix_connector._client.upload.return_value = (
        FakeUploadResponse(content_uri="mxc://example.com/csv789"),
        None,
    )
    matrix_connector._client.room_send.return_value = AsyncMock()

    result = await send_file(
        matrix_connector, "!room:example.com", "report.csv", filename="data.csv"
    )

    assert "sent" in result.lower()
    call_kwargs = matrix_connector._client.room_send.call_args
    content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content")
    assert content["body"] == "data.csv"


@pytest.mark.asyncio
async def test_send_file_not_found(data_dir, matrix_connector):
    """Missing file returns error."""
    result = await send_file(matrix_connector, "!room:example.com", "nope.txt")
    assert "error" in result.lower()
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_send_file_path_traversal(data_dir, matrix_connector):
    """Block path traversal."""
    result = await send_file(matrix_connector, "!room:example.com", "../../etc/passwd")
    assert "error" in result.lower()


@pytest.mark.asyncio
async def test_send_file_upload_error(data_dir, matrix_connector):
    """Upload failure returns error."""
    (data_dir / "doc.txt").write_text("content")

    matrix_connector._client.upload.return_value = (
        FakeUploadError(message="upload failed"),
        None,
    )
    matrix_connector._client.room_send.return_value = AsyncMock()

    result = await send_file(matrix_connector, "!room:example.com", "doc.txt")
    assert "error" in result.lower()
    matrix_connector._client.room_send.assert_not_called()


@pytest.mark.asyncio
async def test_send_file_subdirectory(data_dir, matrix_connector):
    """Files in subdirectories work."""
    sub = data_dir / "uploads"
    sub.mkdir()
    (sub / "file.txt").write_text("nested")

    matrix_connector._client.upload.return_value = (
        FakeUploadResponse(content_uri="mxc://example.com/nested"),
        None,
    )
    matrix_connector._client.room_send.return_value = AsyncMock()

    result = await send_file(matrix_connector, "!room:example.com", "uploads/file.txt")
    assert "sent" in result.lower()
