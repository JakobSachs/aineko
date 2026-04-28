"""Tests for Matrix file sending and receiving."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

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


# --- Incoming file/audio/video handling ---


@dataclass
class FakeDownloadError:
    message: str


def make_connector_with_queue(tmp_path):
    from aineko.config import MatrixSettings

    settings = MatrixSettings(
        homeserver="https://matrix.example.com",
        user_id="@bot:example.com",
        password="secret",
        room_id="!room:example.com",
        owner="@owner:example.com",
    )
    connector = MatrixConnector(settings, tmp_path / "store")
    connector._client = AsyncMock()
    connector._client.user_id = "@bot:example.com"
    connector._store_path = tmp_path / "store"
    queue = AsyncMock()
    connector._queue = queue
    return connector, queue


def make_file_event(
    msgtype="m.audio", filename="voice.ogg", url="mxc://example.com/audio1"
):
    from nio import RoomMessageAudio, RoomMessageFile, RoomMessageVideo

    cls = {
        "m.audio": RoomMessageAudio,
        "m.video": RoomMessageVideo,
        "m.file": RoomMessageFile,
    }[msgtype]
    event = MagicMock(spec=cls)
    event.sender = "@owner:example.com"
    event.url = url
    event.body = filename
    event.server_timestamp = 1_700_000_000_000
    event.event_id = "$evt:audio"
    return event


def make_room(room_id="!room:example.com"):
    room = MagicMock()
    room.room_id = room_id
    return room


@pytest.mark.asyncio
async def test_incoming_audio_enqueued(tmp_path):
    """Voice notes (m.audio) are downloaded and enqueued as file references."""
    connector, queue = make_connector_with_queue(tmp_path)
    from nio import DownloadResponse

    connector._client.download.return_value = DownloadResponse(
        b"ogg-data", "audio/ogg", "voice.ogg"
    )

    event = make_file_event(msgtype="m.audio", filename="voice.ogg")
    await connector._on_room_file(make_room(), event)

    queue.enqueue.assert_called_once()
    msg = queue.enqueue.call_args[0][0]
    assert "voice.ogg" in msg.body
    assert msg.image_b64 is None


@pytest.mark.asyncio
async def test_incoming_video_enqueued(tmp_path):
    """Video messages (m.video) are downloaded and enqueued as file references."""
    connector, queue = make_connector_with_queue(tmp_path)
    from nio import DownloadResponse

    connector._client.download.return_value = DownloadResponse(
        b"mp4-data", "video/mp4", "clip.mp4"
    )

    event = make_file_event(msgtype="m.video", filename="clip.mp4")
    await connector._on_room_file(make_room(), event)

    queue.enqueue.assert_called_once()
    msg = queue.enqueue.call_args[0][0]
    assert "clip.mp4" in msg.body
    assert msg.image_b64 is None


@pytest.mark.asyncio
async def test_incoming_file_enqueued(tmp_path):
    """Generic file attachments (m.file) are downloaded and enqueued."""
    connector, queue = make_connector_with_queue(tmp_path)
    from nio import DownloadResponse

    connector._client.download.return_value = DownloadResponse(
        b"pdf-data", "application/pdf", "doc.pdf"
    )

    event = make_file_event(msgtype="m.file", filename="doc.pdf")
    await connector._on_room_file(make_room(), event)

    queue.enqueue.assert_called_once()
    msg = queue.enqueue.call_args[0][0]
    assert "doc.pdf" in msg.body


@pytest.mark.asyncio
async def test_incoming_file_foreign_room_dropped(tmp_path):
    """Files from foreign rooms are silently dropped."""
    connector, queue = make_connector_with_queue(tmp_path)

    event = make_file_event(msgtype="m.audio", filename="voice.ogg")
    await connector._on_room_file(make_room(room_id="!other:example.com"), event)

    queue.enqueue.assert_not_called()


@pytest.mark.asyncio
async def test_incoming_file_non_owner_dropped(tmp_path):
    """Files from non-owners are dropped when owner is configured."""
    connector, queue = make_connector_with_queue(tmp_path)

    event = make_file_event(msgtype="m.audio", filename="voice.ogg")
    event.sender = "@stranger:example.com"
    await connector._on_room_file(make_room(), event)

    queue.enqueue.assert_not_called()


@pytest.mark.asyncio
async def test_incoming_file_download_failure_ignored(tmp_path):
    """Download errors don't crash the handler."""
    connector, queue = make_connector_with_queue(tmp_path)
    connector._client.download.return_value = FakeDownloadError(message="server error")

    event = make_file_event(msgtype="m.audio", filename="voice.ogg")
    await connector._on_room_file(make_room(), event)

    queue.enqueue.assert_not_called()
