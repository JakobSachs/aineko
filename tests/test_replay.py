"""Tests for missed message replay on startup."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aineko.matrix.client import MatrixConnector
from aineko.queue import MessageQueue
from aineko.schemas.message import IncomingMessage


@dataclass
class FakeRoomMessageText:
    """Mimics nio.RoomMessageText for timeline events."""

    sender: str
    body: str
    server_timestamp: int  # ms since epoch
    event_id: str


@dataclass
class FakeTimeline:
    events: list = field(default_factory=list)


@dataclass
class FakeRoomInfo:
    timeline: FakeTimeline = field(default_factory=FakeTimeline)


@dataclass
class FakeJoinedRooms:
    join: dict = field(default_factory=dict)


@dataclass
class FakeSyncResponse:
    rooms: FakeJoinedRooms = field(default_factory=FakeJoinedRooms)
    next_batch: str = "s123"


def _make_event(sender: str, body: str, ts_offset_sec: int = 0) -> FakeRoomMessageText:
    """Create a fake text event with timestamp offset from now."""
    base = datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc)
    ts_ms = int((base.timestamp() + ts_offset_sec) * 1000)
    return FakeRoomMessageText(
        sender=sender,
        body=body,
        server_timestamp=ts_ms,
        event_id=f"$evt_{body.replace(' ', '_')}",
    )


@pytest.fixture
def connector():
    """Create a MatrixConnector with fake settings."""
    settings = MagicMock()
    settings.homeserver = "https://matrix.test"
    settings.user_id = "@bot:test"
    settings.access_token = "fake"
    settings.password = ""
    settings.room_list = ["!room:test"]

    import tempfile
    from pathlib import Path

    store = Path(tempfile.mkdtemp())

    conn = MatrixConnector(settings, store)
    return conn


@pytest.mark.asyncio
async def test_replay_finds_missed_messages(connector, monkeypatch):
    """Messages newer than last assistant reply are replayed."""
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        received.append(msg)

    connector._queue = MessageQueue(handler)
    connector._client.user_id = "@bot:test"

    # Patch RoomMessageText isinstance check to match our fakes
    import aineko.matrix.client as client_mod

    monkeypatch.setattr(client_mod, "RoomMessageText", FakeRoomMessageText)

    # Build a sync response with messages — one old (before our last reply), one new
    sync_resp = FakeSyncResponse(
        rooms=FakeJoinedRooms(
            join={
                "!room:test": FakeRoomInfo(
                    timeline=FakeTimeline(
                        events=[
                            _make_event(
                                "@user:test", "old message", ts_offset_sec=-100
                            ),
                            _make_event(
                                "@user:test", "missed message", ts_offset_sec=100
                            ),
                        ]
                    )
                )
            }
        )
    )

    # Mock DB: last assistant reply was at base time (offset 0)
    from datetime import datetime as dt

    last_reply = datetime(2026, 3, 25, 12, 0, 0)

    async def fake_get_session():
        mock_db = AsyncMock()
        # First call: select Session
        # Return a mock that chains .scalar_one_or_none()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = last_reply
        mock_db.execute = AsyncMock(return_value=result_mock)
        yield mock_db

    monkeypatch.setattr("aineko.db.get_session", fake_get_session)

    await connector._replay_missed_messages(sync_resp)
    await asyncio.sleep(2)  # wait for queue debounce

    assert len(received) == 1
    assert received[0].body == "missed message"


@pytest.mark.asyncio
async def test_replay_skips_own_messages(connector, monkeypatch):
    """Bot's own messages are not replayed."""
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        received.append(msg)

    connector._queue = MessageQueue(handler)
    connector._client.user_id = "@bot:test"

    import aineko.matrix.client as client_mod

    monkeypatch.setattr(client_mod, "RoomMessageText", FakeRoomMessageText)

    sync_resp = FakeSyncResponse(
        rooms=FakeJoinedRooms(
            join={
                "!room:test": FakeRoomInfo(
                    timeline=FakeTimeline(
                        events=[
                            _make_event(
                                "@bot:test", "my own message", ts_offset_sec=100
                            ),
                            _make_event(
                                "@user:test", "user message", ts_offset_sec=200
                            ),
                        ]
                    )
                )
            }
        )
    )

    # No previous replies in DB
    async def fake_get_session():
        mock_db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=result_mock)
        yield mock_db

    monkeypatch.setattr("aineko.db.get_session", fake_get_session)

    await connector._replay_missed_messages(sync_resp)
    await asyncio.sleep(2)

    assert len(received) == 1
    assert received[0].body == "user message"


@pytest.mark.asyncio
async def test_replay_no_messages_when_all_replied(connector, monkeypatch):
    """No replay when all messages are older than last reply."""
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        received.append(msg)

    connector._queue = MessageQueue(handler)
    connector._client.user_id = "@bot:test"

    import aineko.matrix.client as client_mod

    monkeypatch.setattr(client_mod, "RoomMessageText", FakeRoomMessageText)

    sync_resp = FakeSyncResponse(
        rooms=FakeJoinedRooms(
            join={
                "!room:test": FakeRoomInfo(
                    timeline=FakeTimeline(
                        events=[
                            _make_event(
                                "@user:test", "already handled", ts_offset_sec=-100
                            ),
                        ]
                    )
                )
            }
        )
    )

    # Last reply was after the message
    last_reply = datetime(2026, 3, 25, 12, 0, 0)

    async def fake_get_session():
        mock_db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = last_reply
        mock_db.execute = AsyncMock(return_value=result_mock)
        yield mock_db

    monkeypatch.setattr("aineko.db.get_session", fake_get_session)

    await connector._replay_missed_messages(sync_resp)
    await asyncio.sleep(2)

    assert len(received) == 0
