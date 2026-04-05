"""Tests for Matrix credential file loading and persistence.

Regression coverage for the bug where a stale credentials.json holding only
``next_batch`` (written by _save_sync_token when the access_token branch
didn't persist auth) caused a KeyError in the background sync task and
silently killed the Matrix connection.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock


from aineko.matrix.client import (
    MatrixConnector,
    _load_credentials,
    _write_credentials,
)

# ---------- _load_credentials ----------


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert _load_credentials(tmp_path / "credentials.json") == {}


def test_load_complete_file_returns_creds(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    payload = {
        "user_id": "@me:matrix.org",
        "device_id": "DEV1",
        "access_token": "tok",
    }
    path.write_text(json.dumps(payload))
    assert _load_credentials(path) == payload


def test_load_complete_file_with_next_batch(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    payload = {
        "user_id": "@me:matrix.org",
        "device_id": "DEV1",
        "access_token": "tok",
        "next_batch": "s1234",
    }
    path.write_text(json.dumps(payload))
    assert _load_credentials(path) == payload


def test_load_only_next_batch_is_rejected_and_file_deleted(tmp_path: Path) -> None:
    """Regression: stale file with only next_batch must not KeyError."""
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"next_batch": "s1234"}))
    assert _load_credentials(path) == {}
    assert not path.exists()


def test_load_missing_access_token_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"user_id": "@me:matrix.org", "device_id": "DEV1"}))
    assert _load_credentials(path) == {}
    assert not path.exists()


def test_load_missing_device_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"user_id": "@me:matrix.org", "access_token": "tok"}))
    assert _load_credentials(path) == {}
    assert not path.exists()


def test_load_missing_user_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"device_id": "DEV1", "access_token": "tok"}))
    assert _load_credentials(path) == {}
    assert not path.exists()


def test_load_invalid_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    path.write_text("not valid json {{{")
    assert _load_credentials(path) == {}
    assert not path.exists()


# ---------- _write_credentials ----------


def test_write_credentials_without_next_batch(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    _write_credentials(
        path, user_id="@me:matrix.org", device_id="DEV1", access_token="tok"
    )
    data = json.loads(path.read_text())
    assert data == {
        "user_id": "@me:matrix.org",
        "device_id": "DEV1",
        "access_token": "tok",
    }


def test_write_credentials_with_next_batch(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    _write_credentials(
        path,
        user_id="@me:matrix.org",
        device_id="DEV1",
        access_token="tok",
        next_batch="s1234",
    )
    data = json.loads(path.read_text())
    assert data["next_batch"] == "s1234"


def test_write_credentials_roundtrip_through_load(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    _write_credentials(
        path, user_id="@me:matrix.org", device_id="DEV1", access_token="tok"
    )
    assert _load_credentials(path) == {
        "user_id": "@me:matrix.org",
        "device_id": "DEV1",
        "access_token": "tok",
    }


# ---------- _save_sync_token ----------


def _make_connector(tmp_path: Path, next_batch: str | None) -> MatrixConnector:
    """Build a MatrixConnector whose _client is a mock with a next_batch attr."""
    connector = MatrixConnector.__new__(MatrixConnector)
    connector._store_path = tmp_path
    connector._client = MagicMock()
    connector._client.next_batch = next_batch
    return connector


def test_save_sync_token_updates_existing_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    _write_credentials(
        path, user_id="@me:matrix.org", device_id="DEV1", access_token="tok"
    )

    connector = _make_connector(tmp_path, next_batch="s_new")
    connector._save_sync_token()

    data = json.loads(path.read_text())
    assert data["access_token"] == "tok"
    assert data["device_id"] == "DEV1"
    assert data["user_id"] == "@me:matrix.org"
    assert data["next_batch"] == "s_new"


def test_save_sync_token_noop_when_file_missing(tmp_path: Path) -> None:
    """Regression: must not create a partial file containing only next_batch."""
    connector = _make_connector(tmp_path, next_batch="s_new")
    connector._save_sync_token()
    assert not (tmp_path / "credentials.json").exists()


def test_save_sync_token_noop_when_file_is_partial(tmp_path: Path) -> None:
    """Regression: stale partial file must be cleaned up, not overwritten with more partial junk."""
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"next_batch": "s_old"}))

    connector = _make_connector(tmp_path, next_batch="s_new")
    connector._save_sync_token()

    # _load_credentials inside _save_sync_token removes the stale file and
    # returns {}, so nothing is written back.
    assert not path.exists()


def test_save_sync_token_noop_when_no_next_batch(tmp_path: Path) -> None:
    path = tmp_path / "credentials.json"
    _write_credentials(
        path, user_id="@me:matrix.org", device_id="DEV1", access_token="tok"
    )
    original = path.read_text()

    connector = _make_connector(tmp_path, next_batch=None)
    connector._save_sync_token()

    assert path.read_text() == original


def test_save_then_load_cycle_preserves_auth(tmp_path: Path) -> None:
    """End-to-end: write auth, save sync token, reload — auth must survive.

    This is the core regression: previously this cycle would drop the auth
    fields because _save_sync_token wrote a file containing only next_batch
    whenever the original file was missing.
    """
    path = tmp_path / "credentials.json"
    _write_credentials(
        path, user_id="@me:matrix.org", device_id="DEV1", access_token="tok"
    )

    connector = _make_connector(tmp_path, next_batch="s1234")
    connector._save_sync_token()

    creds = _load_credentials(path)
    assert creds["access_token"] == "tok"
    assert creds["device_id"] == "DEV1"
    assert creds["user_id"] == "@me:matrix.org"
    assert creds["next_batch"] == "s1234"
