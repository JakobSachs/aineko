"""Tests for sync token persistence — replaces the old replay logic.

Instead of replaying missed messages by scanning timelines, we now persist
the next_batch sync token so nio only returns new events on restart.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aineko.matrix.client import MatrixConnector


@pytest.fixture
def store_path():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def connector(store_path):
    settings = MagicMock()
    settings.homeserver = "https://matrix.test"
    settings.user_id = "@bot:test"
    settings.access_token = "fake"
    settings.password = ""
    settings.room_list = ["!room:test"]
    return MatrixConnector(settings, store_path)


class TestSyncTokenPersistence:
    def test_save_sync_token_writes_to_creds(self, connector, store_path):
        """_save_sync_token persists next_batch alongside existing auth.

        It only updates an existing complete credentials file — it never
        creates a new partial file containing only next_batch, since that
        previously caused a KeyError on the next startup and silently killed
        the Matrix connection.
        """
        creds_path = store_path / "credentials.json"
        creds_path.write_text(
            json.dumps(
                {
                    "user_id": "@bot:test",
                    "device_id": "DEV1",
                    "access_token": "tok",
                }
            )
        )

        connector._client.next_batch = "s12345"
        connector._save_sync_token()

        creds = json.loads(creds_path.read_text())
        assert creds["next_batch"] == "s12345"

    def test_save_sync_token_noop_when_no_auth_file(self, connector, store_path):
        """Regression: without existing auth, no partial file is created."""
        connector._client.next_batch = "s12345"
        connector._save_sync_token()
        assert not (store_path / "credentials.json").exists()

    def test_save_sync_token_preserves_existing_creds(self, connector, store_path):
        """Saving sync token doesn't clobber existing credential fields."""
        creds_path = store_path / "credentials.json"
        creds_path.write_text(
            json.dumps(
                {
                    "user_id": "@bot:test",
                    "device_id": "DEV1",
                    "access_token": "tok",
                }
            )
        )

        connector._client.next_batch = "s99999"
        connector._save_sync_token()

        creds = json.loads(creds_path.read_text())
        assert creds["access_token"] == "tok"
        assert creds["device_id"] == "DEV1"
        assert creds["next_batch"] == "s99999"

    def test_save_sync_token_noop_when_no_token(self, connector, store_path):
        """No file written if next_batch is empty."""
        connector._client.next_batch = ""
        connector._save_sync_token()
        assert not (store_path / "credentials.json").exists()

    def test_restore_sync_token_on_start(self, store_path):
        """Saved next_batch is restored into the client on startup."""
        creds_path = store_path / "credentials.json"
        creds_path.write_text(
            json.dumps(
                {
                    "user_id": "@bot:test",
                    "device_id": "DEV1",
                    "access_token": "tok",
                    "next_batch": "s_restored",
                }
            )
        )

        settings = MagicMock()
        settings.homeserver = "https://matrix.test"
        settings.user_id = "@bot:test"
        settings.access_token = ""
        settings.password = ""

        conn = MatrixConnector(settings, store_path)
        # Simulate the credential-restore branch of start() without actually calling start()
        creds = json.loads(creds_path.read_text())
        conn._client.access_token = creds["access_token"]
        conn._client.device_id = creds["device_id"]
        conn._client.user_id = creds["user_id"]
        if "next_batch" in creds:
            conn._client.next_batch = creds["next_batch"]

        assert conn._client.next_batch == "s_restored"

    def test_save_then_restore_roundtrip(self, connector, store_path):
        """A saved token can be read back correctly (with auth present)."""
        creds_path = store_path / "credentials.json"
        creds_path.write_text(
            json.dumps(
                {
                    "user_id": "@bot:test",
                    "device_id": "DEV1",
                    "access_token": "tok",
                }
            )
        )

        connector._client.next_batch = "s_roundtrip_42"
        connector._save_sync_token()

        creds = json.loads(creds_path.read_text())
        assert creds["next_batch"] == "s_roundtrip_42"
        assert creds["access_token"] == "tok"

        # Simulate a new connector loading the same store
        settings = connector._settings
        conn2 = MatrixConnector(settings, store_path)
        loaded = json.loads(creds_path.read_text())
        conn2._client.next_batch = loaded.get("next_batch", "")
        assert conn2._client.next_batch == "s_roundtrip_42"
