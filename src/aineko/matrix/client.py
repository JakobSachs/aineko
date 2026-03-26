"""Async Matrix client using matrix-nio with E2EE support."""

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from nio import (
    AsyncClient,
    AsyncClientConfig,
    DownloadResponse,
    InviteMemberEvent,
    LoginResponse,
    MatrixRoom,
    MegolmEvent,
    RoomKeyRequest,
    RoomMessageFile,
    RoomMessageImage,
    RoomMessageText,
)

from aineko.config import MatrixSettings
from aineko.queue import MessageQueue
from aineko.schemas.message import IncomingMessage

logger = logging.getLogger(__name__)

MessageHandler = Callable[[IncomingMessage], Coroutine[Any, Any, None]]

# File to persist login credentials (device_id + access_token) across restarts
_CREDS_FILE = "credentials.json"


class MatrixConnector:
    def __init__(self, settings: MatrixSettings, store_path: Path) -> None:
        self._settings = settings
        self._store_path = store_path

        # Ensure crypto store directory exists
        store_path.mkdir(parents=True, exist_ok=True)

        config = AsyncClientConfig(
            store_sync_tokens=True,
            encryption_enabled=True,
            store_name="aineko_crypto",
        )

        self._client = AsyncClient(
            settings.homeserver,
            settings.user_id,
            store_path=str(store_path),
            config=config,
        )

        self._handler: MessageHandler | None = None
        self._queue: MessageQueue | None = None
        self._running = False
        self._sync_task: asyncio.Task[None] | None = None

    def on_message(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._queue = MessageQueue(handler)

    async def start(self) -> None:
        self._running = True

        # Try to restore saved credentials, otherwise do a fresh login
        creds_path = self._store_path / _CREDS_FILE
        if creds_path.exists():
            creds = json.loads(creds_path.read_text())
            self._client.access_token = creds["access_token"]
            self._client.device_id = creds["device_id"]
            self._client.user_id = creds["user_id"]
            logger.info("Matrix: restored session, device_id=%s", creds["device_id"])
        elif self._settings.password:
            logger.info("Matrix: logging in with password...")
            resp = await self._client.login(
                password=self._settings.password,
                device_name="aineko",
            )
            if isinstance(resp, LoginResponse):
                logger.info("Matrix: logged in, device_id=%s", resp.device_id)
                # Save credentials for next restart
                creds_path.write_text(
                    json.dumps(
                        {
                            "user_id": resp.user_id,
                            "device_id": resp.device_id,
                            "access_token": resp.access_token,
                        }
                    )
                )
            else:
                logger.error("Matrix: login failed: %s", resp)
                return
        elif self._settings.access_token:
            import httpx

            self._client.access_token = self._settings.access_token
            self._client.user_id = self._settings.user_id
            # Resolve device_id via whoami
            async with httpx.AsyncClient() as http:
                r = await http.get(
                    f"{self._settings.homeserver}/_matrix/client/v3/account/whoami",
                    headers={"Authorization": f"Bearer {self._settings.access_token}"},
                )
                r.raise_for_status()
                self._client.device_id = r.json().get("device_id", "")
            logger.info(
                "Matrix: using access token, device_id=%s", self._client.device_id
            )
        else:
            logger.error("Matrix: no password or access_token configured")
            return

        # Load crypto store
        try:
            self._client.load_store()
            logger.info("Matrix: crypto store loaded")
        except Exception:
            logger.exception("Matrix: failed to load crypto store")

        if not self._client.olm:
            logger.warning("Matrix: olm not initialized, E2EE disabled")

        # Register callbacks
        self._client.add_event_callback(self._on_room_message, RoomMessageText)
        self._client.add_event_callback(self._on_room_file, RoomMessageFile)
        self._client.add_event_callback(self._on_room_file, RoomMessageImage)
        self._client.add_event_callback(self._on_megolm_event, MegolmEvent)
        self._client.add_to_device_callback(self._on_key_request, RoomKeyRequest)
        self._client.add_event_callback(self._on_invite, InviteMemberEvent)

        # Initial sync
        logger.info("Matrix: initial sync...")
        resp = await self._client.sync(timeout=10_000, full_state=True)
        if hasattr(resp, "next_batch"):
            self._client.next_batch = resp.next_batch

        # Auto-join any pending invites
        for room_id in self._client.invited_rooms:
            logger.info("Matrix: auto-joining room %s", room_id)
            await self._client.join(room_id)

        logger.info("Matrix: initial sync done, listening for messages")

        # Trust and establish Olm sessions in the background so we don't block
        asyncio.create_task(self._setup_crypto(), name="crypto-setup")

        # Enter sync loop
        while self._running:
            try:
                self._sync_task = asyncio.current_task()
                await self._client.sync(timeout=30_000)
                await self._trust_all_devices()
            except asyncio.CancelledError:
                logger.info("Matrix: sync loop cancelled")
                break
            except Exception:
                logger.exception("Matrix sync error, retrying...")
                await asyncio.sleep(5)
        self._sync_task = None

    async def stop(self) -> None:
        self._running = False
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
        await self._client.close()
        logger.info("Matrix: disconnected")

    async def send_message(self, room_id: str, body: str) -> None:
        """Send a text message to a Matrix room (auto-encrypted if room has E2EE)."""
        logger.info(
            "message sent",
            extra={
                "event": "msg_out",
                "room": room_id,
                "body": body[:500],
            },
        )
        chunks = _chunk_message(body, max_len=4096)
        for chunk in chunks:
            await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": chunk},
            )

    async def _setup_crypto(self) -> None:
        """Set up crypto in background: trust devices, query keys, claim OTKs."""
        try:
            await self._trust_all_devices()

            user_ids = set()
            for room in self._client.rooms.values():
                for user_id in room.users:
                    user_ids.add(user_id)
            if user_ids and self._client.olm:
                logger.info("Matrix: querying keys for %d users...", len(user_ids))
                await asyncio.wait_for(self._client.keys_query(), timeout=15)
                await self._trust_all_devices()

                users_to_claim: dict[str, list[str]] = {}
                for uid in user_ids:
                    devices = self._client.device_store.active_user_devices(uid)
                    unestablished = [
                        d.device_id
                        for d in devices
                        if not self._client.olm.session_store.get(d.curve25519)
                    ]
                    if unestablished:
                        users_to_claim[uid] = unestablished
                if users_to_claim:
                    logger.info("Matrix: claiming one-time keys for %s", users_to_claim)
                    await asyncio.wait_for(
                        self._client.keys_claim(users_to_claim), timeout=15
                    )
                    logger.info("Matrix: Olm sessions established")
        except asyncio.TimeoutError:
            logger.warning("Matrix: crypto setup timed out, will retry on next sync")
        except Exception:
            logger.exception("Matrix: crypto setup failed")

    async def _on_room_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        if event.sender == self._settings.user_id:
            logger.debug("Ignoring own message in %s", room.room_id)
            return
        # Also ignore messages from the client's resolved user_id (in case settings differ)
        if event.sender == self._client.user_id:
            logger.debug("Ignoring own message (client user_id) in %s", room.room_id)
            return

        if self._settings.room_list and room.room_id not in self._settings.room_list:
            return

        if self._queue is None:
            logger.warning("No message handler registered, dropping message")
            return

        from datetime import datetime, timezone

        msg = IncomingMessage(
            room_id=room.room_id,
            sender=event.sender,
            body=event.body,
            timestamp=datetime.fromtimestamp(
                event.server_timestamp / 1000, tz=timezone.utc
            ),
            event_id=event.event_id,
        )
        logger.info(
            "message received",
            extra={
                "event": "msg_in",
                "sender": msg.sender,
                "room": msg.room_id,
                "body": msg.body,
            },
        )

        await self._queue.enqueue(msg)

    async def _on_room_file(
        self, room: MatrixRoom, event: RoomMessageFile | RoomMessageImage
    ) -> None:
        """Handle file/image uploads — download, save, and pass to handler."""
        if (
            event.sender == self._settings.user_id
            or event.sender == self._client.user_id
        ):
            return
        if self._settings.room_list and room.room_id not in self._settings.room_list:
            return
        if self._queue is None:
            return

        # Download the file
        mxc_url = event.url
        filename = getattr(event, "body", "file")
        logger.info(
            "file received",
            extra={
                "event": "file_in",
                "sender": event.sender,
                "room": room.room_id,
                "body": filename,
            },
        )

        try:
            resp = await self._client.download(mxc_url)
            if not isinstance(resp, DownloadResponse):
                logger.error("file download failed: %s", resp)
                return

            # Save to /data/uploads/
            uploads_dir = self._store_path.parent / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            file_path = uploads_dir / filename
            file_path.write_bytes(resp.body)

            logger.info(
                "file saved",
                extra={
                    "event": "file_saved",
                    "tool": "download",
                    "result_len": len(resp.body),
                },
            )

            # Pass to handler — include base64 image data for vision if applicable
            import base64
            from datetime import datetime, timezone

            image_b64 = None
            image_mime = None
            IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
            if file_path.suffix.lower() in IMAGE_EXTS and len(resp.body) < 5_000_000:
                image_b64 = base64.b64encode(resp.body).decode()
                mime_map = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                image_mime = mime_map.get(file_path.suffix.lower(), "image/png")

            body = f"[file uploaded: /data/uploads/{filename} ({len(resp.body)} bytes)]"
            msg = IncomingMessage(
                room_id=room.room_id,
                sender=event.sender,
                body=body,
                timestamp=datetime.fromtimestamp(
                    event.server_timestamp / 1000, tz=timezone.utc
                ),
                event_id=event.event_id,
                image_b64=image_b64,
                image_mime=image_mime,
            )
            await self._queue.enqueue(msg)
        except Exception:
            logger.exception("Error handling file from %s", event.sender)

    async def _on_megolm_event(self, room: MatrixRoom, event: MegolmEvent) -> None:
        """Handle encrypted messages we couldn't decrypt."""
        logger.warning(
            "Matrix: could not decrypt message from %s in %s (session: %s)",
            event.sender,
            room.room_id,
            event.session_id,
        )

    async def _on_invite(self, room: MatrixRoom, event: InviteMemberEvent) -> None:
        """Auto-join rooms we're invited to."""
        if event.state_key == self._settings.user_id and event.membership == "invite":
            logger.info(
                "Matrix: invited to %s by %s, joining...", room.room_id, event.sender
            )
            await self._client.join(room.room_id)

    async def _on_key_request(self, event: RoomKeyRequest) -> None:
        """Log key requests."""
        logger.info(
            "Matrix: key request from %s device %s",
            event.sender,
            event.requesting_device_id,
        )

    async def _trust_all_devices(self) -> None:
        """Auto-trust all devices for all users in our rooms."""
        for room in self._client.rooms.values():
            for user_id in room.users:
                self._trust_user_devices(user_id)

    def _trust_user_devices(self, user_id: str) -> None:
        """Trust all known devices for a user."""
        if not self._client.olm:
            return
        devices = self._client.device_store.active_user_devices(user_id)
        for device in devices:
            if not self._client.olm.is_device_verified(device):
                self._client.verify_device(device)
                logger.info("Trusted device %s for %s", device.device_id, user_id)


def _chunk_message(text: str, max_len: int = 4096) -> list[str]:
    """Split a long message into chunks, preferring line breaks."""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, max_len)
        if cut <= 0:
            cut = max_len
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks
