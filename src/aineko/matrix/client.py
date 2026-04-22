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

import mimetypes

from aineko.config import MatrixSettings
from aineko.queue import MessageQueue
from aineko.schemas.message import IncomingMessage

logger = logging.getLogger(__name__)

MessageHandler = Callable[[IncomingMessage], Coroutine[Any, Any, None]]

_SEND_FILE_DATA_ROOT = Path("/data")

# File to persist login credentials (device_id + access_token) across restarts
_CREDS_FILE = "credentials.json"

# Fields that make a credentials file "complete" enough to restore a session.
_REQUIRED_CREDS = ("access_token", "device_id", "user_id")


def _load_credentials(creds_path: Path) -> dict[str, str]:
    """Load saved Matrix credentials, returning an empty dict on any issue.

    If the file exists but is missing required auth fields, it is deleted so
    a fresh login path can run. This prevents a stale file (e.g. one holding
    only a ``next_batch`` token from a previous run) from causing a KeyError
    in the background sync task and silently killing the Matrix connection.
    """
    if not creds_path.exists():
        return {}
    try:
        creds = json.loads(creds_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Matrix: failed to read credentials file, ignoring")
        creds_path.unlink(missing_ok=True)
        return {}
    if not all(k in creds for k in _REQUIRED_CREDS):
        logger.warning("Matrix: incomplete credentials file, ignoring")
        creds_path.unlink(missing_ok=True)
        return {}
    return creds


def _write_credentials(
    creds_path: Path,
    *,
    user_id: str,
    device_id: str,
    access_token: str,
    next_batch: str | None = None,
) -> None:
    """Write a complete credentials file (optionally with a sync token)."""
    creds: dict[str, str] = {
        "user_id": user_id,
        "device_id": device_id,
        "access_token": access_token,
    }
    if next_batch:
        creds["next_batch"] = next_batch
    creds_path.write_text(json.dumps(creds))


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

    async def inject_message(
        self,
        room_id: str,
        body: str,
        sender: str = "system",
        *,
        suppress_text_response: bool = False,
    ) -> None:
        """Inject a synthetic message into the processing queue."""
        if self._queue is None:
            logger.warning("inject_message called before queue initialized")
            return
        from datetime import datetime, timezone

        msg = IncomingMessage(
            room_id=room_id,
            sender=sender,
            body=body,
            timestamp=datetime.now(timezone.utc),
            event_id=f"$injected-{id(body)}",
            suppress_text_response=suppress_text_response,
        )
        await self._queue.enqueue(msg)

    async def start(self) -> None:
        self._running = True

        # Try to restore saved credentials, otherwise do a fresh login
        creds_path = self._store_path / _CREDS_FILE
        creds = _load_credentials(creds_path)

        if creds:
            self._client.access_token = creds["access_token"]
            self._client.device_id = creds["device_id"]
            self._client.user_id = creds["user_id"]
            if "next_batch" in creds:
                self._client.next_batch = creds["next_batch"]
            logger.info("Matrix: restored session, device_id=%s", creds["device_id"])
        elif self._settings.password:
            logger.info("Matrix: logging in with password...")
            resp = await self._client.login(
                password=self._settings.password,
                device_name="aineko",
            )
            if isinstance(resp, LoginResponse):
                logger.info("Matrix: logged in, device_id=%s", resp.device_id)
                _write_credentials(
                    creds_path,
                    user_id=resp.user_id,
                    device_id=resp.device_id,
                    access_token=resp.access_token,
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
            # Persist a complete credentials file so sync tokens can be saved
            # alongside auth on subsequent runs.
            _write_credentials(
                creds_path,
                user_id=self._client.user_id,
                device_id=self._client.device_id,
                access_token=self._settings.access_token,
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

        # Discover joined rooms first (no message callbacks yet) so we can
        # leave any foreign rooms before events start firing.
        self._client.add_event_callback(self._on_megolm_event, MegolmEvent)
        self._client.add_to_device_callback(self._on_key_request, RoomKeyRequest)
        self._client.add_event_callback(self._on_invite, InviteMemberEvent)

        logger.info("Matrix: initial sync...")
        resp = await self._client.sync(timeout=10_000, full_state=True)

        if self._settings.room_id:
            for room_id in list(self._client.rooms.keys()):
                if room_id != self._settings.room_id:
                    logger.warning("Matrix: leaving foreign room %s", room_id)
                    await self._client.room_leave(room_id)

        # Now safe to register message callbacks — ingress can assert room_id.
        self._client.add_event_callback(self._on_room_message, RoomMessageText)
        self._client.add_event_callback(self._on_room_file, RoomMessageFile)
        self._client.add_event_callback(self._on_room_file, RoomMessageImage)

        # Auto-join any pending invites — only when no owner is configured.
        # When owner is set we can't easily verify the inviter from the bulk
        # invited_rooms list, so we defer to the per-event _on_invite handler.
        if not self._settings.owner:
            for room_id in self._client.invited_rooms:
                logger.info("Matrix: auto-joining room %s", room_id)
                await self._client.join(room_id)

        # Persist next_batch so the next restart picks up where we left off
        self._save_sync_token()

        logger.info("Matrix: initial sync done, listening for messages")

        # Trust and establish Olm sessions in the background so we don't block
        asyncio.create_task(self._setup_crypto(), name="crypto-setup")

        # Enter sync loop
        while self._running:
            try:
                self._sync_task = asyncio.current_task()
                await self._client.sync(timeout=30_000)
                self._save_sync_token()
                await self._trust_all_devices()
            except asyncio.CancelledError:
                logger.info("Matrix: sync loop cancelled")
                break
            except Exception:
                logger.exception("Matrix sync error, retrying...")
                await asyncio.sleep(5)
        self._sync_task = None

    def _save_sync_token(self) -> None:
        """Persist next_batch token so the next restart resumes from here.

        Only updates an existing complete credentials file. Never writes a
        partial file with only a sync token, since that would cause the next
        startup to fail loading auth.
        """
        if not self._client.next_batch:
            return
        creds_path = self._store_path / _CREDS_FILE
        creds = _load_credentials(creds_path)
        if not creds:
            return
        creds["next_batch"] = self._client.next_batch
        creds_path.write_text(json.dumps(creds))

    async def stop(self) -> None:
        self._running = False
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
        self._save_sync_token()
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
        body = body.strip()
        if not body:
            return
        chunks = _chunk_message(body, max_len=4096)
        for chunk in chunks:
            content: dict[str, str] = {"msgtype": "m.text", "body": chunk}
            html = _markdown_to_html(chunk)
            if html != chunk:
                content["format"] = "org.matrix.custom.html"
                content["formatted_body"] = html
            await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
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

        assert (
            self._settings.room_id and room.room_id == self._settings.room_id
        ), f"received message from foreign room {room.room_id}"

        if self._settings.owner and event.sender != self._settings.owner:
            logger.warning(
                "Matrix: dropping message from non-owner %s in %s",
                event.sender,
                room.room_id,
            )
            return

        if self._queue is None:
            logger.warning("No message handler registered, dropping message")
            return

        from datetime import datetime, timezone

        body = await self._resolve_reply_body(room.room_id, event)

        msg = IncomingMessage(
            room_id=room.room_id,
            sender=event.sender,
            body=body,
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

    async def _resolve_reply_body(self, room_id: str, event: RoomMessageText) -> str:
        """Return the message body with reply context prepended.

        Tries the Matrix fallback quote first; if the client didn't include it
        (modern clients often omit it), fetches the original event via the API.
        """
        body = _format_reply_body(event)
        if body != event.body:
            return body  # fallback parse succeeded

        relates_to = event.source.get("content", {}).get("m.relates_to") or {}
        reply_to_id = (relates_to.get("m.in_reply_to") or {}).get("event_id")
        if not reply_to_id:
            return body  # not a reply

        # Fallback parse failed — fetch the original event
        try:
            from nio import RoomGetEventResponse

            resp = await self._client.room_get_event(room_id, reply_to_id)
            if isinstance(resp, RoomGetEventResponse):
                orig = resp.event
                orig_body = getattr(orig, "body", None) or ""
                orig_sender = getattr(orig, "sender", "")
                prefix = (
                    f'[in reply to {orig_sender}: "{orig_body[:300]}"]'
                    if orig_sender
                    else f'[in reply to: "{orig_body[:300]}"]'
                )
                return f"{prefix}\n{event.body}"
        except Exception:
            logger.debug("Could not fetch reply target %s", reply_to_id)

        return body

    async def _on_room_file(
        self, room: MatrixRoom, event: RoomMessageFile | RoomMessageImage
    ) -> None:
        """Handle file/image uploads — download, save, and pass to handler."""
        if (
            event.sender == self._settings.user_id
            or event.sender == self._client.user_id
        ):
            return
        assert (
            self._settings.room_id and room.room_id == self._settings.room_id
        ), f"received file from foreign room {room.room_id}"
        if self._settings.owner and event.sender != self._settings.owner:
            logger.warning(
                "Matrix: dropping file from non-owner %s in %s",
                event.sender,
                room.room_id,
            )
            return
        if self._queue is None:
            return

        # Download the file
        mxc_url = event.url
        filename = Path(getattr(event, "body", "file")).name
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
        """Auto-join the configured room — reject invites to any other room."""
        if event.state_key != self._settings.user_id or event.membership != "invite":
            return
        if self._settings.owner and event.sender != self._settings.owner:
            logger.warning(
                "Matrix: ignoring invite to %s from non-owner %s",
                room.room_id,
                event.sender,
            )
            return
        if self._settings.room_id and room.room_id != self._settings.room_id:
            logger.warning(
                "Matrix: rejecting invite to foreign room %s (configured: %s)",
                room.room_id,
                self._settings.room_id,
            )
            return
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


def _format_reply_body(event: RoomMessageText) -> str:
    """If the event is a reply, reformat the body so the original message is
    explicit for the LLM.

    Matrix spec puts a quoted fallback at the top of a reply body:

        > <@alice:example.com> original text
        > second line of original

        my reply text

    We turn that into:

        [in reply to @alice:example.com: "original text\\nsecond line of original"]
        my reply text

    so the model doesn't have to guess at the `>`-quote convention. If the event
    isn't a reply (no m.in_reply_to), body is returned unchanged.
    """
    relates_to = event.source.get("content", {}).get("m.relates_to") or {}
    if "m.in_reply_to" not in relates_to:
        return event.body

    logger.info(
        "reply detected",
        extra={
            "event": "reply_in",
            "in_reply_to": relates_to["m.in_reply_to"].get("event_id"),
            "sender": event.sender,
        },
    )

    lines = event.body.split("\n")
    quoted: list[str] = []
    i = 0
    while i < len(lines) and lines[i].startswith(">"):
        # Strip leading "> " or ">" marker
        quoted.append(lines[i][2:] if lines[i].startswith("> ") else lines[i][1:])
        i += 1
    # Skip one or more blank separator lines
    while i < len(lines) and lines[i] == "":
        i += 1
    new_body = "\n".join(lines[i:]).strip()

    if not quoted:
        return event.body  # malformed fallback — leave as-is

    # First quoted line is "<@user:server> first-line"; split off the sender.
    first = quoted[0]
    orig_sender = ""
    if first.startswith("<") and "> " in first:
        end = first.index("> ")
        orig_sender = first[1:end]
        quoted[0] = first[end + 2 :]
    original = "\n".join(quoted).strip()

    prefix = (
        f'[in reply to {orig_sender}: "{original}"]'
        if orig_sender
        else f'[in reply to: "{original}"]'
    )
    return f"{prefix}\n{new_body}" if new_body else prefix


def _markdown_to_html(text: str) -> str:
    """Convert markdown to HTML for Matrix formatted_body."""
    from markdown_it import MarkdownIt

    md = MarkdownIt()
    html = md.render(text).strip()
    # If the result is just a single <p>...</p>, unwrap it for simple messages
    if html.startswith("<p>") and html.endswith("</p>") and html.count("<p>") == 1:
        html = html[3:-4]
    return html


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


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


async def send_file(
    connector: "MatrixConnector",
    room_id: str,
    path: str,
    filename: str = "",
) -> str:
    """Upload a file from /data to a Matrix room.

    Args:
        connector: MatrixConnector instance.
        room_id: Target room.
        path: File path relative to /data.
        filename: Override display filename (defaults to basename).
    """
    resolved = (_SEND_FILE_DATA_ROOT / path).resolve()
    if not str(resolved).startswith(str(_SEND_FILE_DATA_ROOT)):
        return f"Error: path escapes data directory: {path}"
    if not resolved.is_file():
        return f"Error: file not found: {path}"

    display_name = filename or resolved.name
    file_size = resolved.stat().st_size
    mime, _ = mimetypes.guess_type(resolved.name)
    mime = mime or "application/octet-stream"

    import io

    data = io.BytesIO(resolved.read_bytes())

    resp, _ = await connector._client.upload(
        data,
        content_type=mime,
        filename=display_name,
        filesize=file_size,
    )

    if not hasattr(resp, "content_uri"):
        return f"Error: upload failed: {getattr(resp, 'message', resp)}"

    msgtype = "m.image" if resolved.suffix.lower() in _IMAGE_EXTENSIONS else "m.file"

    content: dict[str, Any] = {
        "msgtype": msgtype,
        "body": display_name,
        "url": resp.content_uri,
        "info": {
            "mimetype": mime,
            "size": file_size,
        },
    }

    await connector._client.room_send(
        room_id=room_id,
        message_type="m.room.message",
        content=content,
    )

    logger.info(
        "file sent",
        extra={
            "event": "file_out",
            "room": room_id,
            "filename": display_name,
            "size": file_size,
            "mimetype": mime,
        },
    )
    return f"Sent {display_name} ({file_size} bytes) to room"
