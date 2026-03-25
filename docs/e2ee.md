# Matrix E2EE

End-to-end encryption with Matrix is supported but has several sharp edges. This doc covers what we learned getting it working.

## How it works

Matrix E2EE uses two layers:

1. **Olm** — 1:1 sessions between device pairs (like Signal's double ratchet). Used to exchange Megolm keys.
2. **Megolm** — group sessions per room. One outbound session per sender device, shared with all trusted devices in the room.

When you send a message in an encrypted room, your client:
1. Creates a Megolm outbound session (if it doesn't have one)
2. Shares that session's key with each trusted device via Olm
3. Encrypts the message with the Megolm session

If a device isn't trusted, it won't get the Megolm key and can't decrypt.

## Known issues

### Bot and Element sharing a device

If you log into the bot account with Element **and** run aineko, both sessions may share the same `device_id`. They'll each generate their own Olm keys, overwriting each other. Result: neither can decrypt anything.

**Fix:** Only run one client per device. If you need Element for the bot account, stop aineko first. When done, log out of Element and restart aineko. You may need to wipe the crypto store after.

### Token invalidation on logout

Logging out of Element invalidates the access token. If aineko was using that token, it will get 401 errors.

**Fix:** After logging out of Element, get a new token via the login API (see deployment.md) and update your `.env`.

### Megolm session reuse

When a new device joins a room, existing clients may keep reusing their old Megolm session — which doesn't include the new device. The new device can't decrypt these messages even if it's trusted.

**Fix:** The sender client needs to rotate its Megolm session. This happens automatically when:
- The session exceeds a message/time threshold
- A new device is detected (but some clients cache aggressively)

Creating a **new room** guarantees a fresh Megolm session.

### Cross-signing and device trust

Matrix has two trust models:
- **Device trust** — manually verify each device
- **Cross-signing** — verify a user once, all their devices are trusted

If the bot's device isn't signed by the bot's cross-signing keys, other clients won't share Megolm keys with it. This happens when:
- Cross-signing was set up by Element, then Element was logged out
- The bot creates a new device that isn't cross-signed

**Fix:** Either set up cross-signing from the bot (complex), or use an unencrypted room.

### Quotes in .env files

Podman's `--env-file` does **not** strip quotes. If your `.env` has:
```
MATRIX_HOMESERVER="https://matrix.org"
```
The value will literally be `"https://matrix.org"` including the `"` characters, which breaks URLs.

**Fix:** Never quote values in `.env` files:
```
MATRIX_HOMESERVER=https://matrix.org
```

### Special characters in passwords

If your Matrix password contains `$`, podman may interpret it as a variable reference. The `.env` file format doesn't support escaping.

**Fix:** Use the access token approach instead of password login. Get a token via `curl` (see deployment.md).

## Recommended setup

For the simplest and most reliable experience:

1. Use an **unencrypted room** for the bot
2. Use **access token** auth (not password)
3. Don't log into the bot account with any other client while aineko is running
4. If you must reset, wipe `/data/crypto_store/` and get a fresh access token

E2EE is a nice-to-have for a personal bot on a trusted homeserver. The operational complexity may not be worth it unless you're on a shared server.
