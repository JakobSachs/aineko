# Deployment

aineko runs as a single container via podman (or docker).

## Prerequisites

- podman (or docker) + podman-compose
- A Matrix account for the bot
- An LLM API key (OpenRouter, OpenAI, Kimi, or any OpenAI-compatible provider)

## Quick start

```bash
cp .env.example .env
# Edit .env with your credentials (see below)
podman-compose up -d
podman logs -f aineko_aineko_1
```

## Environment variables

All values should be **unquoted** in the `.env` file. Podman's `--env-file` does not strip quotes, so `MATRIX_HOMESERVER="https://matrix.org"` will include the literal `"` characters.

```
# Correct
MATRIX_HOMESERVER=https://matrix.org

# WRONG — will break URLs
MATRIX_HOMESERVER="https://matrix.org"
```

### Matrix

| Variable | Description |
|----------|------------|
| `MATRIX_HOMESERVER` | Homeserver URL, e.g. `https://matrix.org` |
| `MATRIX_USER_ID` | Bot's full user ID, e.g. `@aineko:matrix.org` |
| `MATRIX_ACCESS_TOKEN` | Access token from login (see below) |
| `MATRIX_PASSWORD` | Bot's password (alternative to access_token) |
| `MATRIX_ROOM_ID` | Room ID to listen in (single room) |

### LLM API

The LLM client is OpenAI-compatible. It works with any provider that implements the `/chat/completions` endpoint.

| Variable | Description | Default |
|----------|------------|---------|
| `KIMI_API_KEY` | API key for your LLM provider | |
| `KIMI_BASE_URL` | API base URL | `https://api.moonshot.cn/v1` |
| `KIMI_MODEL` | Model name | `kimi-latest` |
| `KIMI_USER_AGENT` | Custom User-Agent header (required by some endpoints) | |

**Examples:**

```bash
# OpenRouter (recommended — access to many models)
KIMI_API_KEY=sk-or-...
KIMI_BASE_URL=https://openrouter.ai/api/v1
KIMI_MODEL=moonshotai/kimi-k2.5

# OpenAI
KIMI_API_KEY=sk-...
KIMI_BASE_URL=https://api.openai.com/v1
KIMI_MODEL=gpt-4.1

# Moonshot/Kimi (developer API — separate from consumer subscription)
KIMI_API_KEY=sk-...
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_MODEL=kimi-latest

# Kimi Coding (subscription API — requires Kimi Code membership)
KIMI_API_KEY=sk-kimi-...
KIMI_BASE_URL=https://api.kimi.com/coding/v1
KIMI_MODEL=kimi-for-coding
KIMI_USER_AGENT=claude-code/0.1.0
```

Note: the Moonshot developer API at `api.moonshot.cn` requires a key from [platform.moonshot.cn](https://platform.moonshot.cn). A Kimi consumer subscription does **not** include API access.

## Matrix bot setup

### 1. Create a bot account

Register a new Matrix account for the bot on your homeserver (or matrix.org).

### 2. Get an access token

```bash
curl -s -X POST https://matrix.org/_matrix/client/v3/login \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "m.login.password",
    "identifier": {"type": "m.id.user", "user": "YOUR_BOT_USERNAME"},
    "password": "YOUR_BOT_PASSWORD"
  }'
```

The response contains `access_token` and `device_id`. Put the `access_token` in your `.env`.

**Important:** Do not log into the bot account with Element (or any other Matrix client) at the same time as running aineko. Both sessions will compete for encryption keys and cause decryption failures. If you need to manage the bot account via Element, stop aineko first.

### 3. Get the room ID

Create a room, invite the bot, and find the room ID in your Matrix client (Element: Room Settings > Advanced > Internal room ID). It looks like `!AbCdEf123:matrix.org`. Put it in `MATRIX_ROOM_ID`.

### 4. Encrypted vs unencrypted rooms

aineko supports E2EE via matrix-nio and libolm. However, getting E2EE working reliably requires:

- The bot's device must be trusted by all participants
- Megolm sessions must include the bot's device
- No other client (like Element) should be logged into the bot account simultaneously

**For the simplest setup, use an unencrypted room.** Create the room with encryption disabled. This avoids all key exchange issues and the bot will work immediately.

If you need E2EE, see [e2ee.md](e2ee.md) for details.

## Rebuilding

```bash
podman-compose down
podman build -t aineko_aineko .
podman-compose up -d
```

## Persistent data

All data lives in the `aineko-data` podman volume, mounted at `/data` inside the container:

- `/data/aineko.db` — SQLite database (sessions, messages, cron jobs)
- `/data/crypto_store/` — Matrix E2EE keys (olm/megolm state)
- `/data/skills/` — Skill definitions (hot-reloaded)

To wipe the crypto store (e.g. after token rotation):

```bash
podman run --rm -v aineko_aineko-data:/data alpine sh -c "rm -rf /data/crypto_store/*"
```
