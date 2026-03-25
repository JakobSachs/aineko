# aineko

> A personal AI gateway — named after the AI cat from Charles Stross's *Accelerando*.
>
> Inspired by [OpenClaw](https://github.com/openclaw/openclaw), stripped to essentials:
> single user, single agent, Matrix interface, Kimi brain, self-modifying skills.

---

## What aineko does

A self-hosted daemon that sits between you (on Matrix) and an AI (Kimi). You message your Matrix bot, it thinks, it replies. It can run commands, search the web, create its own skills, and wake itself up on a schedule to do things autonomously.

```
You (Element/Matrix client)
  ↕
aineko gateway (Docker)
  ├── Matrix connector     ← receives/sends messages
  ├── Kimi client          ← AI reasoning
  ├── Tool executor        ← bash, files, web search
  ├── Skills engine        ← self-modifying capabilities
  ├── Cron scheduler       ← timed tasks
  └── Heartbeat            ← periodic autonomous check-in
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Docker Compose                     │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │               aineko gateway                     │  │
│  │                                                  │  │
│  │   Matrix  ←───→  Core  ←───→  Kimi API          │  │
│  │                   │                              │  │
│  │         ┌─────────┼──────────┐                   │  │
│  │         │         │          │                   │  │
│  │     Sessions    Cron      Skills                 │  │
│  │         │     /Heartbeat    │                    │  │
│  │         │         │     File Watcher             │  │
│  │         │         │    (/data/skills/)           │  │
│  │         │         │         │                    │  │
│  │         └─────────┴─────────┘                    │  │
│  │              Tools                               │  │
│  │         (bash, files, web)                       │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  ┌────────────────┐  ┌────────────────────────────┐   │
│  │ Volume: /data   │  │ Volume: /data/skills       │   │
│  │ (state, config) │  │ (writable by agent)        │   │
│  └────────────────┘  └────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Matrix Connector

Connects to a Matrix homeserver as a bot user. Receives messages, sends replies.

**Responsibilities:**
- Join/listen on configured rooms
- Receive incoming messages, normalize to internal format
- Send agent replies back (chunked if needed — Matrix limit is ~65KB but practical limit ~4096 for readability)
- Handle media (images, files) if Kimi supports multimodal input
- Maintain connection lifecycle (sync loop, reconnect on failure)

**Not needed (vs OpenClaw):**
- No channel plugin abstraction — Matrix is hardcoded
- No pairing/allowlists — single user, trust the homeserver auth
- No multi-account — one bot, one homeserver

---

### 2. Kimi Client

Talks to the Kimi API for AI reasoning.

**Responsibilities:**
- Send conversation transcript + system prompt + tool definitions
- Stream responses back
- Handle tool calls (Kimi requests bash execution, file read, etc.)
- Track token usage
- Manage context window (truncate/summarize old messages if needed)

**Key design decision:** Kimi is the only provider. No provider abstraction, no model registry. If you add a second provider later, refactor then.

---

### 3. Session Store

Persists conversation state per Matrix room.

**Responsibilities:**
- Store message history as append-only JSONL (one file per room/session)
- Track metadata: last message timestamp, room ID, delivery state
- Provide transcript to Kimi client for context assembly
- Support isolated sessions for cron jobs (ephemeral, auto-cleaned)

**Storage layout:**
```
/data/sessions/
  {room-id}.jsonl          ← main conversation transcript
  cron/{job-id}/run-{uuid}.jsonl  ← ephemeral cron run transcripts (auto-cleaned)
```

---

### 4. Tools

Capabilities the agent can invoke during reasoning.

**Built-in tools:**
| Tool | What it does |
|------|-------------|
| `bash` | Execute shell commands in the container |
| `read_file` | Read file contents |
| `write_file` | Write/create files |
| `web_search` | Search the web (via DuckDuckGo or similar — no API key needed) |
| `create_skill` | Scaffold a new skill directory + SKILL.md |

**Not needed:**
- No tool plugin system — tools are hardcoded
- No approval workflows — single user, you trust your own agent
- No sandboxing beyond Docker — the container IS the sandbox

---

### 5. Skills Engine

The agent can create, modify, and use its own skills at runtime.

**How it works:**

```
/data/skills/
  skill-creator/           ← bundled, always present
    SKILL.md
    scripts/
      init_skill.sh
  server-monitor/          ← created by agent at runtime
    SKILL.md
    scripts/
      check.sh
    references/
      thresholds.md
```

**Skill format (SKILL.md):**
```markdown
---
name: server-monitor
description: Check server uptime and alert if services are down
---

## Instructions
[Full instructions the agent reads when this skill is triggered]

## Scripts
- `scripts/check.sh` — pings endpoints and returns status
```

**Lifecycle:**
1. Skills discovered by scanning `/data/skills/*/SKILL.md`
2. File watcher monitors for changes (add/edit/delete)
3. On change: skill list reloaded, version bumped
4. Next agent invocation includes fresh skill metadata in system prompt
5. Agent references skill by name, reads full SKILL.md + runs scripts as needed
6. Hot-reload: < 1 second from file write to availability

**Progressive disclosure (saves tokens):**
- Level 1: Only `name` + `description` from frontmatter always in context (~100 tokens each)
- Level 2: Full SKILL.md body loaded only when the agent decides to use that skill
- Level 3: Scripts/references loaded on-demand via `read_file`

**Bundled skill — skill-creator:**
- Always present at `/data/skills/skill-creator/`
- Agent uses it to scaffold new skills
- Creates directory structure, template SKILL.md, empty scripts/

---

### 6. Cron Scheduler

Runs agent tasks on a schedule.

**Job definition (stored in `/data/cron/jobs.json`):**
```json
{
  "id": "uuid",
  "name": "Daily digest",
  "enabled": true,
  "schedule": { "kind": "cron", "expr": "0 9 * * *" },
  "message": "Summarize what happened in my Matrix rooms yesterday",
  "delivery": { "room": "!abc:matrix.org" },
  "deleteAfterRun": false
}
```

**Schedule types:**
| Type | Format | Use case |
|------|--------|----------|
| `cron` | `"0 9 * * *"` | Recurring (daily, hourly, etc.) |
| `every` | `30m`, `2h` | Fixed interval |
| `at` | ISO datetime | One-shot (auto-deleted after run) |

**Execution flow:**
```
Timer fires (job is due)
  ↓
Run agent in isolated session with job's message
  ↓
Agent produces output
  ↓
Deliver output to configured Matrix room
  ↓
Update job state (nextRun, lastStatus, etc.)
  ↓
Clean up ephemeral session after 24h
```

**Failure handling:**
- Consecutive error counter per job
- Configurable alert threshold (e.g., alert after 2 failures)
- Failure notifications sent to Matrix

**Management:** Agent can create/edit/delete cron jobs via tools (read/write `/data/cron/jobs.json`).

---

### 7. Heartbeat

Periodic autonomous agent check-in — with smart pre-filtering to avoid wasting API calls.

**HEARTBEAT.md** — a file at `/data/HEARTBEAT.md` where you (or the agent) write standing instructions:
```markdown
# Heartbeat Tasks

- Check if my server at example.com is responding
- Summarize any unread Matrix messages in !room:matrix.org
```

**Decision tree (before invoking Kimi):**
```
Heartbeat timer fires (e.g. every 30 min)
  │
  ├─ Disabled?                         → skip
  ├─ Outside active hours?             → skip "quiet-hours"
  ├─ User message currently in flight? → skip "busy"
  │
  ├─ Event-triggered? (cron result, etc.)
  │    YES → always invoke agent (bypass file check)
  │    NO  → check HEARTBEAT.md:
  │          ├─ Effectively empty?*     → skip "nothing-to-do"
  │          └─ Has real tasks?         → invoke agent
  │
  ╰─ INVOKE AGENT with HEARTBEAT.md content + any pending events
       │
       ├─ Agent responds "HEARTBEAT_OK" → suppress (don't send to Matrix)
       ├─ Duplicate of last response?   → suppress
       └─ Real content?                 → deliver to Matrix room
```

*\* "Effectively empty" = only whitespace, bare markdown headers, or empty checkboxes.*

**Config:**
```json
{
  "heartbeat": {
    "every": "30m",
    "activeHours": { "start": "08:00", "end": "23:00" },
    "room": "!notifications:matrix.org"
  }
}
```

---

### 8. Config

Flat JSON file + environment variables. No hot-reload needed — restart the container for config changes.

**`/data/config.json`:**
```json
{
  "matrix": {
    "homeserver": "https://matrix.example.org",
    "userId": "@aineko:example.org",
    "accessToken": "@env:MATRIX_ACCESS_TOKEN",
    "rooms": ["!room1:example.org"]
  },
  "kimi": {
    "apiKey": "@env:KIMI_API_KEY",
    "model": "kimi-latest",
    "maxContextTokens": 128000
  },
  "heartbeat": {
    "every": "30m",
    "activeHours": { "start": "08:00", "end": "23:00" },
    "room": "!room1:example.org"
  },
  "cron": {
    "enabled": true,
    "maxConcurrentRuns": 1
  }
}
```

**Secret resolution:** Values prefixed with `@env:` resolve from environment variables (passed via Docker Compose).

---

## Message Flow

### Inbound (you → aineko → Kimi → you)

```
You send "what's the weather?" in Element
  ↓
Matrix connector receives sync event
  ↓
Normalize: { room, sender, text, timestamp, attachments? }
  ↓
Load session transcript for this room
  ↓
Assemble prompt:
  ├── System prompt (persona, available tools, skill summaries)
  ├── Conversation history (from session JSONL)
  └── New user message
  ↓
Send to Kimi API (streaming)
  ↓
Kimi may call tools:
  ├── bash("curl wttr.in/Berlin?format=3") → tool result
  ├── [loop back to Kimi with result]
  └── Kimi produces final text response
  ↓
Chunk response if needed (practical ~4096 char chunks)
  ↓
Send to Matrix room
  ↓
Append both user message + agent response to session JSONL
```

### Heartbeat (autonomous)

```
Timer fires (every 30m)
  ↓
Pre-filter: active hours? busy? HEARTBEAT.md empty?
  ↓ (passes)
Load HEARTBEAT.md + any pending cron/system events
  ↓
Run agent with heartbeat prompt
  ↓
Agent checks tasks, produces report (or "HEARTBEAT_OK")
  ↓
If real content → send to configured Matrix room
If just OK → suppress, save API cost
```

### Cron (scheduled)

```
Cron timer fires (job is due)
  ↓
Run agent in isolated session with job's message
  ↓
Agent executes, produces output
  ↓
Deliver to Matrix room configured in job
  ↓
Clean up isolated session after 24h
```

### Skill creation (self-modification)

```
You: "create a skill that monitors my Hetzner server"
  ↓
Agent invokes skill-creator, then:
  ├── write_file("/data/skills/hetzner-monitor/SKILL.md", ...)
  ├── write_file("/data/skills/hetzner-monitor/scripts/check.sh", ...)
  ↓
File watcher detects new SKILL.md → reload skill list
  ↓
Next invocation: agent sees "hetzner-monitor" in available skills
  ↓
Agent can now use it: read SKILL.md for instructions, run check.sh
```

---

## Data Layout

```
/data/                              ← Docker volume, persists across restarts
  config.json                       ← main config
  HEARTBEAT.md                      ← standing heartbeat instructions
  sessions/
    {room-id}.jsonl                 ← conversation transcripts
    cron/
      {job-id}/
        run-{uuid}.jsonl            ← ephemeral cron run transcripts
  cron/
    jobs.json                       ← cron job definitions + state
    runs/
      {job-id}.jsonl                ← run log (append-only, auto-pruned)
  skills/
    skill-creator/                  ← bundled
      SKILL.md
      scripts/
        init_skill.sh
    {agent-created-skill}/          ← created at runtime
      SKILL.md
      scripts/
      references/
```

---

## Docker Setup

**`docker-compose.yml`:**
```yaml
services:
  aineko:
    build: .
    restart: unless-stopped
    volumes:
      - aineko-data:/data
    environment:
      - MATRIX_ACCESS_TOKEN=${MATRIX_ACCESS_TOKEN}
      - KIMI_API_KEY=${KIMI_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/healthz"]
      interval: 30s
      retries: 3

volumes:
  aineko-data:
```

**`.env`:**
```
MATRIX_ACCESS_TOKEN=syt_...
KIMI_API_KEY=sk-...
```

Single container. No sidecar services. The container IS the sandbox — the agent's bash tool runs inside it.

---

## What We Took from OpenClaw

| Feature | OpenClaw | aineko |
|---------|----------|-------|
| Channel system | 20 channels via plugin abstraction | Matrix hardcoded |
| AI providers | 38 providers via plugin registry | Kimi hardcoded |
| Routing | Binding-driven, multi-agent, multi-user | Single user, single agent, single room set |
| Skills | File-based discovery + watcher + skill-creator | Same approach, simplified |
| Cron | Full scheduler with webhooks, failure alerts, delivery modes | Same core, Matrix-only delivery |
| Heartbeat | 9-gate pre-filter, active hours, HEARTBEAT.md, OK suppression | Same decision tree, simplified |
| Sessions | Multi-user, composite keys, identity linking | One session per Matrix room |
| Tools | Plugin-provided, approval workflows, sandboxed | Hardcoded set (bash, files, web search), no approvals |
| Config | JSON5, hot-reload, secret refs, migration | Flat JSON + @env: refs |
| Deployment | Docker (3-tier sandbox), Fly.io, Render, K8s, native apps | Single Docker container |
| Security | Pairing, allowlists, CODEOWNERS, role-based | Trust the Matrix homeserver auth |
| Apps | macOS, iOS, Android, Web UI | None — Matrix client is the UI |

---

## What We Dropped

- Plugin SDK and extension system (82 extensions)
- Multi-channel abstraction
- Multi-user routing, pairing, allowlists
- Native apps (macOS, iOS, Android)
- Web UI control panel
- Voice/TTS/speech
- Multi-agent orchestration
- Hook system
- MCP protocol
- i18n
- Browser automation / canvas host
- CI/CD pipelines

---

## Future Maybes

| Feature | When | Effort |
|---------|------|--------|
| Email channel | When Matrix isn't enough | Low — add IMAP/SMTP alongside Matrix |
| Web search upgrade | If DuckDuckGo is too limited | Low — swap search implementation |
| Second AI provider | If Kimi isn't enough for some tasks | Medium — add provider selection per task |
| Simple web dashboard | For monitoring/logs | Medium — static page served by gateway |
| Image understanding | If Kimi supports vision | Low — pass image URLs/base64 to Kimi |
| Context compaction | Long conversations exceed context window | Medium — summarize old messages |
