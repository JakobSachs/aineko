# SQLite → Postgres Migration Guide

Plan for moving aineko's persistence from SQLite (`/data/aineko.db`) to a Postgres container colocated in `docker-compose.yml`.

## Decisions (locked in)

- **Tests**: use Postgres in tests too (via testcontainers or a dedicated test DB). Same dialect everywhere — a SQLite-in-tests / Postgres-in-prod split will bite us later.
- **Alembic**: squash the three existing revisions into a single fresh `initial postgres schema` revision.
- **Postgres version**: `postgres:17-alpine` (track minor).
- **Network**: Postgres listens on the compose network AND binds `127.0.0.1:5432` on the host for `psql` / `pg_dump` from outside the container.
- **PR order**: (1) restructure `handle_message` into short-lived sessions on current SQLite, validate; (2) Postgres migration as a second PR.
- **Data migration script**: I write it with a `--dry-run` flag; the user runs the final cutover.
- **`idle_in_transaction_session_timeout`**: TBD — leave off initially, revisit after observing normal behavior.

## Why

- Concurrent writers no longer serialize on a single file lock (the root cause of the recent `database is locked` / cron-tool failures).
- Proper JSON column types (future-friendly for `tool_logs.arguments`, assistant block lists).
- Network-accessible DB opens the door to out-of-band inspection/backup without pausing the app.
- Removes the self-inflicted-damage surface (agent's bash tool can't `rm /data/aineko.db-journal` if there's no journal file).

Cost: one more container, ~50 MB RAM floor, a real backup strategy (pg_dump), and a one-time data migration.

## Scope of changes

Six areas, in roughly the order to touch them:

1. **Dependencies** — swap `aiosqlite` for `asyncpg`
2. **Config** — default `db_url` becomes Postgres; keep SQLite override for tests
3. **Engine (`src/aineko/db.py`)** — drop SQLite PRAGMA listener, add Postgres pool args
4. **Alembic (`alembic/env.py`, `alembic.ini`, existing revisions)** — read URL from env, fix SQLite-only SQL, add missing `rss_seen_items` migration
5. **Compose** — add `postgres` service, volume, healthcheck, depends_on
6. **Data transfer** — one-shot copy of existing rows from `./data/aineko.db` into the new DB
7. **Tests** — keep SQLite-in-memory (fast) OR switch to testcontainers (realistic)

---

## 1. Dependencies (`pyproject.toml`)

```diff
-  "aiosqlite>=0.20,<1",
+  "asyncpg>=0.30,<1",
```

`sqlalchemy[asyncio]>=2.0` and `alembic>=1.14` already support Postgres — no change needed.

Keep `aiosqlite` as a dev/test extra if we go with option A for tests:

```toml
[dependency-groups]
dev = ["aiosqlite>=0.20,<1", ...existing...]
```

## 2. Config (`src/aineko/config.py`)

Change the default in `Settings.db_url`:

```python
@property
def db_url(self) -> str:
    if self.database_url:
        return self.database_url
    return "postgresql+asyncpg://aineko:aineko@postgres:5432/aineko"
```

Runtime URL is expected to come from `AINEKO_DATABASE_URL` in `.env`. The hardcoded fallback only matters for local dev outside the container.

Add to `.env`:

```
AINEKO_DATABASE_URL=postgresql+asyncpg://aineko:${POSTGRES_PASSWORD}@postgres:5432/aineko
POSTGRES_PASSWORD=<generate>
POSTGRES_USER=aineko
POSTGRES_DB=aineko
```

## 3. Engine (`src/aineko/db.py`)

Strip the SQLite-only code:

- Remove the `event.listens_for(engine.sync_engine, "connect")` PRAGMA listener — Postgres has none of those pragmas.
- Remove `connect_args={"timeout": 30}` (it's a sqlite3 arg; asyncpg uses `command_timeout` and a different pool model).
- Add a sensible pool: `pool_size=5, max_overflow=10, pool_pre_ping=True`.

```python
engine = create_async_engine(
    database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)
```

Keep `async_sessionmaker(engine, expire_on_commit=False)` as-is.

The `handler.py` early-commit fix (committing the user message before `kimi.chat_loop`) is still correct and should stay — it's good hygiene regardless of backend.

## 4. Alembic

### `alembic/env.py`

Currently reads `sqlalchemy.url` from `alembic.ini` (hardcoded to a relative SQLite path — the bug the README calls out). Fix:

```python
import os
from aineko.config import Settings

url = os.environ.get("AINEKO_DATABASE_URL") or Settings().db_url
config.set_main_option("sqlalchemy.url", url)
```

Also remove `render_as_batch=True` from both `run_migrations_offline` and `do_run_migrations` — batch mode is only needed to work around SQLite's missing ALTER TABLE. On Postgres it's harmless but misleading.

### `alembic.ini`

Blank out the hardcoded URL; env.py now owns it:

```ini
sqlalchemy.url =
```

### Existing revisions

Three files under `alembic/versions/`:

- `b485554b83a7_initial_schema.py`
- `a8c607db08a0_add_tool_logs_table.py`
- `b1d02f4e9c37_drop_cron_jobs_next_run.py`

Issues in each:

- `server_default=sa.text('(CURRENT_TIMESTAMP)')` — the parentheses are SQLite-specific. Postgres parses `(CURRENT_TIMESTAMP)` as a parenthesized expression which usually works but is non-idiomatic. Change to `sa.func.now()` or `sa.text('CURRENT_TIMESTAMP')`.
- `batch_alter_table(...)` calls are fine to leave (no-ops on Postgres) but can be simplified to direct `op.drop_column`, `op.create_index`, etc.

Because we're starting with a fresh Postgres DB, a cleaner path is:

- **Squash**: delete the three revisions, run `alembic revision --autogenerate -m "initial postgres schema"` against an empty Postgres, commit the single new revision. The old SQLite DB is migrated separately via data export (see §6) — its schema history doesn't need to survive.

### Dialect-specific INSERT in `rss/poller.py`

`src/aineko/rss/poller.py:88` uses `from sqlalchemy.dialects.sqlite import insert as sqlite_insert` with `.on_conflict_do_nothing()`. That's SQLite-only — it will fail on Postgres. Swap for the Postgres dialect:

```python
from sqlalchemy.dialects.postgresql import insert as pg_insert
...
stmt = pg_insert(RssSeenItem).values(rows).on_conflict_do_nothing(
    index_elements=["feed_url", "guid"],
)
```

Note: Postgres's `on_conflict_do_nothing` requires `index_elements` (or `constraint=...`) to identify which conflict target to ignore. SQLite's version inferred it.

### `rss_seen_items` cleanup

The table is append-only dedup state — the poller inserts on every seen item, nothing ever deletes. Once a guid rolls off a feed's own window it will never be seen again, so old rows are dead weight. Low volume today (hundreds/month), but worth handling while we're touching this area.

Add a periodic cleanup: `DELETE FROM rss_seen_items WHERE seen_at < now() - interval '90 days'`. Easiest home is the existing cron scheduler — register it as a built-in job at startup.

### Missing migration: `rss_seen_items`

The RSS table is only created by `db.create_tables()` on startup; there is no alembic revision for it. Fold it into the squashed revision (autogenerate will pick it up since the model is imported via `aineko.models`).

## 5. Compose (`docker-compose.yml`)

```yaml
services:
  postgres:
    image: postgres:17-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 5s
      timeout: 3s
      retries: 10
    # no ports: — aineko talks to it over the compose network only

  aineko:
    # ...existing config...
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  chroma-cache:
  postgres-data:
```

Note: keep `postgres` unexposed on the host. If you want to connect from the host for `psql` or `pg_dump`, add `ports: ["127.0.0.1:5432:5432"]` — bind to loopback only.

Host-side backups: `podman exec aineko_postgres_1 pg_dump -U aineko aineko > backup.sql` on a cron.

Also drop `restart: unless-stopped` from the `aineko` service per the open note in the README (systemd owns restarts for the app, but postgres can keep it since systemd doesn't manage the DB directly).

## 6. Data transfer

Small dataset (thousands of rows, not millions). Simple ETL via SQLAlchemy — no need for `pgloader`.

### Steps

1. Stop aineko: `systemctl --user stop aineko`.
2. Bring up just Postgres: `podman-compose up -d postgres`.
3. Run migrations against the new Postgres: `podman-compose run --rm aineko alembic upgrade head`.
4. Run the one-shot copy script (new file, `scripts/migrate_sqlite_to_postgres.py`):
   - Opens a read-only sqlite connection on `./data/aineko.db`.
   - Opens an asyncpg connection on the new Postgres using `AINEKO_DATABASE_URL`.
   - For each table in FK order (`sessions`, `messages`, `cron_jobs`, `cron_runs`, `tool_logs`, `rss_seen_items`): `SELECT *` from sqlite, bulk `INSERT` into postgres.
   - After all tables: `SELECT setval(pg_get_serial_sequence(...), MAX(id))` on each to realign id sequences so future inserts don't collide.
5. Spot-check row counts: `SELECT count(*)` per table against the sqlite source.
6. Rename the sqlite file out of the way: `mv ./data/aineko.db ./data/aineko.db.preswitch` (don't delete — it's the only backup until you take a pg_dump).
7. Start aineko: `systemctl --user start aineko`. Watch logs for successful startup, scheduled cron jobs, and first message round-trip.

### Sequence realignment

Postgres uses sequences for `Integer` primary keys; sqlite uses AUTOINCREMENT on rowid. After copying, Postgres's sequence is still at 1, so the next insert would collide with existing ids. Run:

```sql
SELECT setval('sessions_id_seq',   COALESCE((SELECT MAX(id) FROM sessions),   1));
SELECT setval('messages_id_seq',   COALESCE((SELECT MAX(id) FROM messages),   1));
SELECT setval('cron_jobs_id_seq',  COALESCE((SELECT MAX(id) FROM cron_jobs),  1));
SELECT setval('cron_runs_id_seq',  COALESCE((SELECT MAX(id) FROM cron_runs),  1));
SELECT setval('tool_logs_id_seq',  COALESCE((SELECT MAX(id) FROM tool_logs),  1));
SELECT setval('rss_seen_items_id_seq', COALESCE((SELECT MAX(id) FROM rss_seen_items), 1));
```

Include this as the final step of the migration script.

## 7. Tests

Current tests use `sqlite+aiosqlite:///:memory:` or tmp-path sqlite files. Two options:

- **Option A (recommended, minimal change)**: keep SQLite for unit tests. Tests aren't about backend fidelity — they're about handler/tool logic. Add a fixture that sets `AINEKO_DATABASE_URL` to the sqlite test URL before `init_engine` is called, and keep `aiosqlite` in the `dev` dependency group.

- **Option B**: switch to `testcontainers-python` or an ephemeral Postgres via `pytest-postgresql`. More realistic (catches Postgres-specific type/default issues) but ~5× slower and adds Docker-in-tests complexity. Worth it only if we hit a behavioral divergence.

Start with A; escalate to B if bugs slip through.

## 8. Concurrency model (what actually changes)

The SQLite failure we hit was a `database is locked` error on a cron-tool INSERT that ran concurrently with `handle_message` while `handle_message` was mid-LLM-call. Understanding why that specific failure goes away — and what *doesn't* — matters for how we structure the code post-migration.

### What goes away

- **File-level writer lock.** SQLite serializes every writer behind one lock per file (WAL helps readers, not writers). Any second writer hits `busy_timeout` and errors. This is *the* reason the cron INSERT failed while the handler was stalled in `kimi.chat_loop`.
- **Postgres**: MVCC with row-level locks. Two transactions writing different rows — or even different tables — don't contend. Readers never block writers; writers never block readers. The cron-handler collision simply cannot happen the way it did on SQLite.

### What's still a problem

Swapping the backend masks the symptom but not the shape of the code. Three things still bite on Postgres:

1. **Connection pool starvation.** `app.py:130` opens `async with _db.async_session_factory() as db` and keeps that connection checked out from the pool through `load_conversation` → `kimi.chat_loop` → `persist_response`. With `pool_size=5, max_overflow=10`, that's 15 concurrent requests before the 16th waits on `pool_timeout` (default 30s) and errors. aiosqlite didn't have a real pool; Postgres does. This limit is real.

2. **Idle-in-transaction anti-pattern.** A transaction left open across a network call holds its MVCC snapshot, shows up in `pg_stat_activity` as `idle in transaction`, and blocks `VACUUM` from reclaiming dead rows in tables the snapshot touched. Most Postgres deployments eventually set `idle_in_transaction_session_timeout` (30s–5min) as a guardrail — once that's on, a stalled handler session gets killed mid-request. The early-commit fix in `handler.py` already shortens the open transaction, but the connection itself stays checked out.

3. **Row-level deadlocks (theoretical).** Postgres detects deadlocks and aborts one side with `DeadlockDetected` — loud, not silent. Our schema has no multi-row update patterns that would realistically deadlock, so this is a future concern, not a current one.

### The right shape

`handle_message` should use **multiple short-lived sessions**, not one session spanning the whole request. Each session: open → do DB work → commit → close → connection returns to pool. The LLM call happens *between* sessions, not inside one.

Three natural scopes in `app.py:handle_message`:

- **Load scope**: command check, load history, build message list, persist user message, handle compaction. Commit and close before calling the LLM.
- **(LLM call happens here, holding no DB connection.)**
- **Persist scope**: save assistant response, tool logs. Open fresh session, commit, close.

Tools that need DB during the LLM call (cron, rss, etc.) already do the right thing — they use `async for db in get_session()` to grab their own short-lived session. No change needed there.

Pseudocode:

```python
async def handle_message(...):
    async with _db.async_session_factory() as db:
        if await handle_command(db, msg, matrix):
            return
        session, user_msg, messages = await load_conversation(db, msg, sys_prompt)
        # compaction logic here, commit
        session_id = session.id
        user_msg_id = user_msg.id
    # No DB connection held across this call:
    response = await kimi.chat_loop(messages, request_tools, on_intermediate=...)

    async with _db.async_session_factory() as db:
        # Re-fetch Session / user_msg by id, or pass ids directly into persist_response
        await persist_response(db, session_id, user_msg_id, response, sent_messages)
```

`persist_response` currently takes `Session` and `Message` ORM objects — refactor to take IDs instead, so they can cross the session boundary without detached-instance headaches.

### Why do this even if Postgres "would work" without it

- Pool sizing becomes predictable. 5 concurrent LLM calls no longer pin 5 connections doing nothing.
- `idle_in_transaction_session_timeout` becomes safe to enable.
- Same pattern the cron/rss tools already use — consistent shape across the codebase.
- Works identically on SQLite (the tests still pass), so the restructure lands cleanly before the backend swap.

**Recommendation**: do the `handle_message` restructure as a *separate PR before* the Postgres switch. That way we validate the new shape on the backend we already understand, and the Postgres migration becomes purely an infra change.

## 9. Rollback plan

If Postgres goes sideways within the first few days:

1. `systemctl --user stop aineko`.
2. Revert the aineko image to the pre-migration commit.
3. `mv ./data/aineko.db.preswitch ./data/aineko.db`.
4. Unset `AINEKO_DATABASE_URL` in `.env`.
5. `systemctl --user start aineko`.

You lose any messages/cron runs that happened post-switch. Acceptable for a personal assistant; if that becomes unacceptable, write a reverse export (`postgres → sqlite`) before deciding rollback is permanent.

## 10. Open questions

- **Backups**: no automated backup exists today. After switching, add a cron that runs `pg_dump` into `./data/backups/` with rotation. Out of scope for the migration itself but should follow immediately.
- **Secrets**: `.env` will gain `POSTGRES_PASSWORD`. Confirm it's gitignored (it already is) and rotate if ever accidentally committed.
- **JSON columns**: we currently store JSON in `Text` columns (`messages.content` for tool-call blocks, `tool_logs.arguments`). Postgres has native `JSONB`. Out of scope for this migration — do it as a separate follow-up once the infra is stable, because it requires code changes in the handler's `json.loads`/`json.dumps` call sites.
