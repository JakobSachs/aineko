"""One-shot SQLite → Postgres data migration.

Usage:
    python scripts/migrate_sqlite_to_postgres.py \
        --sqlite ./data/aineko.db \
        --postgres postgresql://aineko:aineko@127.0.0.1:5432/aineko \
        [--dry-run]

Reads every table from the sqlite file, bulk-inserts into the target postgres DB,
then realigns sequences. Target DB is expected to already have the schema
applied via `alembic upgrade head`.

Run target DB empty. The script aborts if any destination table is non-empty
(to avoid double-inserting a partial migration).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import asyncpg

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("migrate")

# FK-ordered: parents first
TABLES: list[str] = [
    "sessions",
    "cron_jobs",
    "cron_runs",
    "messages",
    "tool_logs",
    "rss_seen_items",
    "facts",
]

SEQUENCES: list[tuple[str, str]] = [(t, f"{t}_id_seq") for t in TABLES]


def sqlite_rows(conn: sqlite3.Connection, table: str) -> tuple[list[str], list[tuple]]:
    cur = conn.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return cols, rows


async def _pg_column_types(
    pg: asyncpg.Connection, table: str
) -> dict[str, str]:
    rows = await pg.fetch(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name=$1",
        table,
    )
    return {r["column_name"]: r["data_type"] for r in rows}


def _coerce_value(value: object, pg_type: str) -> object:
    if value is None:
        return value
    if pg_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            return value in ("1", "true", "True", "t")
        return value
    if isinstance(value, str):
        if pg_type.startswith("timestamp"):
            return datetime.fromisoformat(value)
        if pg_type == "date":
            return date.fromisoformat(value)
        # PG TEXT/VARCHAR cannot store NUL bytes
        if "\x00" in value:
            return value.replace("\x00", "")
    return value


# Tables where we must drop rows whose FK target is absent from the source DB
# (pre-migration SQLite had no FK enforcement, so orphans exist).
ORPHAN_FILTERS: dict[str, list[tuple[str, str, str]]] = {
    # target_table: [(fk_column, parent_table, parent_column), ...]
    "tool_logs": [
        ("session_id", "sessions", "id"),
        ("message_id", "messages", "id"),
    ],
    "messages": [("session_id", "sessions", "id")],
    "cron_runs": [("job_id", "cron_jobs", "id")],
}


def _filter_orphans(
    sq: sqlite3.Connection,
    table: str,
    cols: list[str],
    rows: list[tuple],
) -> list[tuple]:
    filters = ORPHAN_FILTERS.get(table)
    if not filters:
        return rows
    keep = rows
    for fk_col, parent_table, parent_col in filters:
        if fk_col not in cols:
            continue
        idx = cols.index(fk_col)
        valid = {
            r[0] for r in sq.execute(f"SELECT {parent_col} FROM {parent_table}")
        }
        before = len(keep)
        keep = [r for r in keep if r[idx] in valid]
        if before != len(keep):
            log.warning(
                "  %s: dropped %d orphan rows with missing %s",
                table,
                before - len(keep),
                fk_col,
            )
    return keep


async def copy_table(
    pg: asyncpg.Connection,
    table: str,
    cols: list[str],
    rows: list[tuple],
    dry_run: bool,
) -> int:
    if not rows:
        log.info("  %s: empty, skipping", table)
        return 0

    existing = await pg.fetchval(f"SELECT COUNT(*) FROM {table}")
    if existing:
        raise SystemExit(
            f"target table {table} has {existing} rows — refusing to migrate "
            "into non-empty DB"
        )

    types = await _pg_column_types(pg, table)
    # Skip columns that exist in sqlite but were removed from postgres
    keep_idx = [i for i, c in enumerate(cols) if c in types]
    dropped = [c for c in cols if c not in types]
    if dropped:
        log.info("  %s: dropping sqlite-only cols: %s", table, dropped)
    kept_cols = [cols[i] for i in keep_idx]
    col_types = [types[c] for c in kept_cols]
    coerced = [
        tuple(_coerce_value(row[i], t) for i, t in zip(keep_idx, col_types))
        for row in rows
    ]

    placeholders = ", ".join(f"${i + 1}" for i in range(len(kept_cols)))
    col_list = ", ".join(f'"{c}"' for c in kept_cols)
    stmt = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"

    if dry_run:
        log.info("  %s: would insert %d rows", table, len(rows))
        return len(rows)

    await pg.executemany(stmt, coerced)
    log.info("  %s: inserted %d rows", table, len(rows))
    return len(rows)


async def realign_sequences(pg: asyncpg.Connection, dry_run: bool) -> None:
    for table, seq in SEQUENCES:
        max_id = await pg.fetchval(f"SELECT COALESCE(MAX(id), 0) FROM {table}")
        if dry_run:
            log.info("  %s: would setval to %d", seq, max(max_id, 1))
            continue
        await pg.fetchval(f"SELECT setval('{seq}', {max(max_id, 1)})")
        log.info("  %s: setval to %d", seq, max(max_id, 1))


async def main_async(sqlite_path: Path, postgres_url: str, dry_run: bool) -> None:
    if not sqlite_path.exists():
        raise SystemExit(f"sqlite file not found: {sqlite_path}")

    log.info("Opening sqlite: %s", sqlite_path)
    sq = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)

    log.info("Connecting postgres: %s", postgres_url)
    # asyncpg wants plain postgres URL, not the SQLAlchemy prefix
    dsn = postgres_url.replace("postgresql+asyncpg://", "postgresql://")
    pg = await asyncpg.connect(dsn)

    try:
        if dry_run:
            log.info("DRY RUN — no writes will be made")

        log.info("Copying tables...")
        total = 0
        async with pg.transaction():
            for table in TABLES:
                cols, rows = sqlite_rows(sq, table)
                rows = _filter_orphans(sq, table, cols, rows)
                total += await copy_table(pg, table, cols, rows, dry_run)

            log.info("Realigning sequences...")
            await realign_sequences(pg, dry_run)

            if dry_run:
                # Force rollback even though no writes happened
                raise _DryRunAbort

        log.info("Done. Inserted %d rows across %d tables.", total, len(TABLES))
    except _DryRunAbort:
        log.info("Dry run complete — rolled back.")
    finally:
        await pg.close()
        sq.close()


class _DryRunAbort(Exception):
    pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", type=Path, default=Path("./data/aineko.db"))
    p.add_argument(
        "--postgres",
        default="postgresql://aineko:aineko@127.0.0.1:5432/aineko",
        help="Postgres DSN. Uses loopback binding from docker-compose by default.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    try:
        asyncio.run(main_async(args.sqlite, args.postgres, args.dry_run))
    except SystemExit:
        raise
    except Exception as e:
        log.error("Migration failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
