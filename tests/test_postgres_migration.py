"""Smoke tests for the Postgres migration: schema matches models, basic CRUD."""

import pytest
from sqlalchemy import inspect, select, text

from aineko.models.cron import CronJob, ScheduleKind
from aineko.models.message import Message, Role, Session
from aineko.models.rss import RssSeenItem


@pytest.mark.asyncio
async def test_alembic_created_all_tables(pg_engine):
    async with pg_engine.connect() as conn:
        tables = await conn.run_sync(lambda c: inspect(c).get_table_names())

    expected = {
        "alembic_version",
        "sessions",
        "cron_jobs",
        "cron_runs",
        "messages",
        "tool_logs",
        "rss_seen_items",
    }
    assert expected.issubset(set(tables)), f"missing: {expected - set(tables)}"


@pytest.mark.asyncio
async def test_session_message_insert_roundtrip(pg_session):
    s = Session(room_id="!room:test")
    pg_session.add(s)
    await pg_session.flush()

    pg_session.add(Message(session_id=s.id, role=Role.USER, content="hi"))
    await pg_session.commit()

    rows = (await pg_session.execute(select(Message))).scalars().all()
    assert len(rows) == 1
    assert rows[0].content == "hi"
    assert rows[0].role == Role.USER


@pytest.mark.asyncio
async def test_cron_enum_persisted(pg_session):
    j = CronJob(
        name="nightly",
        enabled=True,
        schedule_kind=ScheduleKind.CRON,
        schedule_expr="0 0 * * *",
        message="run",
        delivery_room="!r:test",
        delete_after_run=False,
        consecutive_failures=0,
    )
    pg_session.add(j)
    await pg_session.commit()

    raw = (
        await pg_session.execute(
            text("SELECT schedule_kind::text FROM cron_jobs WHERE name='nightly'")
        )
    ).scalar_one()
    assert raw == "CRON"


@pytest.mark.asyncio
async def test_rss_unique_conflict_do_nothing(pg_session):
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    row = {"feed_url": "https://x/feed", "guid": "abc", "title": "t"}
    await pg_session.execute(pg_insert(RssSeenItem).values([row]))
    await pg_session.execute(
        pg_insert(RssSeenItem)
        .values([row])
        .on_conflict_do_nothing(index_elements=["feed_url", "guid"])
    )
    await pg_session.commit()

    count = (
        await pg_session.execute(text("SELECT COUNT(*) FROM rss_seen_items"))
    ).scalar_one()
    assert count == 1


@pytest.mark.asyncio
async def test_isolation_between_tests_part1(pg_session):
    pg_session.add(Session(room_id="!leak:test"))
    await pg_session.commit()
    count = (
        await pg_session.execute(text("SELECT COUNT(*) FROM sessions"))
    ).scalar_one()
    assert count == 1


@pytest.mark.asyncio
async def test_isolation_between_tests_part2(pg_session):
    # If truncate didn't run, this sees the row from part1.
    count = (
        await pg_session.execute(text("SELECT COUNT(*) FROM sessions"))
    ).scalar_one()
    assert count == 0
