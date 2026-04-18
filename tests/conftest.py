"""Shared test fixtures.

Provides an opt-in Postgres fixture backed by testcontainers. Host must expose
a docker-compatible socket; on podman-rootless, `DOCKER_HOST` is auto-pointed
at the user podman socket and ryuk is disabled (ryuk misbehaves under rootless
podman).

Usage:

    async def test_something(pg_session):
        await pg_session.execute(...)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_ROOT = Path(__file__).resolve().parents[1]


def _ensure_podman_env() -> None:
    if "DOCKER_HOST" in os.environ:
        return
    uid = os.getuid()
    sock = Path(f"/run/user/{uid}/podman/podman.sock")
    if sock.exists():
        os.environ["DOCKER_HOST"] = f"unix://{sock}"
        # Ryuk (the testcontainers reaper) is flaky under rootless podman;
        # containers still get torn down by the stop() calls in our fixtures.
        os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")


@pytest.fixture(scope="session")
def pg_container():
    _ensure_podman_env()
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(
        "postgres:17-alpine",
        username="aineko",
        password="aineko",
        dbname="aineko",
        driver="asyncpg",
    ) as pg:
        yield pg


@pytest.fixture(scope="session")
def pg_url(pg_container) -> str:
    # testcontainers' get_connection_url honours the driver kwarg above,
    # returning `postgresql+asyncpg://...`.
    return pg_container.get_connection_url()


@pytest.fixture(scope="session")
def pg_url_migrated(pg_url: str) -> str:
    """Postgres URL with the alembic schema applied once per session."""
    import subprocess

    env = os.environ.copy()
    env["AINEKO_DATABASE_URL"] = pg_url
    subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        cwd=_ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    return pg_url


@pytest_asyncio.fixture
async def pg_engine(pg_url_migrated: str) -> AsyncIterator:
    """Per-test engine against the migrated session-scoped DB.

    Isolation: every test TRUNCATEs all user tables and resets sequences on
    teardown, so tests can assume an empty DB at start without paying for a
    new container or a fresh alembic run.
    """
    engine = create_async_engine(pg_url_migrated, pool_pre_ping=True)
    try:
        yield engine
    finally:
        async with engine.begin() as conn:
            result = await conn.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname='public' AND tablename != 'alembic_version'"
                )
            )
            tables = [r[0] for r in result]
            if tables:
                quoted = ", ".join(f'"{t}"' for t in tables)
                await conn.execute(
                    text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE")
                )
        await engine.dispose()


@pytest_asyncio.fixture
async def pg_session(pg_engine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def pg_db(pg_url_migrated: str) -> AsyncIterator[str]:
    """Initialize the global engine in `aineko.db` against the test container.

    Use this for tests that call code relying on `aineko.db.async_session_factory`
    instead of receiving a session directly.
    """
    import aineko.db as _db

    _db.init_engine(pg_url_migrated)
    try:
        yield pg_url_migrated
    finally:
        assert _db.engine is not None
        async with _db.engine.begin() as conn:
            result = await conn.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname='public' AND tablename != 'alembic_version'"
                )
            )
            tables = [r[0] for r in result]
            if tables:
                quoted = ", ".join(f'"{t}"' for t in tables)
                await conn.execute(
                    text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE")
                )
        await _db.dispose_engine()
        _db.engine = None
        _db.async_session_factory = None
