"""Tests for cron job runner — success, failure, one-shot deletion, Matrix delivery."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import select

from aineko.cron.runner import run_cron_job
from aineko.db import init_engine, create_tables, dispose_engine, get_session
from aineko.kimi.client import ChatResponse
from aineko.models.cron import CronJob, CronRun, RunStatus, ScheduleKind
from aineko.tools.registry import ToolRegistry


@pytest_asyncio.fixture
async def setup_db(tmp_path):
    db_path = tmp_path / "test.db"
    init_engine(f"sqlite+aiosqlite:///{db_path}")
    await create_tables()
    yield
    await dispose_engine()


@pytest_asyncio.fixture
async def job(setup_db):
    """Insert a CronJob into the DB and return it."""
    async for db in get_session():
        j = CronJob(
            name="test-job",
            enabled=True,
            schedule_kind=ScheduleKind.EVERY,
            schedule_expr="30m",
            message="do the thing",
            delivery_room="!room:test",
            delete_after_run=False,
            consecutive_failures=0,
        )
        db.add(j)
        await db.commit()
        await db.refresh(j)
        return j


@pytest.fixture
def kimi():
    k = AsyncMock()
    k.chat_loop.return_value = ChatResponse(
        content="job output", usage={"total_tokens": 50}
    )
    return k


@pytest.fixture
def tools():
    return ToolRegistry()


@pytest.fixture
def skills():
    return MagicMock()


@pytest.fixture
def matrix():
    return AsyncMock()


class TestRunCronJob:
    @pytest.mark.asyncio
    async def test_success_records_run_and_delivers(
        self, job, kimi, tools, skills, matrix
    ):
        await run_cron_job(job, kimi, tools, skills, matrix, system_prompt="sys")

        kimi.chat_loop.assert_awaited_once()
        matrix.send_message.assert_awaited_once_with("!room:test", "job output")

        # Verify CronRun was recorded
        async for db in get_session():
            runs = (await db.execute(select(CronRun))).scalars().all()
            assert len(runs) == 1
            assert runs[0].status == RunStatus.SUCCESS
            assert runs[0].output == "job output"
            assert runs[0].finished_at is not None

    @pytest.mark.asyncio
    async def test_success_resets_failure_counter(
        self, job, kimi, tools, skills, matrix
    ):
        # Set up pre-existing failures
        async for db in get_session():
            result = await db.execute(select(CronJob).where(CronJob.id == job.id))
            db_job = result.scalar_one()
            db_job.consecutive_failures = 3
            await db.commit()

        await run_cron_job(job, kimi, tools, skills, matrix, system_prompt="sys")

        async for db in get_session():
            result = await db.execute(select(CronJob).where(CronJob.id == job.id))
            db_job = result.scalar_one()
            assert db_job.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_one_shot_job_deleted_after_run(
        self, setup_db, kimi, tools, skills, matrix
    ):
        async for db in get_session():
            j = CronJob(
                name="one-shot",
                enabled=True,
                schedule_kind=ScheduleKind.AT,
                schedule_expr="2026-01-01T00:00:00",
                message="once",
                delivery_room="!room:test",
                delete_after_run=True,
                consecutive_failures=0,
            )
            db.add(j)
            await db.commit()
            await db.refresh(j)
            job_id = j.id

        await run_cron_job(j, kimi, tools, skills, matrix, system_prompt="sys")

        async for db in get_session():
            result = await db.execute(select(CronJob).where(CronJob.id == job_id))
            assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_no_delivery_when_content_empty(
        self, job, kimi, tools, skills, matrix
    ):
        kimi.chat_loop.return_value = ChatResponse(content="", usage={})

        await run_cron_job(job, kimi, tools, skills, matrix, system_prompt="sys")

        matrix.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_delivery_when_no_room(
        self, setup_db, kimi, tools, skills, matrix
    ):
        async for db in get_session():
            j = CronJob(
                name="no-room",
                enabled=True,
                schedule_kind=ScheduleKind.EVERY,
                schedule_expr="1h",
                message="hi",
                delivery_room="",
                delete_after_run=False,
                consecutive_failures=0,
            )
            db.add(j)
            await db.commit()
            await db.refresh(j)

        await run_cron_job(j, kimi, tools, skills, matrix, system_prompt="sys")

        matrix.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failure_increments_counter(self, job, kimi, tools, skills, matrix):
        kimi.chat_loop.side_effect = RuntimeError("boom")

        await run_cron_job(job, kimi, tools, skills, matrix, system_prompt="sys")

        async for db in get_session():
            result = await db.execute(select(CronJob).where(CronJob.id == job.id))
            db_job = result.scalar_one()
            assert db_job.consecutive_failures == 1

            runs = (await db.execute(select(CronRun))).scalars().all()
            assert len(runs) == 1
            assert runs[0].status == RunStatus.FAILURE
            assert "boom" in runs[0].error
            assert runs[0].finished_at is not None

    @pytest.mark.asyncio
    async def test_system_prompt_passed_to_kimi(self, job, kimi, tools, skills, matrix):
        await run_cron_job(
            job, kimi, tools, skills, matrix, system_prompt="you are a cron bot"
        )

        messages = kimi.chat_loop.call_args[0][0]
        assert messages[0] == {"role": "system", "content": "you are a cron bot"}
        assert messages[1] == {"role": "user", "content": "do the thing"}
