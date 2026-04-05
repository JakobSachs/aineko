"""Tests for CronScheduler — trigger parsing and job scheduling."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from hypothesis import assume, given
from hypothesis import strategies as st

from aineko.cron.scheduler import CronScheduler, _parse_interval
from aineko.db import init_engine, create_tables, dispose_engine, get_session
from aineko.models.cron import CronJob, ScheduleKind

# --- Property-based tests ---


class TestParseIntervalProperties:
    @given(
        n=st.integers(min_value=1, max_value=10000),
        suffix=st.sampled_from(["s", "m", "h", "d"]),
    )
    def test_valid_input_produces_single_key(self, n: int, suffix: str):
        result = _parse_interval(f"{n}{suffix}")
        assert len(result) == 1

    @given(
        n=st.integers(min_value=1, max_value=10000),
        suffix=st.sampled_from(["s", "m", "h", "d"]),
    )
    def test_value_matches_input(self, n: int, suffix: str):
        result = _parse_interval(f"{n}{suffix}")
        assert list(result.values())[0] == n

    @given(
        n=st.integers(min_value=1, max_value=10000),
        suffix=st.sampled_from(["s", "m", "h", "d"]),
        padding=st.text(alphabet=" \t", max_size=5),
    )
    def test_whitespace_insensitive(self, n: int, suffix: str, padding: str):
        result = _parse_interval(f"{padding}{n}{suffix}{padding}")
        assert list(result.values())[0] == n

    @given(text=st.text(min_size=1, max_size=20))
    def test_invalid_input_raises(self, text: str):
        stripped = text.strip().lower()
        if stripped and stripped[-1] in "smhd":
            try:
                int(stripped[:-1])
                assume(False)  # skip valid inputs
            except ValueError:
                pass
        with pytest.raises((ValueError, Exception)):
            _parse_interval(text)


# --- Example-based tests ---


class TestParseInterval:
    def test_minutes(self):
        assert _parse_interval("30m") == {"minutes": 30}

    def test_hours(self):
        assert _parse_interval("2h") == {"hours": 2}

    def test_days(self):
        assert _parse_interval("1d") == {"days": 1}

    def test_seconds(self):
        assert _parse_interval("45s") == {"seconds": 45}

    def test_whitespace_stripped(self):
        assert _parse_interval("  10m  ") == {"minutes": 10}

    def test_case_insensitive(self):
        assert _parse_interval("5M") == {"minutes": 5}
        assert _parse_interval("3H") == {"hours": 3}

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown interval format"):
            _parse_interval("10x")

    def test_no_suffix_raises(self):
        with pytest.raises(ValueError):
            _parse_interval("10")


@pytest_asyncio.fixture
async def setup_db(tmp_path):
    db_path = tmp_path / "test.db"
    init_engine(f"sqlite+aiosqlite:///{db_path}")
    await create_tables()
    yield
    await dispose_engine()


class TestCronScheduler:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        scheduler = CronScheduler()
        scheduler.start()
        scheduler.stop()

    def test_set_runner(self):
        scheduler = CronScheduler()
        runner = AsyncMock()
        scheduler.set_runner(runner)
        assert scheduler._runner is runner

    @pytest.mark.asyncio
    async def test_run_job_calls_runner(self):
        scheduler = CronScheduler()
        runner = AsyncMock()
        scheduler.set_runner(runner)

        job = MagicMock()
        await scheduler._run_job(job)

        runner.assert_awaited_once_with(job)

    @pytest.mark.asyncio
    async def test_run_job_no_runner_set(self):
        scheduler = CronScheduler()
        job = MagicMock()
        # Should not raise, just log error
        await scheduler._run_job(job)

    @pytest.mark.asyncio
    async def test_run_job_handles_exception(self):
        scheduler = CronScheduler()
        runner = AsyncMock(side_effect=RuntimeError("fail"))
        scheduler.set_runner(runner)

        job = MagicMock()
        job.name = "test"
        # Should not raise
        await scheduler._run_job(job)

    @pytest.mark.asyncio
    async def test_load_jobs(self, setup_db):
        async for db in get_session():
            db.add(
                CronJob(
                    name="every-job",
                    enabled=True,
                    schedule_kind=ScheduleKind.EVERY,
                    schedule_expr="30m",
                    message="hi",
                    delivery_room="!room:test",
                )
            )
            db.add(
                CronJob(
                    name="disabled-job",
                    enabled=False,
                    schedule_kind=ScheduleKind.EVERY,
                    schedule_expr="1h",
                    message="nope",
                    delivery_room="!room:test",
                )
            )
            await db.commit()

        scheduler = CronScheduler()
        scheduler.set_runner(AsyncMock())

        async for db in get_session():
            await scheduler.load_jobs(db)

        # Only enabled job should be scheduled
        assert len(scheduler._scheduler.get_jobs()) == 1

    @pytest.mark.asyncio
    async def test_add_job_cron_trigger(self, setup_db):
        scheduler = CronScheduler()
        scheduler.set_runner(AsyncMock())

        job = MagicMock(spec=CronJob)
        job.id = 1
        job.schedule_kind = ScheduleKind.CRON
        job.schedule_expr = "*/5 * * * *"

        scheduler._add_job(job)
        assert len(scheduler._scheduler.get_jobs()) == 1

    @pytest.mark.asyncio
    async def test_add_job_every_trigger(self, setup_db):
        scheduler = CronScheduler()
        scheduler.set_runner(AsyncMock())

        job = MagicMock(spec=CronJob)
        job.id = 2
        job.schedule_kind = ScheduleKind.EVERY
        job.schedule_expr = "15m"

        scheduler._add_job(job)
        assert len(scheduler._scheduler.get_jobs()) == 1

    @pytest.mark.asyncio
    async def test_add_job_at_trigger(self, setup_db):
        scheduler = CronScheduler()
        scheduler.set_runner(AsyncMock())

        job = MagicMock(spec=CronJob)
        job.id = 3
        job.schedule_kind = ScheduleKind.AT
        job.schedule_expr = "2030-06-01T12:00:00"

        scheduler._add_job(job)
        assert len(scheduler._scheduler.get_jobs()) == 1
