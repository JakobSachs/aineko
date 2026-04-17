"""Cron scheduler using APScheduler, backed by SQLAlchemy."""

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aineko.models.cron import CronJob, ScheduleKind

logger = logging.getLogger(__name__)

# Callback type: receives the CronJob and should run the agent + deliver output
JobRunner = Callable[[CronJob], Coroutine[Any, Any, None]]


class CronScheduler:
    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler()
        self._runner: JobRunner | None = None

    def set_runner(self, runner: JobRunner) -> None:
        self._runner = runner

    async def load_jobs(self, session: AsyncSession) -> None:
        """Load all enabled jobs from DB and schedule them."""
        result = await session.execute(select(CronJob).where(CronJob.enabled.is_(True)))
        jobs = result.scalars().all()

        for job in jobs:
            self.add_job(job)

        logger.info("Scheduled %d cron jobs", len(jobs))

    def add_job(self, job: CronJob) -> None:
        """Schedule a job (or replace its existing schedule)."""
        self._add_job(job)

    def remove_job(self, job_id: int) -> None:
        """Remove a job from the scheduler. No-op if not scheduled."""
        from apscheduler.jobstores.base import JobLookupError

        try:
            self._scheduler.remove_job(str(job_id))
        except JobLookupError:
            pass

    def _add_job(self, job: CronJob) -> None:
        trigger_kwargs: dict[str, Any] = {}

        match job.schedule_kind:
            case ScheduleKind.CRON:
                from apscheduler.triggers.cron import CronTrigger

                trigger = CronTrigger.from_crontab(job.schedule_expr)
            case ScheduleKind.EVERY:
                from apscheduler.triggers.interval import IntervalTrigger

                trigger = IntervalTrigger(**_parse_interval(job.schedule_expr))
            case ScheduleKind.AT:
                from apscheduler.triggers.date import DateTrigger
                from datetime import datetime

                trigger = DateTrigger(
                    run_date=datetime.fromisoformat(job.schedule_expr)
                )

        self._scheduler.add_job(
            self._run_job,
            trigger=trigger,
            id=str(job.id),
            args=[job],
            replace_existing=True,
            **trigger_kwargs,
        )

    async def _run_job(self, job: CronJob) -> None:
        if self._runner is None:
            logger.error("No job runner set")
            return
        try:
            await self._runner(job)
        except Exception:
            logger.exception("Cron job %s failed", job.name)

    def start(self) -> None:
        self._scheduler.start()
        logger.info("Cron scheduler started")

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        logger.info("Cron scheduler stopped")


def _parse_interval(expr: str) -> dict[str, int]:
    """Parse interval expressions like '30m', '2h', '1d'."""
    expr = expr.strip().lower()
    if expr.endswith("m"):
        return {"minutes": int(expr[:-1])}
    elif expr.endswith("h"):
        return {"hours": int(expr[:-1])}
    elif expr.endswith("d"):
        return {"days": int(expr[:-1])}
    elif expr.endswith("s"):
        return {"seconds": int(expr[:-1])}
    else:
        raise ValueError(f"Unknown interval format: {expr}")
