"""Cron job runner — executes a job as an isolated agent session and delivers output."""

import logging
from datetime import datetime, timezone

from sqlalchemy import select

from aineko.db import get_session
from aineko.handler import format_tool_footer
from aineko.kimi.client import KimiClient
from aineko.matrix.client import MatrixConnector
from aineko.models.cron import CronJob, CronRun, RunStatus
from aineko.skills.engine import SkillsEngine
from aineko.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


async def run_cron_job(
    job: CronJob,
    kimi: KimiClient,
    tools: ToolRegistry,
    skills: SkillsEngine,
    matrix: MatrixConnector,
    system_prompt: str,
) -> None:
    """Execute a cron job: run agent in isolated context, record result, deliver to Matrix."""
    logger.info("Running cron job: %s", job.name)

    async for db in get_session():
        run = CronRun(job_id=job.id, status=RunStatus.RUNNING)
        db.add(run)
        await db.flush()

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": job.message},
            ]

            response = await kimi.chat_loop(messages, tools)

            run.status = RunStatus.SUCCESS
            run.output = response.content
            run.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)

            # Reset failure counter on success
            result = await db.execute(select(CronJob).where(CronJob.id == job.id))
            db_job = result.scalar_one()
            db_job.consecutive_failures = 0

            # Delete one-shot jobs
            if job.delete_after_run:
                await db.delete(db_job)

            await db.commit()

            # Deliver to Matrix
            if response.content and job.delivery_room:
                footer = format_tool_footer(response.tool_history)
                await matrix.send_message(job.delivery_room, response.content + footer)

        except Exception as e:
            logger.exception("Cron job %s failed", job.name)
            run.status = RunStatus.FAILURE
            run.error = str(e)
            run.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)

            result = await db.execute(select(CronJob).where(CronJob.id == job.id))
            db_job = result.scalar_one()
            db_job.consecutive_failures += 1

            await db.commit()
