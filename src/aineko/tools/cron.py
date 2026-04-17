"""Tools for the agent to manage its own cron jobs."""

from sqlalchemy import select

from aineko.cron.scheduler import CronScheduler
from aineko.db import get_session
from aineko.models.cron import CronJob, CronRun, ScheduleKind
from aineko.tools.registry import ToolDef

_SCHEDULE_KINDS = tuple(k.value for k in ScheduleKind)


def make_cron_tools(scheduler: CronScheduler, default_room: str) -> list[ToolDef]:
    """Build cron management tools bound to a scheduler and default delivery room."""

    async def _list_cron_jobs() -> str:
        async for db in get_session():
            result = await db.execute(select(CronJob).order_by(CronJob.id))
            jobs = list(result.scalars().all())

        if not jobs:
            return "No cron jobs defined."

        lines = []
        for j in jobs:
            state = "enabled" if j.enabled else "disabled"
            lines.append(
                f"#{j.id} [{state}] {j.name}: {j.schedule_kind.value}={j.schedule_expr} "
                f"-> {j.delivery_room} "
                f"(fails={j.consecutive_failures}, one_shot={j.delete_after_run})\n"
                f"  message: {j.message[:200]}"
            )
        return "\n".join(lines)

    async def _create_cron_job(
        name: str,
        schedule_kind: str,
        schedule_expr: str,
        message: str,
        delivery_room: str = "",
        delete_after_run: bool = False,
    ) -> str:
        if schedule_kind not in _SCHEDULE_KINDS:
            return f"Error: schedule_kind must be one of {_SCHEDULE_KINDS}"
        room = delivery_room or default_room
        if not room:
            return "Error: no delivery_room provided and no default available"

        async for db in get_session():
            existing = await db.execute(select(CronJob).where(CronJob.name == name))
            if existing.scalar_one_or_none() is not None:
                return f"Error: a cron job named '{name}' already exists"

            job = CronJob(
                name=name,
                enabled=True,
                schedule_kind=ScheduleKind(schedule_kind),
                schedule_expr=schedule_expr,
                message=message,
                delivery_room=room,
                delete_after_run=delete_after_run,
            )
            db.add(job)
            await db.commit()
            await db.refresh(job)
            job_snapshot = job

        try:
            scheduler.add_job(job_snapshot)
        except Exception as e:
            return (
                f"Created job #{job_snapshot.id} in DB but failed to schedule it: {e}. "
                f"Fix the schedule expression and call update_cron_job."
            )

        return (
            f"Created cron job #{job_snapshot.id} '{name}' "
            f"({schedule_kind}={schedule_expr}) -> {room}"
        )

    async def _update_cron_job(
        job_id: int,
        enabled: bool | None = None,
        schedule_kind: str | None = None,
        schedule_expr: str | None = None,
        message: str | None = None,
        delivery_room: str | None = None,
        delete_after_run: bool | None = None,
    ) -> str:
        if schedule_kind is not None and schedule_kind not in _SCHEDULE_KINDS:
            return f"Error: schedule_kind must be one of {_SCHEDULE_KINDS}"

        async for db in get_session():
            result = await db.execute(select(CronJob).where(CronJob.id == job_id))
            job = result.scalar_one_or_none()
            if job is None:
                return f"Error: no cron job with id {job_id}"

            if enabled is not None:
                job.enabled = enabled
            if schedule_kind is not None:
                job.schedule_kind = ScheduleKind(schedule_kind)
            if schedule_expr is not None:
                job.schedule_expr = schedule_expr
            if message is not None:
                job.message = message
            if delivery_room is not None:
                job.delivery_room = delivery_room
            if delete_after_run is not None:
                job.delete_after_run = delete_after_run

            await db.commit()
            await db.refresh(job)
            job_snapshot = job

        scheduler.remove_job(job_id)
        if job_snapshot.enabled:
            try:
                scheduler.add_job(job_snapshot)
            except Exception as e:
                return f"Updated job #{job_id} in DB but rescheduling failed: {e}"

        return f"Updated cron job #{job_id} '{job_snapshot.name}'"

    async def _delete_cron_job(job_id: int) -> str:
        async for db in get_session():
            result = await db.execute(select(CronJob).where(CronJob.id == job_id))
            job = result.scalar_one_or_none()
            if job is None:
                return f"Error: no cron job with id {job_id}"
            name = job.name
            await db.delete(job)
            await db.commit()

        scheduler.remove_job(job_id)
        return f"Deleted cron job #{job_id} '{name}'"

    async def _get_cron_job_runs(job_id: int, limit: int = 10) -> str:
        limit = min(max(limit, 1), 50)
        async for db in get_session():
            result = await db.execute(
                select(CronRun)
                .where(CronRun.job_id == job_id)
                .order_by(CronRun.started_at.desc())
                .limit(limit)
            )
            runs = list(result.scalars().all())

        if not runs:
            return f"No runs recorded for job #{job_id}."

        lines = []
        for r in runs:
            ts = r.started_at.strftime("%Y-%m-%d %H:%M")
            summary = (r.output or r.error or "").strip().replace("\n", " ")
            lines.append(f"[{ts}] {r.status.value}: {summary[:200]}")
        return "\n".join(lines)

    list_tool = ToolDef(
        name="list_cron_jobs",
        description="List all cron jobs (enabled and disabled) with their schedule and status.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_list_cron_jobs,
    )

    create_tool = ToolDef(
        name="create_cron_job",
        description=(
            "Create a scheduled job that will run the given message as a fresh agent "
            "session and deliver the reply to a Matrix room. "
            "schedule_kind='cron' uses a 5-field crontab expr (e.g. '0 9 * * *'). "
            "'every' uses an interval like '30m', '2h', '1d', '45s'. "
            "'at' uses an ISO datetime for a one-shot run (set delete_after_run=true). "
            "If delivery_room is omitted the current room is used."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique short name for the job.",
                },
                "schedule_kind": {
                    "type": "string",
                    "enum": list(_SCHEDULE_KINDS),
                    "description": "'cron', 'every', or 'at'.",
                },
                "schedule_expr": {
                    "type": "string",
                    "description": "Schedule expression matching schedule_kind.",
                },
                "message": {
                    "type": "string",
                    "description": "The prompt to run as a fresh agent session when the job fires.",
                },
                "delivery_room": {
                    "type": "string",
                    "description": "Matrix room id for output. Defaults to the current room.",
                },
                "delete_after_run": {
                    "type": "boolean",
                    "description": "If true, the job is deleted after a successful run (for one-shots).",
                },
            },
            "required": ["name", "schedule_kind", "schedule_expr", "message"],
        },
        handler=_create_cron_job,
    )

    update_tool = ToolDef(
        name="update_cron_job",
        description=(
            "Modify an existing cron job by id. Only provided fields are changed. "
            "Disabling a job unschedules it; enabling reschedules it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "job_id": {"type": "integer"},
                "enabled": {"type": "boolean"},
                "schedule_kind": {"type": "string", "enum": list(_SCHEDULE_KINDS)},
                "schedule_expr": {"type": "string"},
                "message": {"type": "string"},
                "delivery_room": {"type": "string"},
                "delete_after_run": {"type": "boolean"},
            },
            "required": ["job_id"],
        },
        handler=_update_cron_job,
    )

    delete_tool = ToolDef(
        name="delete_cron_job",
        description="Delete a cron job by id. Removes it from the schedule and the database.",
        parameters={
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
        handler=_delete_cron_job,
    )

    runs_tool = ToolDef(
        name="get_cron_job_runs",
        description="Show recent run history for a cron job (most recent first).",
        parameters={
            "type": "object",
            "properties": {
                "job_id": {"type": "integer"},
                "limit": {
                    "type": "integer",
                    "description": "Max runs to return (default 10, max 50).",
                },
            },
            "required": ["job_id"],
        },
        handler=_get_cron_job_runs,
    )

    return [list_tool, create_tool, update_tool, delete_tool, runs_tool]
