"""Cron job schemas."""

from datetime import datetime

from pydantic import BaseModel

from aineko.models.cron import RunStatus, ScheduleKind


class CronJobCreate(BaseModel):
    name: str
    schedule_kind: ScheduleKind
    schedule_expr: str
    message: str
    delivery_room: str
    delete_after_run: bool = False


class CronJobSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    enabled: bool
    schedule_kind: ScheduleKind
    schedule_expr: str
    message: str
    delivery_room: str
    delete_after_run: bool
    consecutive_failures: int
    created_at: datetime


class CronRunSchema(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    job_id: int
    status: RunStatus
    output: str | None
    error: str | None
    started_at: datetime
    finished_at: datetime | None
