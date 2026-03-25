"""Cron job and run history models."""

import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aineko.db import Base


class ScheduleKind(str, enum.Enum):
    CRON = "cron"
    EVERY = "every"
    AT = "at"


class RunStatus(str, enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class CronJob(Base):
    __tablename__ = "cron_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    schedule_kind: Mapped[ScheduleKind] = mapped_column(Enum(ScheduleKind))
    schedule_expr: Mapped[str] = mapped_column(String)
    message: Mapped[str] = mapped_column(Text)
    delivery_room: Mapped[str] = mapped_column(String)
    delete_after_run: Mapped[bool] = mapped_column(Boolean, default=False)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    next_run: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    runs: Mapped[list["CronRun"]] = relationship(
        back_populates="job", order_by="CronRun.started_at.desc()", cascade="all, delete-orphan"
    )


class CronRun(Base):
    __tablename__ = "cron_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("cron_jobs.id"), index=True)
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus))
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    job: Mapped[CronJob] = relationship(back_populates="runs")
