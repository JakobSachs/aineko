"""initial postgres schema

Revision ID: 0001
Revises:
Create Date: 2026-04-17

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("room_id", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_sessions_room_id", "sessions", ["room_id"], unique=True)

    op.create_table(
        "cron_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column(
            "schedule_kind",
            sa.Enum("CRON", "EVERY", "AT", name="schedulekind"),
            nullable=False,
        ),
        sa.Column("schedule_expr", sa.String(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("delivery_room", sa.String(), nullable=False),
        sa.Column("delete_after_run", sa.Boolean(), nullable=False),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "cron_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "job_id",
            sa.Integer(),
            sa.ForeignKey("cron_jobs.id"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum("SUCCESS", "FAILURE", "RUNNING", name="runstatus"),
            nullable=False,
        ),
        sa.Column("output", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_cron_runs_job_id", "cron_runs", ["job_id"])

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("sessions.id"),
            nullable=False,
        ),
        sa.Column(
            "role",
            sa.Enum("USER", "ASSISTANT", "SYSTEM", "TOOL", name="role"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_call_id", sa.String(), nullable=True),
        sa.Column("tool_name", sa.String(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])

    op.create_table(
        "tool_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("sessions.id"),
            nullable=False,
        ),
        sa.Column(
            "message_id",
            sa.Integer(),
            sa.ForeignKey("messages.id"),
            nullable=False,
        ),
        sa.Column("tool_name", sa.String(), nullable=False),
        sa.Column("arguments", sa.Text(), nullable=False),
        sa.Column("result", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_tool_logs_session_id", "tool_logs", ["session_id"])
    op.create_index("ix_tool_logs_message_id", "tool_logs", ["message_id"])
    op.create_index("ix_tool_logs_tool_name", "tool_logs", ["tool_name"])

    op.create_table(
        "rss_seen_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("feed_url", sa.String(), nullable=False),
        sa.Column("guid", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False, server_default=""),
        sa.Column(
            "seen_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("feed_url", "guid"),
    )
    op.create_index("ix_rss_seen_items_feed_url", "rss_seen_items", ["feed_url"])

    op.create_table(
        "facts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("predicate", sa.String(), nullable=False),
        sa.Column("object", sa.String(), nullable=False),
        sa.Column("valid_from", sa.String(), nullable=True),
        sa.Column("valid_to", sa.String(), nullable=True),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_facts_subject", "facts", ["subject"])
    op.create_index("ix_facts_object", "facts", ["object"])


def downgrade() -> None:
    op.drop_table("facts")
    op.drop_table("rss_seen_items")
    op.drop_table("tool_logs")
    op.drop_table("messages")
    op.drop_table("cron_runs")
    op.drop_table("cron_jobs")
    op.drop_table("sessions")
    sa.Enum(name="role").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="runstatus").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="schedulekind").drop(op.get_bind(), checkfirst=True)
