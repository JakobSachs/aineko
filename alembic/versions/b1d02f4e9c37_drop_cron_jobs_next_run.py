"""drop cron_jobs.next_run

Revision ID: b1d02f4e9c37
Revises: a8c607db08a0
Create Date: 2026-04-17 10:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b1d02f4e9c37'
down_revision: Union[str, Sequence[str], None] = 'a8c607db08a0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('cron_jobs', schema=None) as batch_op:
        batch_op.drop_column('next_run')


def downgrade() -> None:
    with op.batch_alter_table('cron_jobs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('next_run', sa.DATETIME(), nullable=True))
