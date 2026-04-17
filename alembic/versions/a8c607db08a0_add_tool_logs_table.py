"""add tool_logs table

Revision ID: a8c607db08a0
Revises: b485554b83a7
Create Date: 2026-04-17 10:04:27.085958

NOTE: In environments that were running before this migration was added,
the tool_logs table already exists because db.create_tables() created it
on startup. Run `alembic stamp a8c607db08a0` on those databases to mark
this migration applied without executing it.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a8c607db08a0'
down_revision: Union[str, Sequence[str], None] = 'b485554b83a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'tool_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('message_id', sa.Integer(), nullable=False),
        sa.Column('tool_name', sa.String(), nullable=False),
        sa.Column('arguments', sa.Text(), nullable=False),
        sa.Column('result', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('(CURRENT_TIMESTAMP)'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['message_id'], ['messages.id']),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('tool_logs', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_tool_logs_message_id'), ['message_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_tool_logs_session_id'), ['session_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_tool_logs_tool_name'), ['tool_name'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('tool_logs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_tool_logs_tool_name'))
        batch_op.drop_index(batch_op.f('ix_tool_logs_session_id'))
        batch_op.drop_index(batch_op.f('ix_tool_logs_message_id'))
    op.drop_table('tool_logs')
