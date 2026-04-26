"""add messages.blocks JSONB and backfill from sentinel rows

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-26

Persists API-shape content blocks (text/thinking/tool_use/tool_result) so
thinking blocks survive replay. Sentinel `tool_name` values
(`__has_tool_calls__`, `__tool_results__`) are retired in favor of
`blocks IS NOT NULL` plus role.

Backfill is validated row-by-row before any sentinel is cleared. If any
row fails decoding or validation, the migration aborts; the partial
`blocks` updates roll back with the alembic transaction. `content` is
left in place for search_chat / tool_history readers.
"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_HAS_TOOL_CALLS = "__has_tool_calls__"
_TOOL_RESULTS = "__tool_results__"


def _validate_assistant_blocks(blocks: list) -> str | None:
    if not isinstance(blocks, list) or not blocks:
        return "blocks is not a non-empty list"
    for b in blocks:
        if not isinstance(b, dict) or "type" not in b:
            return f"block missing type field: {b!r}"
        t = b["type"]
        if t not in ("text", "thinking", "tool_use", "redacted_thinking"):
            return f"unexpected block type for assistant: {t!r}"
    return None


def _validate_tool_result_blocks(blocks: list) -> str | None:
    if not isinstance(blocks, list) or not blocks:
        return "blocks is not a non-empty list"
    for b in blocks:
        if not isinstance(b, dict) or b.get("type") != "tool_result":
            return f"non-tool_result block in tool_results row: {b!r}"
        if "tool_use_id" not in b:
            return "tool_result missing tool_use_id"
    return None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("blocks", JSONB(), nullable=True),
    )

    bind = op.get_bind()

    rows = bind.execute(
        sa.text(
            "SELECT id, role, tool_name, content FROM messages "
            "WHERE tool_name IN (:a, :b) ORDER BY id"
        ),
        {"a": _HAS_TOOL_CALLS, "b": _TOOL_RESULTS},
    ).fetchall()

    failures: list[str] = []
    decoded: list[tuple[int, list]] = []

    for row in rows:
        msg_id, role, tool_name, content = row
        try:
            blocks = json.loads(content) if content else None
        except json.JSONDecodeError as e:
            failures.append(f"id={msg_id} JSON decode error: {e}")
            continue

        if tool_name == _HAS_TOOL_CALLS:
            if role != "ASSISTANT":
                failures.append(
                    f"id={msg_id} {_HAS_TOOL_CALLS} on non-assistant role={role!r}"
                )
                continue
            err = _validate_assistant_blocks(blocks)
        else:
            if role != "TOOL":
                failures.append(
                    f"id={msg_id} {_TOOL_RESULTS} on non-tool role={role!r}"
                )
                continue
            err = _validate_tool_result_blocks(blocks)

        if err:
            failures.append(f"id={msg_id} validation: {err}")
            continue
        decoded.append((msg_id, blocks))

    if failures:
        sample = "\n  ".join(failures[:20])
        raise RuntimeError(
            f"backfill aborted: {len(failures)} row(s) failed validation. "
            f"First {min(20, len(failures))}:\n  {sample}"
        )

    for msg_id, blocks in decoded:
        bind.execute(
            sa.text(
                "UPDATE messages SET blocks = CAST(:blocks AS JSONB), "
                "tool_name = NULL WHERE id = :id"
            ),
            {"blocks": json.dumps(blocks), "id": msg_id},
        )

    remaining = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM messages WHERE tool_name IN (:a, :b)"
        ),
        {"a": _HAS_TOOL_CALLS, "b": _TOOL_RESULTS},
    ).scalar_one()
    if remaining:
        raise RuntimeError(
            f"backfill incomplete: {remaining} sentinel rows remain after backfill"
        )


def downgrade() -> None:
    bind = op.get_bind()

    rows = bind.execute(
        sa.text(
            "SELECT id, role, blocks FROM messages "
            "WHERE blocks IS NOT NULL ORDER BY id"
        ),
    ).fetchall()

    for row in rows:
        msg_id, role, blocks = row
        sentinel = _HAS_TOOL_CALLS if role == "ASSISTANT" else _TOOL_RESULTS
        bind.execute(
            sa.text(
                "UPDATE messages SET tool_name = :tn, content = :c "
                "WHERE id = :id"
            ),
            {"tn": sentinel, "c": json.dumps(blocks), "id": msg_id},
        )

    op.drop_column("messages", "blocks")
