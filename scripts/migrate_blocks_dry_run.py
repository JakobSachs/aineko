"""Dry-run for the 0003 messages.blocks backfill.

Connects to the live DB, runs the same decode+validate logic the alembic
migration will run, and prints a summary. Writes nothing.

Usage:
    uv run python scripts/migrate_blocks_dry_run.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

HAS_TOOL_CALLS = "__has_tool_calls__"
TOOL_RESULTS = "__tool_results__"


def validate_assistant_blocks(blocks: object) -> str | None:
    if not isinstance(blocks, list) or not blocks:
        return "blocks is not a non-empty list"
    for b in blocks:
        if not isinstance(b, dict) or "type" not in b:
            return f"block missing type field: {b!r}"
        t = b["type"]
        if t not in ("text", "thinking", "tool_use", "redacted_thinking"):
            return f"unexpected block type for assistant: {t!r}"
    return None


def validate_tool_result_blocks(blocks: object) -> str | None:
    if not isinstance(blocks, list) or not blocks:
        return "blocks is not a non-empty list"
    for b in blocks:
        if not isinstance(b, dict) or b.get("type") != "tool_result":
            return f"non-tool_result block in tool_results row: {b!r}"
        if "tool_use_id" not in b:
            return "tool_result missing tool_use_id"
    return None


def _db_url() -> str:
    url = os.environ.get("AINEKO_DATABASE_URL")
    if url:
        return url
    from aineko.config import Settings

    return Settings().db_url


async def _scan() -> tuple[int, Counter[str], Counter[str], list, list, list]:
    url = _db_url()
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(url)

    role_total: Counter[str] = Counter()
    role_with_blocks_target: Counter[str] = Counter()
    decode_failures: list[tuple[int, str]] = []
    validation_failures: list[tuple[int, str]] = []
    samples: list[tuple[int, str, list]] = []

    async with engine.connect() as conn:
        total = (
            await conn.execute(sa.text("SELECT COUNT(*) FROM messages"))
        ).scalar_one()
        result = await conn.execute(
            sa.text("SELECT id, role, tool_name, content FROM messages ORDER BY id")
        )
        for row in result:
            msg_id, role, tool_name, content = row
            role_total[role] += 1

            if tool_name not in (HAS_TOOL_CALLS, TOOL_RESULTS):
                continue

            try:
                blocks = json.loads(content) if content else None
            except json.JSONDecodeError as e:
                decode_failures.append((msg_id, str(e)))
                continue

            if tool_name == HAS_TOOL_CALLS:
                err = validate_assistant_blocks(blocks)
                if role != "ASSISTANT":
                    err = err or f"sentinel on non-assistant role={role!r}"
            else:
                err = validate_tool_result_blocks(blocks)
                if role != "TOOL":
                    err = err or f"sentinel on non-tool role={role!r}"

            if err:
                validation_failures.append((msg_id, err))
                continue

            role_with_blocks_target[role] += 1
            if len(samples) < 5 and isinstance(blocks, list):
                samples.append((msg_id, role, blocks))

    await engine.dispose()
    return total, role_total, role_with_blocks_target, decode_failures, validation_failures, samples


def main() -> int:
    (
        total,
        role_total,
        role_with_blocks_target,
        decode_failures,
        validation_failures,
        samples,
    ) = asyncio.run(_scan())

    print(f"messages total: {total}")
    print("\nrole counts:")
    for r, n in sorted(role_total.items()):
        will = role_with_blocks_target.get(r, 0)
        print(f"  {r:<10} total={n:<6} will_get_blocks={will}")

    print(f"\ndecode failures: {len(decode_failures)}")
    for mid, err in decode_failures[:10]:
        print(f"  id={mid}: {err}")

    print(f"\nvalidation failures: {len(validation_failures)}")
    for mid, err in validation_failures[:10]:
        print(f"  id={mid}: {err}")

    print("\nsample decoded rows:")
    for mid, role, blocks in samples:
        types = [b.get("type") for b in blocks if isinstance(b, dict)]
        print(f"  id={mid} role={role} blocks={len(blocks)} types={types}")

    if decode_failures or validation_failures:
        print("\nFAIL: real migration would abort. Inspect failures above.")
        return 1
    print("\nOK: backfill is safe to run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
