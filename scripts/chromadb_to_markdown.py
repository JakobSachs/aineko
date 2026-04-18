#!/usr/bin/env python3
"""One-shot migration: dump ChromaDB chunks + facts → daily markdown logs.

Usage:
    python scripts/chromadb_to_markdown.py \
        --chromadb-dir /data/memory/chromadb \
        --db postgresql+asyncpg://aineko:aineko@localhost:5432/aineko \
        --out /data/memory \
        [--dry-run] [--reformat]

Produces ``memory/YYYY-MM-DD.md`` files keyed by each group's earliest
ingestion timestamp (fallback: today). Groups chunks by ``source`` in the
ChromaDB metadata. When ``--reformat`` is passed, raw content is sent to
Kimi for cleanup before being written; otherwise the raw chunks are
dumped verbatim.

This script runs once, manually. It is NOT part of the runtime.
Leaves ChromaDB on disk untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chromadb
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aineko.memory.kg import Fact


REFORMAT_PROMPT = """\
Reformat the following raw memory fragments into a clean, human-readable
markdown section. Preserve every fact. Merge duplicates. Group related
points under short subheadings. Do NOT invent information. Output markdown
only — no preamble.

Raw content:

{raw}
"""


def _group_chroma_by_source(
    chromadb_dir: Path,
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    client = chromadb.PersistentClient(path=str(chromadb_dir))
    try:
        col = client.get_collection("memories")
    except Exception:
        print(f"no 'memories' collection at {chromadb_dir}", file=sys.stderr)
        return {}

    data = col.get(include=["documents", "metadatas"])
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for doc, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
        src = (meta or {}).get("source", "unknown")
        groups[src].append((doc, meta or {}))
    return groups


def _date_for_source(src: str, entries: list[tuple[str, dict[str, Any]]]) -> date:
    """Pick a date for the daily log file. Tries tags, then today."""
    for _, meta in entries:
        for key in ("date", "created_at", "timestamp"):
            raw = meta.get(key)
            if not raw:
                continue
            try:
                return datetime.fromisoformat(str(raw)).date()
            except ValueError:
                continue
    return date.today()


async def _reformat(raw: str, api_key: str, base_url: str, model: str) -> str:
    import httpx

    async with httpx.AsyncClient(
        base_url=base_url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        timeout=120,
    ) as http:
        resp = await http.post(
            "/messages",
            json={
                "model": model,
                "max_tokens": 8000,
                "messages": [
                    {"role": "user", "content": REFORMAT_PROMPT.format(raw=raw)}
                ],
            },
        )
        resp.raise_for_status()
        body = resp.json()
        for block in body.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "").strip()
    return raw


def _write_entry(out_dir: Path, d: date, header: str, body: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{d.isoformat()}.md"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {header}\n\n{body.strip()}\n")
    return path


async def _dump_facts(db_url: str, out_dir: Path, dry_run: bool) -> int:
    engine = create_async_engine(db_url, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    count = 0
    async with factory() as db:
        rows = (await db.execute(select(Fact).order_by(Fact.created_at))).scalars().all()
        grouped: dict[date, list[Fact]] = defaultdict(list)
        for f in rows:
            d = (f.created_at or datetime.now()).date()
            grouped[d].append(f)
        for d, facts in sorted(grouped.items()):
            lines: list[str] = []
            for f in facts:
                line = f"- {f.subject} —{f.predicate}→ {f.object_}"
                if f.valid_from:
                    line += f" (from {f.valid_from}"
                    line += f" to {f.valid_to})" if f.valid_to else ")"
                lines.append(line)
            body = "\n".join(lines)
            if dry_run:
                print(f"[facts] {d}: {len(facts)} fact(s)")
            else:
                _write_entry(out_dir, d, "Facts (migrated)", body)
            count += len(facts)
    await engine.dispose()
    return count


async def migrate(
    chromadb_dir: Path,
    db_url: str,
    out_dir: Path,
    dry_run: bool,
    reformat: bool,
) -> None:
    groups = _group_chroma_by_source(chromadb_dir)
    print(f"found {len(groups)} chromadb source group(s)", file=sys.stderr)

    api_key = os.environ.get("KIMI_API_KEY", "")
    base_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    model = os.environ.get("KIMI_MODEL", "kimi-k2.5")
    if reformat and not api_key:
        print("--reformat requires KIMI_API_KEY in env", file=sys.stderr)
        sys.exit(2)

    for src, entries in groups.items():
        d = _date_for_source(src, entries)
        raw = "\n\n---\n\n".join(doc for doc, _ in entries if doc)
        if not raw.strip():
            continue
        body = await _reformat(raw, api_key, base_url, model) if reformat else raw
        header = f"Memory (migrated from chromadb: {src})"
        if dry_run:
            print(f"[chroma] {d} :: {src}: {len(entries)} chunk(s), {len(body)} chars")
        else:
            path = _write_entry(out_dir, d, header, body)
            print(f"wrote {path} ← source={src} ({len(entries)} chunk(s))")

    fact_count = await _dump_facts(db_url, out_dir, dry_run)
    print(f"facts migrated: {fact_count}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Dump ChromaDB + facts to daily markdown")
    p.add_argument("--chromadb-dir", type=Path, default=Path("/data/memory/chromadb"))
    p.add_argument(
        "--db",
        default="postgresql+asyncpg://aineko:aineko@localhost:5432/aineko",
    )
    p.add_argument("--out", type=Path, default=Path("/data/memory"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--reformat",
        action="store_true",
        help="reformat raw chunks with Kimi before writing",
    )
    args = p.parse_args()

    asyncio.run(
        migrate(args.chromadb_dir, args.db, args.out, args.dry_run, args.reformat)
    )


if __name__ == "__main__":
    main()
