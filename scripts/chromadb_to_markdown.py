#!/usr/bin/env python3
"""One-shot migration: dump ChromaDB chunks + facts → daily markdown logs.

Usage:
    python scripts/chromadb_to_markdown.py \
        --chromadb-dir /data/memory/chromadb \
        --db postgresql+asyncpg://aineko:aineko@localhost:5432/aineko \
        --out /data/memory \
        [--dry-run] [--reformat]

Produces ``memory/YYYY-MM-DD.md`` files keyed by the SQLite insertion date
of each chunk (from the internal chroma.sqlite3). Groups chunks by date, then
by source within that date. When ``--reformat`` is passed, each (date, source)
group is sent to Kimi for cleanup; otherwise raw chunks are written verbatim.

This script runs once, manually. It is NOT part of the runtime.
Leaves ChromaDB on disk untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))  # noqa: E402

import chromadb  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from aineko.memory.kg import Fact  # noqa: E402


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — populate os.environ for any key not already set."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(_PROJECT_ROOT / ".env")

REFORMAT_PROMPT = """\
Reformat the following raw memory fragments into a clean, human-readable
markdown section. Preserve every fact. Merge duplicates. Group related
points under short subheadings. Do NOT invent information. Output markdown
only — no preamble.

Raw content:

{raw}
"""


def _read_chroma_with_dates(
    chromadb_dir: Path,
) -> dict[date, dict[str, list[str]]]:
    """Return {date: {source: [doc, ...]}} using SQLite insertion timestamps.

    Falls back to today() if the sqlite file or created_at column is missing.
    """
    sqlite_path = chromadb_dir / "chroma.sqlite3"
    id_to_date: dict[str, date] = {}

    if sqlite_path.exists():
        con = sqlite3.connect(sqlite_path)
        try:
            rows = con.execute(
                "SELECT embedding_id, created_at FROM embeddings"
            ).fetchall()
            for eid, created_at in rows:
                if created_at:
                    try:
                        d = datetime.fromisoformat(created_at).date()
                    except ValueError:
                        d = date.today()
                else:
                    d = date.today()
                id_to_date[eid] = d
        finally:
            con.close()

    client = chromadb.PersistentClient(path=str(chromadb_dir))
    try:
        col = client.get_collection("memories")
    except Exception:
        print(f"no 'memories' collection at {chromadb_dir}", file=sys.stderr)
        return {}

    data = col.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    # {date: {source: [doc]}}
    grouped: dict[date, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for eid, doc, meta in zip(ids, docs, metas):
        d = id_to_date.get(eid, date.today())
        src = (meta or {}).get("source", "unknown")
        if doc:
            grouped[d][src].append(doc)

    return grouped


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
    grouped = _read_chroma_with_dates(chromadb_dir)

    api_key = os.environ.get("KIMI_API_KEY", "")
    base_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    model = os.environ.get("KIMI_MODEL", "kimi-k2.5")
    if reformat and not api_key:
        print("--reformat requires KIMI_API_KEY in env", file=sys.stderr)
        sys.exit(2)

    total_groups = sum(len(srcs) for srcs in grouped.values())
    print(
        f"found {len(grouped)} date(s), {total_groups} (date, source) group(s)",
        file=sys.stderr,
    )

    for d in sorted(grouped.keys()):
        for src, docs in grouped[d].items():
            raw = "\n\n---\n\n".join(docs)
            if not raw.strip():
                continue
            body = await _reformat(raw, api_key, base_url, model) if reformat else raw
            header = f"Memory (migrated from chromadb: {src})"
            if dry_run:
                print(
                    f"[chroma] {d} :: {src}: {len(docs)} chunk(s), {len(body)} chars"
                )
            else:
                path = _write_entry(out_dir, d, header, body)
                print(f"wrote {path} ← {d} source={src} ({len(docs)} chunk(s))")

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
