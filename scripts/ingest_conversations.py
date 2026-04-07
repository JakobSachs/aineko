#!/usr/bin/env python3
"""Ingest past conversations from the aineko DB into the ChromaDB memory store.

Usage:
    python scripts/ingest_conversations.py                  # ingest all sessions
    python scripts/ingest_conversations.py --dry-run        # show what would be stored
    python scripts/ingest_conversations.py --session 3      # ingest one session
    python scripts/ingest_conversations.py --db sqlite+aiosqlite:///data/aineko.db

Pairs user+assistant messages into exchange chunks, skips system/tool messages,
and stores them tagged with session ID and type=conversation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

# Add project root to path so we can import aineko
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aineko.memory.chunker import chunk_text
from aineko.memory.store import MemoryStore
from aineko.models.message import Message, Role, Session


def clean_content(raw: str) -> str:
    """Extract human-readable text from a message, stripping tool-call JSON.

    Assistant messages are stored as either plain text or a JSON array of
    content blocks like [{"type": "text", "text": "..."}, {"type": "tool_use", ...}].
    We keep only the text blocks.
    """
    if not raw:
        return ""

    # Try parsing as JSON content blocks
    try:
        blocks = json.loads(raw)
        if isinstance(blocks, list):
            parts = []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"].strip())
            return "\n".join(p for p in parts if p)
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    # Fall back: strip any inline JSON objects that look like tool calls
    cleaned = re.sub(
        r'\{"type":\s*"tool_use".*?\}\]',
        "",
        raw,
        flags=re.DOTALL,
    )
    # Also strip the <|tool_calls_section_begin|> markers
    cleaned = re.sub(r"<\|tool_call[^>]*\|>", "", cleaned)
    return cleaned.strip()


def extract_exchanges(messages: list[Message]) -> list[str]:
    """Pair user+assistant messages into exchange strings.

    Skips system and tool messages.  Strips tool-call JSON from assistant
    content, keeping only human-readable text.  Each exchange is:
        > user message
        assistant response
    """
    exchanges: list[str] = []
    pending_user: str | None = None

    for msg in messages:
        if msg.role == Role.USER:
            pending_user = msg.content
        elif msg.role == Role.ASSISTANT and pending_user is not None:
            text = clean_content(msg.content)
            if text:
                exchanges.append(f"> {pending_user}\n{text}")
            pending_user = None

    return exchanges


async def ingest(
    db_url: str,
    chromadb_dir: str,
    session_id: int | None = None,
    dry_run: bool = False,
) -> None:
    engine = create_async_engine(db_url, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    store = None if dry_run else MemoryStore(persist_dir=chromadb_dir)

    async with factory() as db:
        query = select(Session)
        if session_id is not None:
            query = query.where(Session.id == session_id)
        sessions = (await db.execute(query)).scalars().all()

        total_chunks = 0
        for sess in sessions:
            msgs = (
                await db.execute(
                    select(Message)
                    .where(Message.session_id == sess.id)
                    .order_by(Message.created_at)
                )
            ).scalars().all()

            exchanges = extract_exchanges(msgs)
            if not exchanges:
                continue

            text = "\n\n---\n\n".join(exchanges)
            chunks = chunk_text(text)

            if dry_run:
                print(f"session {sess.id} (room {sess.room_id}): {len(exchanges)} exchanges → {len(chunks)} chunks")
                for i, chunk in enumerate(chunks):
                    preview = chunk[:120].replace("\n", "\\n")
                    print(f"  [{i}] {preview}...")
            else:
                tags = {"type": "conversation", "session": str(sess.id), "room": sess.room_id}
                ids = store.add(text, source=f"conversation:{sess.id}", tags=tags)
                total_chunks += len(ids)
                print(f"session {sess.id}: {len(ids)} chunks stored")

    await engine.dispose()

    if not dry_run:
        print(f"\ndone — {total_chunks} total chunks ingested")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest conversations into memory store")
    parser.add_argument("--db", default="sqlite+aiosqlite:///data/aineko.db", help="database URL")
    parser.add_argument("--chromadb-dir", default="/data/memory/chromadb", help="ChromaDB persist directory")
    parser.add_argument("--session", type=int, default=None, help="ingest only this session ID")
    parser.add_argument("--dry-run", action="store_true", help="show what would be stored without storing")
    args = parser.parse_args()

    asyncio.run(ingest(args.db, args.chromadb_dir, args.session, args.dry_run))


if __name__ == "__main__":
    main()
