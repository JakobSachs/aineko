"""Memory tool — file-based daily logs + temporal knowledge graph.

Durable memories are stored as append-only markdown in
``<memory_dir>/YYYY-MM-DD.md``. Structured facts live in the Postgres
knowledge-graph table.
"""

import logging
from pathlib import Path

from aineko.db import get_session
from aineko.memory.daily_log import append_to_today, read_log, search_logs
from aineko.memory.kg import KnowledgeGraph
from aineko.tools.registry import ToolDef

logger = logging.getLogger(__name__)

# Default path used when app.py doesn't call init_memory_dir() (e.g. in tests
# that patch the module-level var directly).
_memory_dir: Path = Path("/data/memory")
_kg = KnowledgeGraph()


def init_memory_dir(path: Path) -> None:
    """Configure where the memory tool reads/writes daily logs."""
    global _memory_dir
    _memory_dir = path


async def _memory(
    action: str,
    query: str = "",
    content: str = "",
    source: str = "",
    tags: str = "",
    entity: str = "",
    subject: str = "",
    predicate: str = "",
    object: str = "",
    valid_from: str = "",
    as_of: str = "",
    path: str = "",
    from_line: int = 1,
    lines: int = 0,
    n_results: int = 20,
) -> str:
    if action == "search":
        if not query:
            return "Error: 'query' is required for search."
        results = search_logs(_memory_dir, query, limit=n_results)
        if not results:
            return f"No memories matching '{query}'."
        out = []
        for r in results:
            out.append(f"{r.path}:{r.line}: {r.text}")
        return "\n".join(out)

    elif action == "store":
        if not content:
            return "Error: 'content' is required for store."
        prefix = ""
        if source:
            prefix = f"**source:** {source}  \n"
        if tags:
            prefix += f"**tags:** {tags}  \n"
        body = (prefix + "\n" + content.strip()) if prefix else content.strip()
        file_path = append_to_today(_memory_dir, body)
        return f"Appended to {file_path}."

    elif action == "read":
        if not path:
            return "Error: 'path' is required for read."
        target = Path(path)
        if not target.is_absolute():
            target = _memory_dir / target
        lim = lines if lines > 0 else None
        text = read_log(target, from_line=from_line, lines=lim)
        if not text:
            return f"No content at {target} (from_line={from_line})."
        return f"{target} (from line {from_line}):\n{text}"

    elif action == "facts_query":
        if not entity:
            return "Error: 'entity' is required for facts_query."
        async for db in get_session():
            results = await _kg.query(db, entity, as_of=as_of or None)
        if not results:
            return f"No facts about '{entity}'."
        out = []
        for f in results:
            line = f"{f['subject']} —{f['predicate']}→ {f['object']}"
            if f["valid_from"]:
                line += f" (from {f['valid_from']}"
                line += f" to {f['valid_to']})" if f["valid_to"] else ")"
            out.append(line)
        return "\n".join(out)

    elif action == "facts_add":
        if not subject or not predicate or not object:
            return "Error: 'subject', 'predicate', and 'object' are required."
        async for db in get_session():
            fact_id = await _kg.add(
                db,
                subject,
                predicate,
                object,
                valid_from=valid_from or None,
                source=source or None,
            )
            await db.commit()
        return f"Fact #{fact_id}: {subject} —{predicate}→ {object}"

    elif action == "facts_invalidate":
        if not subject or not predicate or not object:
            return "Error: 'subject', 'predicate', and 'object' are required."
        async for db in get_session():
            found = await _kg.invalidate(
                db,
                subject,
                predicate,
                object,
                ended=valid_from or None,
            )
            await db.commit()
        if found:
            return f"Invalidated: {subject} —{predicate}→ {object}"
        return "No matching active fact found."

    elif action == "facts_timeline":
        if not entity:
            return "Error: 'entity' is required for facts_timeline."
        async for db in get_session():
            results = await _kg.timeline(db, entity)
        if not results:
            return f"No facts about '{entity}'."
        out = []
        for f in results:
            date = f["valid_from"] or "?"
            end = f" → {f['valid_to']}" if f["valid_to"] else ""
            out.append(f"[{date}{end}] {f['subject']} —{f['predicate']}→ {f['object']}")
        return "\n".join(out)

    else:
        actions = (
            "search, store, read, facts_query, facts_add, facts_invalidate, "
            "facts_timeline"
        )
        return f"Unknown action '{action}'. Use: {actions}"


memory_tool = ToolDef(
    name="memory",
    description=(
        "Mid/long-term memory — substring search across daily markdown logs, "
        "append-only writes, and a temporal knowledge graph.\n\n"
        "Call search at the START of every conversation, using keywords from "
        "what the user is asking about. A missed search costs trust; a wasted "
        "search costs nothing.\n\n"
        "After any non-trivial conversation: store what you learned — "
        "preferences, decisions, corrections, people, project context. "
        "Writes append to memory/YYYY-MM-DD.md (never overwrite). "
        "Use facts_add for structured entity facts (person X does Y), "
        "facts_invalidate when facts change."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "search",
                    "store",
                    "read",
                    "facts_query",
                    "facts_add",
                    "facts_invalidate",
                    "facts_timeline",
                ],
                "description": (
                    "search: substring search over daily memory logs. "
                    "store: append content to today's memory log. "
                    "read: targeted read of a memory log by path + line range. "
                    "facts_query: look up what you know about an entity. "
                    "facts_add: record a fact (subject-predicate-object). "
                    "facts_invalidate: mark a fact as no longer true. "
                    "facts_timeline: chronological history of an entity."
                ),
            },
            "query": {
                "type": "string",
                "description": "Substring to search for (for search).",
            },
            "content": {
                "type": "string",
                "description": "Text to append to today's log (for store).",
            },
            "source": {
                "type": "string",
                "description": "Optional label for the memory source, e.g. 'user', 'conversation'.",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated key=value metadata, recorded in the entry.",
            },
            "path": {
                "type": "string",
                "description": "Log file path for read action (absolute, or relative to memory dir).",
            },
            "from_line": {
                "type": "integer",
                "description": "1-indexed line to start reading from (for read). Default 1.",
            },
            "lines": {
                "type": "integer",
                "description": "Max lines to return (for read). 0 means read to end.",
            },
            "entity": {
                "type": "string",
                "description": "Entity name to query (for facts_query, facts_timeline).",
            },
            "subject": {
                "type": "string",
                "description": "Subject entity (for facts_add, facts_invalidate).",
            },
            "predicate": {
                "type": "string",
                "description": "Relationship type, e.g. 'likes', 'lives_in', 'works_at'.",
            },
            "object": {
                "type": "string",
                "description": "Object entity (for facts_add, facts_invalidate).",
            },
            "valid_from": {
                "type": "string",
                "description": "ISO date (YYYY-MM-DD) when a fact became true, or end date for invalidation.",
            },
            "as_of": {
                "type": "string",
                "description": "ISO date to query facts as of a specific point in time.",
            },
            "n_results": {
                "type": "integer",
                "description": "Max results for search (default 20).",
            },
        },
        "required": ["action"],
    },
    handler=_memory,
)
