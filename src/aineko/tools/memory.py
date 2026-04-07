"""Memory tool — semantic search and storage + temporal knowledge graph.

Replaces the old file-based memory with ChromaDB embeddings for search
and a SQLite knowledge graph for structured facts.
"""

import logging

from aineko.db import get_session
from aineko.memory.kg import KnowledgeGraph
from aineko.memory.store import MemoryStore
from aineko.tools.registry import ToolDef

logger = logging.getLogger(__name__)

CHROMADB_DIR = "/data/memory/chromadb"

# Module-level singletons — initialized on first use
_store: MemoryStore | None = None
_kg = KnowledgeGraph()


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore(persist_dir=CHROMADB_DIR)
    return _store


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
    n_results: int = 5,
) -> str:
    store = _get_store()

    if action == "search":
        if not query:
            return "Error: 'query' is required for search."
        where = None
        if tags:
            where = dict(pair.split("=") for pair in tags.split(",") if "=" in pair)
        results = store.search(query, n_results=n_results, where=where)
        if not results:
            return f"No memories matching '{query}'."
        lines = []
        for r in results:
            sim = f"{r.similarity:.0%}"
            lines.append(f"[{sim}] (source: {r.source})")
            lines.append(r.content)
            lines.append("")
        return "\n".join(lines)

    elif action == "store":
        if not content:
            return "Error: 'content' is required for store."
        tag_dict = None
        if tags:
            tag_dict = dict(pair.split("=") for pair in tags.split(",") if "=" in pair)
        src = source or "agent"
        ids = store.add(content, source=src, tags=tag_dict)
        if not ids:
            return "Content already exists in memory (duplicate)."
        return f"Stored {len(ids)} chunk(s) under source '{src}'."

    elif action == "facts_query":
        if not entity:
            return "Error: 'entity' is required for facts_query."
        async for db in get_session():
            results = await _kg.query(db, entity, as_of=as_of or None)
        if not results:
            return f"No facts about '{entity}'."
        lines = []
        for f in results:
            line = f"{f['subject']} —{f['predicate']}→ {f['object']}"
            if f["valid_from"]:
                line += f" (from {f['valid_from']}"
                line += f" to {f['valid_to']})" if f["valid_to"] else ")"
            lines.append(line)
        return "\n".join(lines)

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
                ended=valid_from or None,  # reuse valid_from as end date
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
        lines = []
        for f in results:
            date = f["valid_from"] or "?"
            end = f" → {f['valid_to']}" if f["valid_to"] else ""
            lines.append(
                f"[{date}{end}] {f['subject']} —{f['predicate']}→ {f['object']}"
            )
        return "\n".join(lines)

    else:
        actions = (
            "search, store, facts_query, facts_add, facts_invalidate, facts_timeline"
        )
        return f"Unknown action '{action}'. Use: {actions}"


memory_tool = ToolDef(
    name="memory",
    description=(
        "Mid/long-term memory — semantic search, storage, and a temporal knowledge graph. "
        "Use 'search' to recall information, 'store' to save something, "
        "and 'facts_*' actions to manage structured entity facts that can change over time."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "search",
                    "store",
                    "facts_query",
                    "facts_add",
                    "facts_invalidate",
                    "facts_timeline",
                ],
                "description": (
                    "search: semantic search over memories. "
                    "store: save content to memory. "
                    "facts_query: look up what you know about an entity. "
                    "facts_add: record a fact (subject-predicate-object). "
                    "facts_invalidate: mark a fact as no longer true. "
                    "facts_timeline: chronological history of an entity."
                ),
            },
            "query": {
                "type": "string",
                "description": "Natural-language search query (for search).",
            },
            "content": {
                "type": "string",
                "description": "Text to store in memory (for store).",
            },
            "source": {
                "type": "string",
                "description": "Label for the memory source, e.g. 'user', 'conversation'. Default: 'agent'.",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated key=value metadata, e.g. 'type=preference,topic=food'.",
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
                "description": "Relationship type, e.g. 'likes', 'lives_in', 'works_at' (for facts_add, facts_invalidate).",
            },
            "object": {
                "type": "string",
                "description": "Object entity (for facts_add, facts_invalidate).",
            },
            "valid_from": {
                "type": "string",
                "description": "ISO date (YYYY-MM-DD) when this fact became true, or end date for invalidation.",
            },
            "as_of": {
                "type": "string",
                "description": "ISO date to query facts as of a specific point in time (for facts_query).",
            },
            "n_results": {
                "type": "integer",
                "description": "Max results for search (default 5).",
            },
        },
        "required": ["action"],
    },
    handler=_memory,
)
