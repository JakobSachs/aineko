"""Semantic memory system for aineko.

Three subsystems:
- MemoryStore: ChromaDB-backed chunk store with embedding search
- KnowledgeGraph: Temporal fact store (SQLAlchemy on existing DB)
- ingest: Conversation → chunk pipeline
"""

from aineko.memory.chunker import chunk_text
from aineko.memory.store import MemoryStore, SearchResult

__all__ = ["chunk_text", "MemoryStore", "SearchResult"]
