"""Semantic memory store backed by ChromaDB.

Stores text as embedded chunks with metadata tags.
Supports filtered semantic search and source-level deletion.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from aineko.memory.chunker import chunk_text

DEDUP_THRESHOLD = 0.95


@dataclass
class SearchResult:
    """A single search hit."""

    content: str
    source: str
    similarity: float
    tags: dict[str, str] = field(default_factory=dict)
    chunk_index: int = 0


class MemoryStore:
    """Semantic memory store backed by ChromaDB.

    Parameters
    ----------
    persist_dir:
        Directory for persistent storage.
        Pass ``None`` for an ephemeral (in-memory) store (useful in tests).
    """

    def __init__(self, persist_dir: str | Path | None = None) -> None:
        if persist_dir is None:
            self._client = chromadb.EphemeralClient()
            name = f"memories_{uuid.uuid4().hex[:8]}"
        else:
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            name = "memories"
        self._collection = self._client.get_or_create_collection(
            name,
            metadata={"hnsw:space": "cosine"},
        )

    def _chunk_id(self, source: str, index: int, text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()[:12]
        return f"{source}:{index}:{h}"

    def _is_duplicate(self, text: str) -> bool:
        if self._collection.count() == 0:
            return False
        results = self._collection.query(query_texts=[text], n_results=1)
        if results["distances"] and results["distances"][0]:
            distance = results["distances"][0][0]
            similarity = 1.0 - distance
            if similarity >= DEDUP_THRESHOLD:
                return True
        return False

    def add(
        self,
        text: str,
        source: str,
        tags: dict[str, str] | None = None,
    ) -> list[str]:
        """Chunk text and store with metadata.

        Returns list of chunk IDs that were actually stored.
        Near-duplicate chunks (>= 0.95 cosine similarity to an existing
        chunk) are silently skipped.
        """
        chunks = chunk_text(text)
        if not chunks:
            return []

        stored_ids: list[str] = []
        for i, chunk in enumerate(chunks):
            if self._is_duplicate(chunk):
                continue

            chunk_id = self._chunk_id(source, i, chunk)
            meta = {"source": source, "chunk_index": i}
            if tags:
                meta.update(tags)

            self._collection.add(
                documents=[chunk],
                ids=[chunk_id],
                metadatas=[meta],
            )
            stored_ids.append(chunk_id)

        return stored_ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """Semantic search over stored memories."""
        if self._collection.count() == 0:
            return []

        kwargs: dict = {
            "query_texts": [query],
            "n_results": min(n_results, self._collection.count()),
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(
            **kwargs,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        hits: list[SearchResult] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = round(1.0 - dist, 4)
            source = meta.pop("source", "")
            chunk_index = meta.pop("chunk_index", 0)
            hits.append(
                SearchResult(
                    content=doc,
                    source=source,
                    similarity=similarity,
                    tags=meta,
                    chunk_index=chunk_index,
                )
            )

        return hits

    def delete_source(self, source: str) -> int:
        """Delete all chunks filed under *source*.  Returns count deleted."""
        existing = self._collection.get(
            where={"source": source},
            include=[],
        )
        ids = existing["ids"]
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        """Total number of chunks in the store."""
        return self._collection.count()
