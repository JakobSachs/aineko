"""Tests for the semantic memory store — mocks ChromaDB, tests our wrapping logic."""

import pytest
from unittest.mock import MagicMock, patch
from hypothesis import given, settings as hsettings
from hypothesis import strategies as st

from aineko.memory.store import MemoryStore, SearchResult

# --- helpers ---


def _make_collection(**overrides):
    """Build a mock ChromaDB collection with sensible defaults."""
    col = MagicMock()
    col.count.return_value = overrides.get("count", 0)
    col.query.return_value = overrides.get(
        "query_result", {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    )
    col.get.return_value = overrides.get("get_result", {"ids": []})
    col.add.return_value = None
    col.delete.return_value = None
    return col


def _make_store(collection):
    """Build a MemoryStore with a pre-injected mock collection."""
    with patch("aineko.memory.store.chromadb"):
        store = object.__new__(MemoryStore)
        store._client = MagicMock()
        store._collection = collection
    return store


# --- strategies ---

storable_text = st.text(min_size=25, max_size=500).filter(
    lambda s: len(s.strip()) >= 25
)


# --- property-based tests ---


class TestMemoryStoreProperties:
    @given(query=st.text(min_size=1, max_size=200))
    @hsettings(max_examples=30)
    def test_search_empty_store_returns_empty(self, query):
        col = _make_collection(count=0)
        store = _make_store(col)
        result = store.search(query)
        assert result == []
        col.query.assert_not_called()  # short-circuits on count==0

    @given(source=st.text(min_size=1, max_size=50))
    @hsettings(max_examples=30)
    def test_delete_nonexistent_returns_zero(self, source):
        col = _make_collection()
        store = _make_store(col)
        result = store.delete_source(source)
        assert result == 0
        col.delete.assert_not_called()

    @given(n=st.integers(min_value=1, max_value=50))
    @hsettings(max_examples=20)
    def test_search_clamps_n_results_to_count(self, n):
        total = 3
        col = _make_collection(
            count=total,
            query_result={
                "documents": [["doc"] * min(n, total)],
                "metadatas": [[{"source": "s", "chunk_index": 0}] * min(n, total)],
                "distances": [[0.1] * min(n, total)],
            },
        )
        store = _make_store(col)
        store.search("test", n_results=n)
        called_n = (
            col.query.call_args[1].get("n_results") or col.query.call_args[0][0]
            if col.query.call_args[0]
            else col.query.call_args[1]["n_results"]
        )
        assert called_n <= total


# --- add tests ---


class TestMemoryStoreAdd:
    def test_add_chunks_and_stores(self):
        col = _make_collection(count=0)  # empty → no dedup checks
        store = _make_store(col)
        # Patch chunk_text to control output
        with patch(
            "aineko.memory.store.chunk_text", return_value=["chunk one", "chunk two"]
        ):
            ids = store.add("ignored", source="test")
        assert len(ids) == 2
        assert col.add.call_count == 2

    def test_add_empty_text_returns_empty(self):
        col = _make_collection()
        store = _make_store(col)
        with patch("aineko.memory.store.chunk_text", return_value=[]):
            ids = store.add("", source="test")
        assert ids == []
        col.add.assert_not_called()

    def test_add_includes_source_in_metadata(self):
        col = _make_collection(count=0)
        store = _make_store(col)
        with patch("aineko.memory.store.chunk_text", return_value=["hello world"]):
            store.add("ignored", source="my-source")
        meta = col.add.call_args[1]["metadatas"][0]
        assert meta["source"] == "my-source"

    def test_add_includes_tags_in_metadata(self):
        col = _make_collection(count=0)
        store = _make_store(col)
        with patch("aineko.memory.store.chunk_text", return_value=["hello world"]):
            store.add("ignored", source="s", tags={"type": "diary", "topic": "work"})
        meta = col.add.call_args[1]["metadatas"][0]
        assert meta["type"] == "diary"
        assert meta["topic"] == "work"
        assert meta["source"] == "s"

    def test_add_includes_chunk_index(self):
        col = _make_collection(count=0)
        store = _make_store(col)
        with patch("aineko.memory.store.chunk_text", return_value=["a", "b", "c"]):
            store.add("ignored", source="s")
        indexes = [
            call[1]["metadatas"][0]["chunk_index"] for call in col.add.call_args_list
        ]
        assert indexes == [0, 1, 2]

    def test_add_skips_duplicates(self):
        # Collection has 1 item, query returns high similarity
        col = _make_collection(
            count=1,
            query_result={
                "documents": [["existing"]],
                "metadatas": [[{}]],
                "distances": [[0.01]],
            },
        )
        store = _make_store(col)
        with patch("aineko.memory.store.chunk_text", return_value=["near duplicate"]):
            ids = store.add("ignored", source="s")
        assert ids == []
        col.add.assert_not_called()

    def test_add_stores_when_not_duplicate(self):
        # Collection has 1 item, query returns low similarity
        col = _make_collection(
            count=1,
            query_result={
                "documents": [["different"]],
                "metadatas": [[{}]],
                "distances": [[0.8]],
            },
        )
        store = _make_store(col)
        with patch("aineko.memory.store.chunk_text", return_value=["new content"]):
            ids = store.add("ignored", source="s")
        assert len(ids) == 1
        col.add.assert_called_once()


# --- search tests ---


class TestMemoryStoreSearch:
    def test_search_empty_store(self):
        col = _make_collection(count=0)
        store = _make_store(col)
        assert store.search("anything") == []

    def test_search_returns_search_results(self):
        col = _make_collection(
            count=2,
            query_result={
                "documents": [["doc one", "doc two"]],
                "metadatas": [
                    [
                        {"source": "a", "chunk_index": 0, "type": "note"},
                        {"source": "b", "chunk_index": 1},
                    ]
                ],
                "distances": [[0.1, 0.4]],
            },
        )
        store = _make_store(col)
        results = store.search("query")
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].content == "doc one"
        assert results[0].source == "a"
        assert results[0].similarity == round(1.0 - 0.1, 4)
        assert results[0].tags == {"type": "note"}
        assert results[1].source == "b"

    def test_search_passes_where_filter(self):
        col = _make_collection(
            count=1,
            query_result={
                "documents": [["x"]],
                "metadatas": [[{"source": "s", "chunk_index": 0}]],
                "distances": [[0.1]],
            },
        )
        store = _make_store(col)
        store.search("q", where={"type": "diary"})
        assert col.query.call_args[1]["where"] == {"type": "diary"}

    def test_search_no_where_by_default(self):
        col = _make_collection(
            count=1,
            query_result={
                "documents": [["x"]],
                "metadatas": [[{"source": "s", "chunk_index": 0}]],
                "distances": [[0.1]],
            },
        )
        store = _make_store(col)
        store.search("q")
        assert "where" not in col.query.call_args[1]

    def test_search_similarity_ordering(self):
        col = _make_collection(
            count=3,
            query_result={
                "documents": [["a", "b", "c"]],
                "metadatas": [[{"source": "s", "chunk_index": 0}] * 3],
                "distances": [[0.1, 0.3, 0.7]],
            },
        )
        store = _make_store(col)
        results = store.search("q")
        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_search_empty_query_result(self):
        col = _make_collection(
            count=5,
            query_result={"documents": [[]], "metadatas": [[]], "distances": [[]]},
        )
        store = _make_store(col)
        assert store.search("q") == []


# --- delete tests ---


class TestMemoryStoreDelete:
    def test_delete_source_removes_matching_ids(self):
        col = _make_collection(get_result={"ids": ["id1", "id2", "id3"]})
        store = _make_store(col)
        deleted = store.delete_source("old-source")
        assert deleted == 3
        col.delete.assert_called_once_with(ids=["id1", "id2", "id3"])

    def test_delete_source_filters_by_source(self):
        col = _make_collection(get_result={"ids": []})
        store = _make_store(col)
        store.delete_source("my-source")
        col.get.assert_called_once_with(where={"source": "my-source"}, include=[])

    def test_delete_nonexistent_returns_zero(self):
        col = _make_collection(get_result={"ids": []})
        store = _make_store(col)
        assert store.delete_source("nope") == 0
        col.delete.assert_not_called()


# --- count tests ---


class TestMemoryStoreCount:
    def test_count_delegates_to_collection(self):
        col = _make_collection(count=42)
        store = _make_store(col)
        assert store.count() == 42
        col.count.assert_called()


# --- dedup threshold tests ---


class TestMemoryStoreDedup:
    @pytest.mark.parametrize(
        "distance,expected_dup",
        [
            (0.0, True),  # identical (similarity=1.0)
            (0.04, True),  # similarity=0.96, above threshold
            (0.05, True),  # similarity=0.95, at threshold (>= 0.95)
            (0.06, False),  # similarity=0.94, below threshold
            (0.5, False),  # clearly different
            (1.0, False),  # maximally different
        ],
    )
    def test_dedup_threshold_boundary(self, distance, expected_dup):
        col = _make_collection(
            count=1,
            query_result={
                "documents": [["x"]],
                "metadatas": [[{}]],
                "distances": [[distance]],
            },
        )
        store = _make_store(col)
        is_dup = store._is_duplicate("test text")
        assert is_dup == expected_dup

    def test_dedup_skips_check_on_empty_store(self):
        col = _make_collection(count=0)
        store = _make_store(col)
        assert store._is_duplicate("anything") is False
        col.query.assert_not_called()
