"""Tests for the memory tool (daily-log search/store + KG wrapper)."""

import pytest
import pytest_asyncio
from hypothesis import given, settings as hsettings, HealthCheck
from hypothesis import strategies as st

from aineko.tools.memory import _memory
import aineko.tools.memory as memory_mod

# --- strategies ---

VALID_ACTIONS = [
    "search",
    "store",
    "read",
    "facts_query",
    "facts_add",
    "facts_invalidate",
    "facts_timeline",
]

invalid_action = st.text(min_size=1, max_size=50).filter(
    lambda s: s not in VALID_ACTIONS
)

entity_text = st.text(
    alphabet=st.sampled_from(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-"
    ),
    min_size=1,
    max_size=40,
)


# --- fixtures ---


@pytest_asyncio.fixture
async def setup_db(pg_db):
    yield


@pytest.fixture(autouse=True)
def memory_dir(tmp_path, monkeypatch):
    """Give each test a fresh empty memory dir."""
    d = tmp_path / "memory"
    d.mkdir()
    monkeypatch.setattr(memory_mod, "_memory_dir", d)
    return d


# --- property-based tests ---


class TestMemoryProperties:
    @given(action=invalid_action)
    @hsettings(
        max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_invalid_action_returns_unknown(self, action):
        result = await _memory(action)
        assert "Unknown action" in result

    @given(query=st.text(min_size=1, max_size=200))
    @hsettings(
        max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_search_never_crashes(self, query):
        result = await _memory("search", query=query)
        assert isinstance(result, str)

    @given(content=st.text(min_size=1, max_size=500))
    @hsettings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    @pytest.mark.asyncio
    async def test_store_never_crashes(self, content):
        result = await _memory("store", content=content)
        assert isinstance(result, str)

    @given(entity=entity_text)
    @hsettings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    @pytest.mark.asyncio
    async def test_facts_query_never_crashes(self, setup_db, entity):
        result = await _memory("facts_query", entity=entity)
        assert isinstance(result, str)

    @given(
        subject=entity_text,
        predicate=entity_text,
        obj=entity_text,
    )
    @hsettings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    @pytest.mark.asyncio
    async def test_facts_add_never_crashes(self, setup_db, subject, predicate, obj):
        result = await _memory(
            "facts_add", subject=subject, predicate=predicate, object=obj
        )
        assert isinstance(result, str)
        assert "Error" not in result

    @given(entity=entity_text)
    @hsettings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    @pytest.mark.asyncio
    async def test_facts_timeline_never_crashes(self, setup_db, entity):
        result = await _memory("facts_timeline", entity=entity)
        assert isinstance(result, str)


# --- targeted tests ---


class TestMemorySearch:
    @pytest.mark.asyncio
    async def test_search_requires_query(self):
        result = await _memory("search")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        result = await _memory("search", query="nonexistent topic")
        assert "No memories" in result

    @pytest.mark.asyncio
    async def test_search_finds_stored(self):
        await _memory(
            "store", content="Jakob loves hiking in the mountains on weekends."
        )
        result = await _memory("search", query="hiking")
        assert "hiking" in result

    @pytest.mark.asyncio
    async def test_search_returns_path_and_line(self, memory_dir):
        await _memory("store", content="unique marker xyz123")
        result = await _memory("search", query="xyz123")
        assert str(memory_dir) in result
        assert ":" in result

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self):
        await _memory("store", content="Project Alpha launches next quarter.")
        result = await _memory("search", query="project alpha")
        assert "Alpha" in result or "alpha" in result


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_store_requires_content(self):
        result = await _memory("store")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_store_writes_file(self, memory_dir):
        result = await _memory(
            "store", content="Important note about the deployment process."
        )
        assert "Appended" in result
        files = list(memory_dir.glob("*.md"))
        assert len(files) == 1
        assert "deployment" in files[0].read_text()

    @pytest.mark.asyncio
    async def test_store_is_append_only(self, memory_dir):
        await _memory("store", content="first entry text")
        await _memory("store", content="second entry text")
        files = list(memory_dir.glob("*.md"))
        assert len(files) == 1
        text = files[0].read_text()
        assert "first entry text" in text
        assert "second entry text" in text

    @pytest.mark.asyncio
    async def test_store_records_source_and_tags(self, memory_dir):
        await _memory(
            "store",
            content="sprint planning notes",
            source="user",
            tags="type=meeting",
        )
        text = next(memory_dir.glob("*.md")).read_text()
        assert "user" in text
        assert "meeting" in text

    @pytest.mark.asyncio
    async def test_store_then_search_roundtrip(self):
        await _memory("store", content="The server runs on port 8080 in production.")
        result = await _memory("search", query="8080")
        assert "8080" in result


class TestMemoryRead:
    @pytest.mark.asyncio
    async def test_read_requires_path(self):
        result = await _memory("read")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_read_missing_file(self, memory_dir):
        result = await _memory("read", path="nope.md")
        assert "No content" in result

    @pytest.mark.asyncio
    async def test_read_returns_content(self, memory_dir):
        await _memory("store", content="line one two three")
        log_path = next(memory_dir.glob("*.md"))
        result = await _memory("read", path=log_path.name)
        assert "line one two three" in result


class TestMemoryFacts:
    @pytest.mark.asyncio
    async def test_facts_query_requires_entity(self, setup_db):
        result = await _memory("facts_query")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_facts_add_and_query(self, setup_db):
        await _memory(
            "facts_add", subject="Jakob", predicate="lives_in", object="Berlin"
        )
        result = await _memory("facts_query", entity="Jakob")
        assert "Berlin" in result
        assert "lives_in" in result

    @pytest.mark.asyncio
    async def test_facts_invalidate(self, setup_db):
        await _memory("facts_add", subject="Jakob", predicate="works_at", object="Acme")
        result = await _memory(
            "facts_invalidate", subject="Jakob", predicate="works_at", object="Acme"
        )
        assert "Invalidated" in result

    @pytest.mark.asyncio
    async def test_facts_invalidate_not_found(self, setup_db):
        result = await _memory(
            "facts_invalidate", subject="Nobody", predicate="does", object="nothing"
        )
        assert "No matching" in result

    @pytest.mark.asyncio
    async def test_facts_timeline(self, setup_db):
        await _memory(
            "facts_add",
            subject="Max",
            predicate="does",
            object="swimming",
            valid_from="2025-01-01",
        )
        await _memory(
            "facts_add",
            subject="Max",
            predicate="does",
            object="chess",
            valid_from="2025-06-01",
        )
        result = await _memory("facts_timeline", entity="Max")
        assert "swimming" in result
        assert "chess" in result
        assert result.index("swimming") < result.index("chess")

    @pytest.mark.asyncio
    async def test_facts_timeline_requires_entity(self, setup_db):
        result = await _memory("facts_timeline")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_facts_add_requires_all_fields(self, setup_db):
        result = await _memory("facts_add", subject="Max")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_facts_add_requires_predicate(self, setup_db):
        result = await _memory("facts_add", subject="Max", object="Berlin")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_facts_invalidate_requires_all_fields(self, setup_db):
        result = await _memory("facts_invalidate", subject="Max")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_facts_query_no_results(self, setup_db):
        result = await _memory("facts_query", entity="GhostEntity")
        assert "No facts" in result

    @pytest.mark.asyncio
    async def test_facts_add_with_valid_from(self, setup_db):
        result = await _memory(
            "facts_add",
            subject="Max",
            predicate="started",
            object="university",
            valid_from="2025-09-01",
        )
        assert "Fact #" in result
        query = await _memory("facts_query", entity="Max")
        assert "2025-09-01" in query
