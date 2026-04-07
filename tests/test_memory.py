"""Tests for the memory tool (semantic search + KG wrapper)."""

import pytest
import pytest_asyncio
from hypothesis import given, settings as hsettings, HealthCheck
from hypothesis import strategies as st
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from unittest.mock import patch

from aineko.db import Base
from aineko.tools.memory import _memory
import aineko.tools.memory as memory_mod
import aineko.db as db_mod

# --- strategies ---

VALID_ACTIONS = [
    "search",
    "store",
    "facts_query",
    "facts_add",
    "facts_invalidate",
    "facts_timeline",
]

# Strings that are NOT valid actions
invalid_action = st.text(min_size=1, max_size=50).filter(
    lambda s: s not in VALID_ACTIONS
)

# Realistic entity-like strings
entity_text = st.text(
    alphabet=st.sampled_from(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-"
    ),
    min_size=1,
    max_size=40,
)

# --- fixtures ---


@pytest_asyncio.fixture
async def setup_db(tmp_path):
    """Initialize test database for KG operations."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_async_engine(url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    original = db_mod.async_session_factory
    db_mod.async_session_factory = factory
    yield
    db_mod.async_session_factory = original
    await engine.dispose()


@pytest.fixture(autouse=True)
def fresh_store():
    """Give each test a fresh ephemeral MemoryStore."""
    from aineko.memory.store import MemoryStore

    store = MemoryStore(persist_dir=None)
    with patch.object(memory_mod, "_store", store):
        with patch.object(memory_mod, "_get_store", return_value=store):
            yield store


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
        assert "Error" not in result  # all fields provided, should succeed

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
    async def test_search_finds_stored(self, fresh_store):
        fresh_store.add(
            "Jakob loves hiking in the mountains on weekends.",
            source="user",
        )
        result = await _memory("search", query="hiking mountains")
        assert "hiking" in result

    @pytest.mark.asyncio
    async def test_search_shows_similarity(self, fresh_store):
        fresh_store.add(
            "The project deadline is next Friday for the demo.",
            source="work",
        )
        result = await _memory("search", query="project deadline")
        assert "%" in result

    @pytest.mark.asyncio
    async def test_search_with_tags_filter(self, fresh_store):
        fresh_store.add(
            "Important meeting notes about project alpha launch.",
            source="work",
            tags={"type": "meeting"},
        )
        fresh_store.add(
            "Grocery list for the week including fruits and vegetables.",
            source="personal",
            tags={"type": "todo"},
        )
        result = await _memory("search", query="weekly plans", tags="type=meeting")
        assert "meeting" in result.lower() or "project" in result.lower()


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_store_requires_content(self):
        result = await _memory("store")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_store_saves_content(self, fresh_store):
        result = await _memory(
            "store", content="Important note about the deployment process."
        )
        assert "Stored" in result
        assert fresh_store.count() >= 1

    @pytest.mark.asyncio
    async def test_store_with_source(self, fresh_store):
        result = await _memory(
            "store",
            content="User prefers dark mode in every application.",
            source="user",
        )
        assert "user" in result

    @pytest.mark.asyncio
    async def test_store_default_source(self, fresh_store):
        result = await _memory(
            "store", content="A note without an explicit source label."
        )
        assert "agent" in result

    @pytest.mark.asyncio
    async def test_store_dedup(self, fresh_store):
        await _memory(
            "store",
            content="Duplicate content that should not be stored twice in memory.",
        )
        result = await _memory(
            "store",
            content="Duplicate content that should not be stored twice in memory.",
        )
        assert "duplicate" in result.lower()

    @pytest.mark.asyncio
    async def test_store_with_tags(self, fresh_store):
        result = await _memory(
            "store",
            content="Meeting notes from the sprint planning session on Monday.",
            tags="type=meeting,topic=sprint",
        )
        assert "Stored" in result

    @pytest.mark.asyncio
    async def test_store_then_search_roundtrip(self, fresh_store):
        await _memory("store", content="The server runs on port 8080 in production.")
        result = await _memory("search", query="server port production")
        assert "8080" in result


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
        # swimming should appear before chess (chronological)
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
