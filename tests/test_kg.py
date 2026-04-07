"""Tests for the temporal knowledge graph."""

import pytest
import pytest_asyncio
from hypothesis import given, settings as hsettings, HealthCheck
from hypothesis import strategies as st
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from aineko.db import Base
from aineko.memory.kg import KnowledgeGraph

# --- strategies ---

entity_text = st.text(
    alphabet=st.sampled_from(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-"
    ),
    min_size=1,
    max_size=40,
)

iso_date = st.dates(
    min_value=__import__("datetime").date(2000, 1, 1),
    max_value=__import__("datetime").date(2030, 12, 31),
).map(lambda d: d.isoformat())


# --- fixtures ---


@pytest_asyncio.fixture
async def db_session(tmp_path):
    """Fresh SQLite database with facts table for each test."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_async_engine(url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture
def kg():
    return KnowledgeGraph()


# --- property-based tests ---


class TestKGProperties:
    @given(subject=entity_text, predicate=entity_text, obj=entity_text)
    @hsettings(
        max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_add_never_crashes(self, db_session, kg, subject, predicate, obj):
        result = await kg.add(db_session, subject, predicate, obj)
        assert isinstance(result, int)
        assert result > 0

    @given(entity=entity_text)
    @hsettings(
        max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_query_never_crashes(self, db_session, kg, entity):
        result = await kg.query(db_session, entity)
        assert isinstance(result, list)

    @given(entity=entity_text)
    @hsettings(
        max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_timeline_never_crashes(self, db_session, kg, entity):
        result = await kg.timeline(db_session, entity)
        assert isinstance(result, list)

    @given(subject=entity_text, predicate=entity_text, obj=entity_text)
    @hsettings(
        max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_invalidate_never_crashes(
        self, db_session, kg, subject, predicate, obj
    ):
        result = await kg.invalidate(db_session, subject, predicate, obj)
        assert isinstance(result, bool)

    @given(subject=entity_text, predicate=entity_text, obj=entity_text)
    @hsettings(
        max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_add_then_query_roundtrip(
        self, db_session, kg, subject, predicate, obj
    ):
        await kg.add(db_session, subject, predicate, obj)
        results = await kg.query(db_session, subject)
        assert len(results) >= 1
        match = [
            r
            for r in results
            if r["predicate"].lower() == predicate.lower()
            and r["object"].lower() == obj.lower()
        ]
        assert len(match) >= 1

    @given(
        subject=entity_text,
        predicate=entity_text,
        obj=entity_text,
        case_fn=st.sampled_from([str.lower, str.upper, str.title, str.swapcase]),
    )
    @hsettings(
        max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_case_insensitive_query(
        self, db_session, kg, subject, predicate, obj, case_fn
    ):
        await kg.add(db_session, subject, predicate, obj)
        results = await kg.query(db_session, case_fn(subject))
        assert len(results) >= 1

    @given(
        subject=entity_text,
        predicate=entity_text,
        obj=entity_text,
        date=iso_date,
    )
    @hsettings(
        max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_add_with_date_queryable(
        self, db_session, kg, subject, predicate, obj, date
    ):
        await kg.add(db_session, subject, predicate, obj, valid_from=date)
        results = await kg.query(db_session, subject, as_of=date)
        assert len(results) >= 1

    @given(subject=entity_text, predicate=entity_text, obj=entity_text)
    @hsettings(
        max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_add_invalidate_roundtrip(
        self, db_session, kg, subject, predicate, obj
    ):
        await kg.add(db_session, subject, predicate, obj)
        result = await kg.invalidate(db_session, subject, predicate, obj)
        assert result is True
        # After invalidation, adding again creates a new fact
        id2 = await kg.add(db_session, subject, predicate, obj)
        assert isinstance(id2, int)


# --- targeted tests ---


class TestKGAdd:
    @pytest.mark.asyncio
    async def test_add_returns_id(self, db_session, kg):
        fact_id = await kg.add(db_session, "Max", "child_of", "Alice")
        assert isinstance(fact_id, int)
        assert fact_id > 0

    @pytest.mark.asyncio
    async def test_add_with_valid_from(self, db_session, kg):
        fact_id = await kg.add(
            db_session, "Max", "does", "swimming", valid_from="2025-01-15"
        )
        assert fact_id > 0

    @pytest.mark.asyncio
    async def test_add_with_source(self, db_session, kg):
        fact_id = await kg.add(
            db_session, "Max", "likes", "chess", source="conversation:42"
        )
        assert fact_id > 0

    @pytest.mark.asyncio
    async def test_add_dedup_active_fact(self, db_session, kg):
        id1 = await kg.add(db_session, "Max", "child_of", "Alice")
        id2 = await kg.add(db_session, "Max", "child_of", "Alice")
        assert id1 == id2

    @pytest.mark.asyncio
    async def test_add_dedup_case_insensitive(self, db_session, kg):
        id1 = await kg.add(db_session, "Max", "child_of", "Alice")
        id2 = await kg.add(db_session, "max", "child_of", "alice")
        assert id1 == id2

    @pytest.mark.asyncio
    async def test_add_different_predicate_not_deduped(self, db_session, kg):
        id1 = await kg.add(db_session, "Max", "child_of", "Alice")
        id2 = await kg.add(db_session, "Max", "friend_of", "Alice")
        assert id1 != id2


class TestKGQuery:
    @pytest.mark.asyncio
    async def test_query_as_subject(self, db_session, kg):
        await kg.add(db_session, "Max", "child_of", "Alice")
        results = await kg.query(db_session, "Max")
        assert len(results) == 1
        assert results[0]["subject"] == "Max"
        assert results[0]["predicate"] == "child_of"
        assert results[0]["object"] == "Alice"

    @pytest.mark.asyncio
    async def test_query_as_object(self, db_session, kg):
        await kg.add(db_session, "Max", "child_of", "Alice")
        results = await kg.query(db_session, "Alice")
        assert len(results) == 1
        assert results[0]["object"] == "Alice"

    @pytest.mark.asyncio
    async def test_query_case_insensitive(self, db_session, kg):
        await kg.add(db_session, "Max", "child_of", "Alice")
        results = await kg.query(db_session, "max")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_no_results(self, db_session, kg):
        results = await kg.query(db_session, "Nobody")
        assert results == []

    @pytest.mark.asyncio
    async def test_query_multiple_facts(self, db_session, kg):
        await kg.add(db_session, "Max", "child_of", "Alice")
        await kg.add(db_session, "Max", "does", "swimming")
        await kg.add(db_session, "Max", "likes", "chess")
        results = await kg.query(db_session, "Max")
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_as_of_includes_valid(self, db_session, kg):
        await kg.add(db_session, "Max", "does", "swimming", valid_from="2025-01-01")
        results = await kg.query(db_session, "Max", as_of="2025-06-01")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_as_of_excludes_future(self, db_session, kg):
        await kg.add(db_session, "Max", "does", "swimming", valid_from="2025-06-01")
        results = await kg.query(db_session, "Max", as_of="2025-01-01")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_query_as_of_excludes_ended(self, db_session, kg):
        await kg.add(db_session, "Max", "does", "swimming", valid_from="2025-01-01")
        await kg.invalidate(db_session, "Max", "does", "swimming", ended="2025-03-01")
        results = await kg.query(db_session, "Max", as_of="2025-06-01")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_query_returns_dicts(self, db_session, kg):
        await kg.add(db_session, "Max", "child_of", "Alice")
        results = await kg.query(db_session, "Max")
        assert isinstance(results[0], dict)
        expected_keys = {
            "id",
            "subject",
            "predicate",
            "object",
            "valid_from",
            "valid_to",
            "source",
        }
        assert expected_keys == set(results[0].keys())


class TestKGInvalidate:
    @pytest.mark.asyncio
    async def test_invalidate_sets_valid_to(self, db_session, kg):
        await kg.add(db_session, "Max", "does", "swimming")
        result = await kg.invalidate(
            db_session, "Max", "does", "swimming", ended="2025-06-01"
        )
        assert result is True
        facts = await kg.query(db_session, "Max")
        assert facts[0]["valid_to"] == "2025-06-01"

    @pytest.mark.asyncio
    async def test_invalidate_defaults_to_today(self, db_session, kg):
        await kg.add(db_session, "Max", "does", "swimming")
        result = await kg.invalidate(db_session, "Max", "does", "swimming")
        assert result is True
        facts = await kg.query(db_session, "Max")
        assert facts[0]["valid_to"] is not None

    @pytest.mark.asyncio
    async def test_invalidate_not_found(self, db_session, kg):
        result = await kg.invalidate(db_session, "Max", "does", "flying")
        assert result is False

    @pytest.mark.asyncio
    async def test_invalidate_case_insensitive(self, db_session, kg):
        await kg.add(db_session, "Max", "does", "swimming")
        result = await kg.invalidate(db_session, "max", "does", "swimming")
        assert result is True

    @pytest.mark.asyncio
    async def test_invalidate_only_active(self, db_session, kg):
        await kg.add(db_session, "Max", "does", "swimming")
        await kg.invalidate(db_session, "Max", "does", "swimming", ended="2025-03-01")
        result = await kg.invalidate(db_session, "Max", "does", "swimming")
        assert result is False

    @pytest.mark.asyncio
    async def test_add_after_invalidate_creates_new(self, db_session, kg):
        id1 = await kg.add(
            db_session, "Max", "does", "swimming", valid_from="2025-01-01"
        )
        await kg.invalidate(db_session, "Max", "does", "swimming", ended="2025-03-01")
        id2 = await kg.add(
            db_session, "Max", "does", "swimming", valid_from="2025-09-01"
        )
        assert id1 != id2


class TestKGTimeline:
    @pytest.mark.asyncio
    async def test_timeline_chronological(self, db_session, kg):
        await kg.add(db_session, "Max", "does", "chess", valid_from="2025-10-01")
        await kg.add(db_session, "Max", "does", "swimming", valid_from="2025-01-01")
        await kg.add(db_session, "Max", "child_of", "Alice", valid_from="2015-04-01")
        timeline = await kg.timeline(db_session, "Max")
        assert len(timeline) == 3
        dates = [f["valid_from"] for f in timeline]
        assert dates == sorted(dates)

    @pytest.mark.asyncio
    async def test_timeline_null_dates_first(self, db_session, kg):
        await kg.add(db_session, "Max", "likes", "cheese")
        await kg.add(db_session, "Max", "does", "swimming", valid_from="2025-01-01")
        timeline = await kg.timeline(db_session, "Max")
        assert timeline[0]["valid_from"] is None
        assert timeline[1]["valid_from"] == "2025-01-01"

    @pytest.mark.asyncio
    async def test_timeline_empty(self, db_session, kg):
        timeline = await kg.timeline(db_session, "Nobody")
        assert timeline == []

    @pytest.mark.asyncio
    async def test_timeline_includes_invalidated(self, db_session, kg):
        await kg.add(db_session, "Max", "does", "swimming", valid_from="2025-01-01")
        await kg.invalidate(db_session, "Max", "does", "swimming", ended="2025-06-01")
        timeline = await kg.timeline(db_session, "Max")
        assert len(timeline) == 1
        assert timeline[0]["valid_to"] == "2025-06-01"
