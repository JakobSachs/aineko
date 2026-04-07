"""Temporal knowledge graph — entity facts with time validity.

Stores (subject, predicate, object) triples with optional valid_from / valid_to
dates.  Built on the existing async SQLAlchemy engine so it shares the same
database as conversations.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import DateTime, Integer, String, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from aineko.db import Base


class Fact(Base):
    """A single fact / triple with temporal validity."""

    __tablename__ = "facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject: Mapped[str] = mapped_column(String, index=True)
    predicate: Mapped[str] = mapped_column(String)
    object_: Mapped[str] = mapped_column("object", String, index=True)
    valid_from: Mapped[str | None] = mapped_column(String, nullable=True)
    valid_to: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object_,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "source": self.source,
        }


class KnowledgeGraph:
    """Async operations on the temporal fact store."""

    async def add(
        self,
        db: AsyncSession,
        subject: str,
        predicate: str,
        obj: str,
        *,
        valid_from: str | None = None,
        source: str | None = None,
    ) -> int:
        """Add a fact.  Returns the fact ID.

        If an identical *active* fact (same subject/predicate/object,
        ``valid_to IS NULL``) already exists, returns its ID without
        inserting a duplicate.

        Entity matching is case-insensitive.
        """
        # Check for existing active duplicate
        existing = await db.execute(
            select(Fact).where(
                func.lower(Fact.subject) == subject.lower(),
                func.lower(Fact.predicate) == predicate.lower(),
                func.lower(Fact.object_) == obj.lower(),
                Fact.valid_to.is_(None),
            )
        )
        found = existing.scalar_one_or_none()
        if found is not None:
            return found.id

        fact = Fact(
            subject=subject,
            predicate=predicate,
            object_=obj,
            valid_from=valid_from,
            source=source,
        )
        db.add(fact)
        await db.flush()
        return fact.id

    async def query(
        self,
        db: AsyncSession,
        entity: str,
        *,
        as_of: str | None = None,
    ) -> list[dict]:
        """All facts where *entity* appears as subject **or** object.

        If *as_of* is given (ISO date string ``YYYY-MM-DD``), only facts
        valid at that point in time are returned.

        Returns list of dicts (see :meth:`Fact.to_dict`).
        """
        entity_lower = entity.lower()
        stmt = select(Fact).where(
            or_(
                func.lower(Fact.subject) == entity_lower,
                func.lower(Fact.object_) == entity_lower,
            )
        )

        if as_of is not None:
            stmt = stmt.where(
                or_(Fact.valid_from.is_(None), Fact.valid_from <= as_of),
                or_(Fact.valid_to.is_(None), Fact.valid_to >= as_of),
            )

        result = await db.execute(stmt)
        return [f.to_dict() for f in result.scalars().all()]

    async def invalidate(
        self,
        db: AsyncSession,
        subject: str,
        predicate: str,
        obj: str,
        *,
        ended: str | None = None,
    ) -> bool:
        """Mark an active fact as no longer true by setting ``valid_to``.

        *ended* defaults to today's ISO date if not provided.
        Returns ``True`` if a matching active fact was found and updated,
        ``False`` otherwise.
        """
        if ended is None:
            ended = date.today().isoformat()

        result = await db.execute(
            select(Fact).where(
                func.lower(Fact.subject) == subject.lower(),
                func.lower(Fact.predicate) == predicate.lower(),
                func.lower(Fact.object_) == obj.lower(),
                Fact.valid_to.is_(None),
            )
        )
        fact = result.scalar_one_or_none()
        if fact is None:
            return False

        fact.valid_to = ended
        await db.flush()
        return True

    async def timeline(
        self,
        db: AsyncSession,
        entity: str,
    ) -> list[dict]:
        """All facts about *entity*, ordered chronologically by valid_from.

        Facts with ``valid_from IS NULL`` sort first.
        """
        entity_lower = entity.lower()
        result = await db.execute(
            select(Fact)
            .where(
                or_(
                    func.lower(Fact.subject) == entity_lower,
                    func.lower(Fact.object_) == entity_lower,
                )
            )
            .order_by(
                Fact.valid_from.is_(None).desc(),  # NULLs first
                Fact.valid_from.asc(),
            )
        )
        return [f.to_dict() for f in result.scalars().all()]
