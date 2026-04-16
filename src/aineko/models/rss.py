"""RSS seen-item tracking model."""

from datetime import datetime

from sqlalchemy import DateTime, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from aineko.db import Base


class RssSeenItem(Base):
    __tablename__ = "rss_seen_items"
    __table_args__ = (UniqueConstraint("feed_url", "guid"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    feed_url: Mapped[str] = mapped_column(String, index=True)
    guid: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String, default="")
    seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
