"""Memory subsystems for aineko.

- daily_log: Append-only daily markdown logs (active runtime store)
- kg: Temporal fact store (SQLAlchemy on existing DB)
"""

from aineko.memory.daily_log import append_to_today, read_log, search_logs

__all__ = ["append_to_today", "read_log", "search_logs"]
