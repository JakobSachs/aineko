"""Local-time helpers driven by a user-editable timezone file.

Human-facing code (heartbeat active hours, daily log filenames, message
timestamps for the LLM) should use ``now_local()`` / ``today_local()`` so it
tracks whichever timezone the user is currently in. Storage code keeps using
UTC-aware datetimes.

The timezone is read from ``<data_dir>/timezone`` (single line, e.g.
``Europe/Berlin``). Edits are picked up on next read without restart via an
mtime check. Falls back to ``Europe/Berlin`` on missing or invalid content.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TZ = "Europe/Berlin"

_cached_tz: ZoneInfo | None = None
_cached_mtime: float | None = None
_cached_name: str | None = None


def _tz_file() -> Path:
    data_dir = os.environ.get("AINEKO_DATA_DIR", "/data")
    return Path(data_dir) / "timezone"


def local_tz() -> ZoneInfo:
    """Return the currently-configured local timezone, re-reading the file if
    it has changed since last call."""
    global _cached_tz, _cached_mtime, _cached_name

    path = _tz_file()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None

    if _cached_tz is not None and mtime == _cached_mtime:
        return _cached_tz

    name = DEFAULT_TZ
    if mtime is not None:
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if raw:
                ZoneInfo(raw)  # validate
                name = raw
        except (OSError, ZoneInfoNotFoundError, ValueError):
            name = DEFAULT_TZ

    _cached_tz = ZoneInfo(name)
    _cached_mtime = mtime
    _cached_name = name
    return _cached_tz


def now_local() -> datetime:
    """Timezone-aware ``datetime`` in the configured local timezone."""
    return datetime.now(timezone.utc).astimezone(local_tz())


def today_local() -> date:
    """Current date in the configured local timezone."""
    return now_local().date()
