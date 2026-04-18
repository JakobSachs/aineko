"""Workspace bootstrap files — user-editable markdown loaded into the system prompt.

Files live at the data dir root (uppercase convention). Non-recursive. Missing
files are skipped silently; oversize files are truncated with a warning.

Session types (`main`/`subagent`/`cron`) currently all return the full set —
the hook is present so that when PromptMode lands we can filter without
touching consumers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

SessionType = Literal["main", "subagent", "cron"]

# Load order matches desired render order in the system prompt.
BOOTSTRAP_FILES: tuple[str, ...] = (
    "SOUL.md",
    "AGENTS.md",
    "TOOLS.md",
    "USER.md",
    "HEARTBEAT.md",
    "MEMORY.md",
)

# Backwards-compat fallbacks: canonical name → legacy path (relative to data_dir).
FALLBACKS: dict[str, str] = {
    "SOUL.md": "soul.md",
    "MEMORY.md": "memory/memory.md",
}

# Subagents/cron skip heartbeat/memory — they're ephemeral and task-scoped.
# All three currently resolve to the full set; wire filters here when PromptMode lands.
SESSION_FILTERS: dict[SessionType, tuple[str, ...]] = {
    "main": BOOTSTRAP_FILES,
    "subagent": BOOTSTRAP_FILES,
    "cron": BOOTSTRAP_FILES,
}

MAX_FILE_BYTES = 500_000


@dataclass(frozen=True)
class BootstrapFile:
    name: str
    path: Path
    content: str


_cache: dict[tuple[Path, str], tuple[float, BootstrapFile]] = {}
_deprecation_warned: set[str] = set()


def _resolve_path(data_dir: Path, name: str) -> Path | None:
    canonical = data_dir / name
    if canonical.exists():
        return canonical
    fallback_rel = FALLBACKS.get(name)
    if fallback_rel:
        legacy = data_dir / fallback_rel
        if legacy.exists():
            if name not in _deprecation_warned:
                logger.warning(
                    "bootstrap: using legacy path %s for %s; rename to %s",
                    legacy,
                    name,
                    canonical,
                )
                _deprecation_warned.add(name)
            return legacy
    return None


def _read(path: Path) -> str:
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        logger.warning(
            "bootstrap: %s exceeds %d bytes (%d); truncating",
            path,
            MAX_FILE_BYTES,
            len(data),
        )
        data = data[:MAX_FILE_BYTES]
    return data.decode("utf-8", errors="replace")


def load_bootstrap_files(
    data_dir: Path, session_type: SessionType = "main"
) -> list[BootstrapFile]:
    """Load bootstrap files for the given session type. Missing files are skipped."""
    files: list[BootstrapFile] = []
    for name in SESSION_FILTERS[session_type]:
        path = _resolve_path(data_dir, name)
        if path is None:
            continue
        mtime = path.stat().st_mtime
        cache_key = (data_dir, name)
        cached = _cache.get(cache_key)
        if cached and cached[0] == mtime and cached[1].path == path:
            files.append(cached[1])
            continue
        bf = BootstrapFile(name=name, path=path, content=_read(path))
        _cache[cache_key] = (mtime, bf)
        files.append(bf)
    return files


def render_bootstrap_section(bf: BootstrapFile) -> str:
    """Render a single bootstrap file as a prompt section."""
    return f"## {bf.name}\n\n{bf.content.rstrip()}"
