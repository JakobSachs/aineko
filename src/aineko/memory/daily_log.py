"""File-based daily memory log.

Append-only markdown files at ``<memory_dir>/YYYY-MM-DD.md``. Replaces the
ChromaDB-backed semantic store with human-readable, git-diffable logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aineko.time import now_local


@dataclass
class LogMatch:
    path: str
    line: int
    text: str


def _today_path(memory_dir: Path) -> Path:
    return memory_dir / f"{now_local().strftime('%Y-%m-%d')}.md"


def append_to_today(memory_dir: Path, text: str) -> Path:
    """Append ``text`` as a timestamped entry to today's log. Creates the file
    and parent directory if needed. Returns the file path."""
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = _today_path(memory_dir)
    ts = now_local().strftime("%H:%M:%S")
    entry = f"\n## {ts}\n\n{text.strip()}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)
    return path


def search_logs(
    memory_dir: Path,
    query: str,
    limit: int = 20,
) -> list[LogMatch]:
    """Case-insensitive substring search across all ``memory_dir/*.md`` files.
    Returns up to ``limit`` matches with file path, 1-indexed line, and text."""
    if not memory_dir.exists():
        return []
    needle = query.lower()
    out: list[LogMatch] = []
    for path in sorted(memory_dir.glob("*.md")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, start=1):
            if needle in line.lower():
                out.append(LogMatch(path=str(path), line=i, text=line))
                if len(out) >= limit:
                    return out
    return out


def read_log(
    path: Path,
    from_line: int = 1,
    lines: int | None = None,
) -> str:
    """Read a slice of a log file. ``from_line`` is 1-indexed. Returns empty
    string if the file does not exist."""
    if not path.exists():
        return ""
    text_lines = path.read_text(encoding="utf-8").splitlines()
    start = max(0, from_line - 1)
    end = len(text_lines) if lines is None else start + lines
    return "\n".join(text_lines[start:end])
