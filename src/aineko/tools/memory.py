"""Memory recall tool — browse and search the tiered memory system."""

import logging
from pathlib import Path

from aineko.tools.registry import ToolDef

logger = logging.getLogger(__name__)

MEMORY_DIR = Path("/data/memory")


def _list_topics() -> str:
    """List all topic folders and their files."""
    if not MEMORY_DIR.exists():
        return "Memory directory is empty."

    lines: list[str] = []
    # Top-level files
    for f in sorted(MEMORY_DIR.iterdir()):
        if f.is_file() and f.suffix == ".md":
            lines.append(f"  {f.name}")

    # Topic folders
    for d in sorted(MEMORY_DIR.iterdir()):
        if d.is_dir():
            files = sorted(f.name for f in d.iterdir() if f.is_file())
            lines.append(f"  {d.name}/")
            for fname in files:
                lines.append(f"    {fname}")

    return "\n".join(lines) if lines else "No memories stored yet."


def _search(keyword: str) -> str:
    """Search all memory files for a keyword (case-insensitive)."""
    if not MEMORY_DIR.exists():
        return "No memories found."

    results: list[str] = []
    keyword_lower = keyword.lower()

    for path in sorted(MEMORY_DIR.rglob("*.md")):
        try:
            content = path.read_text()
        except OSError:
            continue

        matches = []
        for i, line in enumerate(content.splitlines(), 1):
            if keyword_lower in line.lower():
                matches.append(f"  L{i}: {line.strip()}")

        if matches:
            rel = path.relative_to(MEMORY_DIR)
            results.append(f"{rel}:")
            results.extend(matches[:5])  # max 5 matches per file

    if not results:
        return f"No memories matching '{keyword}'."
    return "\n".join(results)


def _read_note(path: str) -> str:
    """Read a specific memory file."""
    note_path = MEMORY_DIR / path
    if not note_path.exists():
        return f"File not found: {path}"
    if not note_path.resolve().is_relative_to(MEMORY_DIR.resolve()):
        return "Access denied: path outside memory directory."
    try:
        return note_path.read_text()
    except OSError as e:
        return f"Error reading {path}: {e}"


async def _memory_recall(action: str, keyword: str = "", path: str = "") -> str:
    if action == "list":
        return _list_topics()
    elif action == "search":
        if not keyword:
            return "Error: 'keyword' is required for search."
        return _search(keyword)
    elif action == "read":
        if not path:
            return "Error: 'path' is required for read (relative to /data/memory/)."
        return _read_note(path)
    else:
        return f"Unknown action '{action}'. Use: list, search, or read."


memory_recall_tool = ToolDef(
    name="memory_recall",
    description="Browse and search your persistent memory. Actions: 'list' (show all topics and files), 'search' (find memories by keyword), 'read' (read a specific file).",
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "search", "read"],
                "description": "What to do: list topics, search by keyword, or read a specific file.",
            },
            "keyword": {
                "type": "string",
                "description": "Search term (for action=search).",
            },
            "path": {
                "type": "string",
                "description": "Relative path to a memory file, e.g. 'projects/aineko.md' (for action=read).",
            },
        },
        "required": ["action"],
    },
    handler=_memory_recall,
)
