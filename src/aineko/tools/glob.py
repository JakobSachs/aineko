"""Glob file search tool — find files by pattern."""

import os
from pathlib import Path

from aineko.tools.registry import ToolDef

DATA_ROOT = Path("/data")
MAX_RESULTS = 100


def _resolve_path(path: str) -> Path:
    resolved = (DATA_ROOT / path).resolve()
    if not str(resolved).startswith(str(DATA_ROOT)):
        raise ValueError(f"Path escapes data directory: {path}")
    return resolved


async def glob_files(pattern: str, path: str = "") -> str:
    """Find files matching a glob pattern, sorted by modification time."""
    try:
        search_dir = _resolve_path(path) if path else DATA_ROOT

        matches = list(search_dir.glob(pattern))
        # Filter to files only
        matches = [m for m in matches if m.is_file()]
        # Sort by modification time (newest first)
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        if not matches:
            return "No files found"

        truncated = len(matches) > MAX_RESULTS
        shown = matches[:MAX_RESULTS]

        lines = [str(p) for p in shown]
        result = "\n".join(lines)

        if truncated:
            result += (
                f"\n\n(Results truncated: showing first {MAX_RESULTS} of "
                f"{len(matches)} results. Use a more specific path or pattern.)"
            )

        return result
    except Exception as e:
        return f"Error: {e}"


glob_tool = ToolDef(
    name="glob",
    description=(
        "Fast file pattern matching tool.\n\n"
        "- Supports glob patterns like `**/*.py` or `src/**/*.ts`\n"
        "- Returns matching file paths sorted by modification time (newest first)\n"
        "- Use this when you need to find files by name pattern\n"
        "- Call this in parallel with grep when doing broad codebase exploration"
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (e.g. '**/*.py', 'src/**/*.ts')",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in, relative to /data (defaults to /data root)",
            },
        },
        "required": ["pattern"],
    },
    handler=glob_files,
)
