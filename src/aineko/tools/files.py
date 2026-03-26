"""File read/write/edit tools."""

import logging
from pathlib import Path

from aineko.tools.registry import ToolDef

logger = logging.getLogger(__name__)

DATA_ROOT = Path("/data")
MAX_READ = 50_000  # chars


def _resolve_path(path: str) -> Path:
    """Resolve path relative to /data, block traversal."""
    resolved = (DATA_ROOT / path).resolve()
    if not str(resolved).startswith(str(DATA_ROOT)):
        raise ValueError(f"Path escapes data directory: {path}")
    return resolved


async def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """Read a file with optional line offset and limit."""
    try:
        resolved = _resolve_path(path)
        content = resolved.read_text(errors="replace")
        lines = content.splitlines(keepends=True)

        if offset > 0:
            lines = lines[offset - 1:]  # 1-based
        if limit > 0:
            lines = lines[:limit]

        result = "".join(lines)
        if len(result) > MAX_READ:
            result = result[:MAX_READ] + f"\n... (truncated, {len(content)} total chars)"

        logger.info("read file", extra={
            "event": "file_read",
            "tool": "read_file",
            "result_len": len(result),
        })
        return result
    except Exception as e:
        return f"Error: {e}"


async def write_file(path: str, content: str) -> str:
    """Write full content to a file (for new files or full rewrites)."""
    try:
        resolved = _resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        logger.info("wrote file", extra={
            "event": "file_write",
            "tool": "write_file",
            "result_len": len(content),
        })
        return f"Wrote {len(content)} chars to {resolved}"
    except Exception as e:
        return f"Error: {e}"


async def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in a file. Finds old_text and replaces with new_text."""
    try:
        resolved = _resolve_path(path)
        if not resolved.exists():
            return f"Error: file not found: {path}"

        content = resolved.read_text(errors="replace")
        count = content.count(old_text)

        if count == 0:
            # Help the model by showing nearby content
            preview = content[:500] if len(content) > 500 else content
            return f"Error: old_text not found in {path}. File starts with:\n{preview}"
        if count > 1:
            return f"Error: old_text found {count} times in {path}. Be more specific to match exactly once."

        new_content = content.replace(old_text, new_text, 1)
        resolved.write_text(new_content)

        logger.info("edited file", extra={
            "event": "file_edit",
            "tool": "edit_file",
            "result_len": len(new_content),
        })
        return f"Edited {path}: replaced {len(old_text)} chars with {len(new_text)} chars"
    except Exception as e:
        return f"Error: {e}"


read_file_tool = ToolDef(
    name="read_file",
    description="Read a file from /data. Path is relative to /data. Supports line offset and limit for partial reads.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to /data"},
            "offset": {"type": "integer", "description": "Start reading from this line number (1-based, optional)"},
            "limit": {"type": "integer", "description": "Max number of lines to read (optional, 0 = all)"},
        },
        "required": ["path"],
    },
    handler=read_file,
)

write_file_tool = ToolDef(
    name="write_file",
    description="Write content to a file in /data. Creates parent directories. Use for new files or full rewrites. For small edits to existing files, prefer edit_file.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to /data"},
            "content": {"type": "string", "description": "Full file content to write"},
        },
        "required": ["path", "content"],
    },
    handler=write_file,
)

edit_file_tool = ToolDef(
    name="edit_file",
    description="Edit a file by replacing exact text. Finds old_text and replaces it with new_text. The old_text must match exactly once. For new files use write_file instead.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to /data"},
            "old_text": {"type": "string", "description": "Exact text to find (must appear exactly once)"},
            "new_text": {"type": "string", "description": "Replacement text (can be empty to delete)"},
        },
        "required": ["path", "old_text", "new_text"],
    },
    handler=edit_file,
)
