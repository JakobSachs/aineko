"""File read/write tools."""

from pathlib import Path

from aineko.tools.registry import ToolDef

DATA_ROOT = Path("/data")
MAX_READ = 50_000  # chars


def _resolve_path(path: str) -> Path:
    """Resolve path relative to /data, block traversal."""
    resolved = (DATA_ROOT / path).resolve()
    if not str(resolved).startswith(str(DATA_ROOT)):
        raise ValueError(f"Path escapes data directory: {path}")
    return resolved


async def read_file(path: str) -> str:
    try:
        resolved = _resolve_path(path)
        content = resolved.read_text(errors="replace")
        if len(content) > MAX_READ:
            content = content[:MAX_READ] + f"\n... (truncated, {len(content)} total chars)"
        return content
    except Exception as e:
        return f"Error: {e}"


async def write_file(path: str, content: str) -> str:
    try:
        resolved = _resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f"Wrote {len(content)} chars to {resolved}"
    except Exception as e:
        return f"Error: {e}"


read_file_tool = ToolDef(
    name="read_file",
    description="Read a file from /data. Path is relative to /data.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to /data"},
        },
        "required": ["path"],
    },
    handler=read_file,
)

write_file_tool = ToolDef(
    name="write_file",
    description="Write content to a file in /data. Creates parent directories. Path is relative to /data.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to /data"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    },
    handler=write_file,
)
