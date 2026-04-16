"""File read/write/edit tools."""

import logging
from pathlib import Path

from aineko.tools.registry import ToolDef

logger = logging.getLogger(__name__)

DATA_ROOT = Path("/data")
MAX_READ = 50_000  # chars
DEFAULT_LINE_LIMIT = 2000
MAX_LINE_LENGTH = 2000


def _resolve_path(path: str) -> Path:
    """Resolve path. Absolute paths are used as-is; relative paths are relative to /data."""
    p = Path(path)
    return p.resolve() if p.is_absolute() else (DATA_ROOT / path).resolve()


BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".mp3",
    ".mp4",
    ".wav",
    ".flac",
    ".ogg",
    ".avi",
    ".mkv",
    ".mov",
    ".exe",
    ".bin",
    ".so",
    ".dll",
    ".o",
    ".pyc",
    ".whl",
    ".db",
    ".sqlite",
    ".sqlite3",
}


async def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """Read a file or directory with line numbers, OpenCode-style output."""
    try:
        resolved = _resolve_path(path)

        # Directory listing
        if resolved.is_dir():
            entries = sorted(resolved.iterdir())
            entry_lines = []
            for e in entries:
                name = e.name + ("/" if e.is_dir() else "")
                entry_lines.append(name)
            total = len(entry_lines)
            effective_limit = limit if limit > 0 else DEFAULT_LINE_LIMIT
            start = max(0, offset - 1) if offset > 0 else 0
            shown = entry_lines[start : start + effective_limit]
            body = "\n".join(shown)
            if len(shown) < total:
                body += f"\n(Showing {len(shown)} of {total} entries. Use offset to read beyond entry {start + len(shown)})"
            else:
                body += f"\n({total} entries)"
            return f"<path>{path}</path>\n<type>directory</type>\n<entries>\n{body}\n</entries>"

        # Refuse to read binary files as text
        if resolved.suffix.lower() in BINARY_EXTENSIONS:
            size = resolved.stat().st_size
            return f"Binary file: {path} ({size} bytes, type: {resolved.suffix}). Use bash to inspect (e.g. file, hexdump, pdftotext)."

        content = resolved.read_text(errors="replace")

        # Detect binary content (high ratio of non-printable chars)
        if len(content) > 100:
            sample = content[:1000]
            non_printable = sum(
                1 for c in sample if not c.isprintable() and c not in "\n\r\t"
            )
            if non_printable / len(sample) > 0.1:
                return (
                    f"Binary file: {path} ({len(content)} bytes). Use bash to inspect."
                )

        lines = content.splitlines()
        total_lines = len(lines)

        # Apply offset (1-based)
        start_line = max(0, offset - 1) if offset > 0 else 0
        view = lines[start_line:]
        effective_limit = limit if limit > 0 else DEFAULT_LINE_LIMIT
        view = view[:effective_limit]

        # Format with line numbers, truncate long lines
        numbered: list[str] = []
        for i, line in enumerate(view, start=start_line + 1):
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + "... (line truncated to 2000 chars)"
            numbered.append(f"{i}: {line}")

        last_shown = start_line + len(view)
        remaining = total_lines - last_shown

        if remaining > 0:
            footer = f"(Showing lines {start_line + 1}-{last_shown} of {total_lines}. Use offset={last_shown + 1} to continue.)"
        else:
            footer = f"(End of file — total {total_lines} lines)"

        body = "\n".join(numbered) + "\n" + footer

        # Final size cap
        if len(body) > MAX_READ:
            body = body[:MAX_READ] + f"\n... (truncated, {len(content)} total chars)"

        logger.info(
            "read file",
            extra={
                "event": "file_read",
                "tool": "read_file",
                "result_len": len(body),
            },
        )
        return f"<path>{path}</path>\n<type>file</type>\n<content>\n{body}\n</content>"
    except Exception as e:
        return f"Error: {e}"


async def write_file(path: str, content: str) -> str:
    """Write full content to a file (for new files or full rewrites)."""
    try:
        resolved = _resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        logger.info(
            "wrote file",
            extra={
                "event": "file_write",
                "tool": "write_file",
                "result_len": len(content),
            },
        )
        return f"Wrote {len(content)} chars to {resolved}"
    except Exception as e:
        return f"Error: {e}"


async def edit_file(
    path: str, old_text: str, new_text: str, replace_all: bool = False
) -> str:
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
        if count > 1 and not replace_all:
            return f"Error: old_text found {count} times in {path}. Be more specific to match exactly once."

        if replace_all:
            new_content = content.replace(old_text, new_text)
        else:
            new_content = content.replace(old_text, new_text, 1)
        resolved.write_text(new_content)

        logger.info(
            "edited file",
            extra={
                "event": "file_edit",
                "tool": "edit_file",
                "result_len": len(new_content),
            },
        )
        return (
            f"Edited {path}: replaced {len(old_text)} chars with {len(new_text)} chars"
        )
    except Exception as e:
        return f"Error: {e}"


read_file_tool = ToolDef(
    name="read_file",
    description=(
        "Read a file or directory. Relative paths are relative to /data; absolute paths (e.g. /app/src) work anywhere in the container.\n\n"
        "Usage:\n"
        "- By default returns up to 2000 lines from the start of the file.\n"
        "- The offset parameter is the line number to start from (1-indexed).\n"
        "- Contents are returned with each line prefixed by its line number as `N: content`.\n"
        "- For directories, entries are listed one per line with trailing `/` for subdirectories.\n"
        "- Any line longer than 2000 characters is truncated.\n"
        "- Use the grep tool to find specific content instead of reading entire large files.\n"
        "- Use the glob tool to find files by name pattern.\n"
        "- Avoid tiny repeated slices (30 line chunks). Read a larger window instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path — relative to /data, or absolute (e.g. /app/src/aineko/app.py)",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start from (1-indexed)",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to read (default 2000)",
            },
        },
        "required": ["path"],
    },
    handler=read_file,
)

write_file_tool = ToolDef(
    name="write_file",
    description=(
        "Write content to a file. Creates parent directories. "
        "Use for new files or full rewrites. For small edits to existing files, prefer edit_file.\n\n"
        "IMPORTANT: The user cannot see file contents unless you explicitly share them. "
        "After writing a file the user asked to see or use, either:\n"
        "- call send_file to deliver it as an attachment, or\n"
        "- include the content in a send_message / your final response as a code block."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path — relative to /data, or absolute",
            },
            "content": {"type": "string", "description": "Full file content to write"},
        },
        "required": ["path", "content"],
    },
    handler=write_file,
)

edit_file_tool = ToolDef(
    name="edit_file",
    description=(
        "Edit a file by replacing exact text. Finds old_text and replaces with new_text.\n\n"
        "Usage:\n"
        "- You must read the file first before editing.\n"
        "- When using text from read_file output, preserve exact indentation but do NOT include "
        "the line number prefix (the `N: ` part). Everything after the `N: ` is the actual content.\n"
        "- The old_text must match exactly once, unless replace_all is true.\n"
        "- Use replace_all for renaming variables or replacing a string everywhere in the file.\n"
        "- For new files use write_file instead.\n\n"
        "IMPORTANT: The user cannot see file contents unless you explicitly share them. "
        "If the user asked to see the result, use send_file or include the content in your response."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path — relative to /data, or absolute",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to find",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text (can be empty to delete)",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences (default false)",
                "default": False,
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
    handler=edit_file,
)
