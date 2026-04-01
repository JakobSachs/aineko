"""Tool output truncation — saves full output to disk when too large."""

import uuid
from pathlib import Path
from typing import Any

MAX_LINES = 2000
MAX_BYTES = 50 * 1024  # 50 KB
TRUNCATION_DIR = Path("/data/.truncated")


def truncate_output(
    output: str,
) -> tuple[str, dict[str, Any]]:
    """Truncate tool output if it exceeds limits.

    Returns (possibly_truncated_output, metadata_dict).
    If truncated, saves full output to a file and includes the path in metadata.
    """
    if not output:
        return output, {"truncated": False}

    lines = output.splitlines(keepends=True)
    byte_size = len(output.encode("utf-8", errors="replace"))

    if len(lines) <= MAX_LINES and byte_size <= MAX_BYTES:
        return output, {"truncated": False}

    # Save full output to file
    TRUNCATION_DIR.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex[:12]
    output_path = TRUNCATION_DIR / f"{file_id}.txt"
    output_path.write_text(output)

    # Build preview: take lines up to limit, also respect byte budget
    preview_lines: list[str] = []
    preview_bytes = 0
    for line in lines:
        line_bytes = len(line.encode("utf-8", errors="replace"))
        if len(preview_lines) >= MAX_LINES or preview_bytes + line_bytes > MAX_BYTES:
            break
        preview_lines.append(line)
        preview_bytes += line_bytes

    preview = "".join(preview_lines)
    remaining = len(lines) - len(preview_lines)

    truncation_msg = (
        f"\n\n... {remaining} lines truncated ...\n\n"
        f"Full output saved to {output_path}. "
        f"Use read_file with offset/limit to view the rest."
    )

    return preview + truncation_msg, {
        "truncated": True,
        "output_path": str(output_path),
    }
