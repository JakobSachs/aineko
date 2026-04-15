"""Grep content search tool — find files containing patterns via ripgrep."""

import asyncio
import os
from pathlib import Path

from aineko.tools.registry import ToolDef

DATA_ROOT = Path("/data")
MAX_RESULTS = 100
MAX_LINE_LENGTH = 2000


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (DATA_ROOT / path).resolve()


async def grep_files(pattern: str, path: str = "", include: str = "") -> str:
    """Search file contents using ripgrep."""
    proc: asyncio.subprocess.Process | None = None
    try:
        search_dir = _resolve_path(path) if path else DATA_ROOT

        cmd = ["rg", "-nH", "--no-messages", "--field-match-separator=|", pattern]
        if include:
            cmd.extend(["--glob", include])
        cmd.append(str(search_dir))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode == 1:
            return "No files found"
        if proc.returncode == 2:
            err = stderr.decode(errors="replace").strip()
            return f"Grep error: {err}"

        output = stdout.decode(errors="replace")
        lines = output.strip().splitlines()

        if not lines:
            return "No files found"

        # Parse rg output: file:line:content or file|line|content
        # Group by file
        by_file: dict[str, list[tuple[int, str]]] = {}
        total_matches = 0
        for line in lines:
            # rg -nH format: path:linenum:content (or with field-match-separator: path|linenum|content)
            parts = line.split("|", 2)
            if len(parts) < 3:
                # Fallback to colon-separated
                parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            filepath, linenum_str, content = parts
            try:
                linenum = int(linenum_str)
            except ValueError:
                continue
            if len(content) > MAX_LINE_LENGTH:
                content = content[:MAX_LINE_LENGTH] + "..."
            if filepath not in by_file:
                by_file[filepath] = []
            by_file[filepath].append((linenum, content))
            total_matches += 1

        if not by_file:
            return "No files found"

        # Sort files by modification time (newest first)
        sorted_files = sorted(
            by_file.keys(),
            key=lambda f: os.path.getmtime(f) if os.path.exists(f) else 0,
            reverse=True,
        )

        # Build output, cap at MAX_RESULTS matches
        result_lines = [f"Found {total_matches} matches"]
        shown = 0
        truncated = total_matches > MAX_RESULTS
        if truncated:
            result_lines[0] += f" (showing first {MAX_RESULTS})"
        result_lines.append("")

        for filepath in sorted_files:
            if shown >= MAX_RESULTS:
                break
            result_lines.append(f"{filepath}:")
            for linenum, content in by_file[filepath]:
                if shown >= MAX_RESULTS:
                    break
                result_lines.append(f"  Line {linenum}: {content}")
                shown += 1
            result_lines.append("")

        if truncated:
            hidden = total_matches - MAX_RESULTS
            result_lines.append(
                f"(Results truncated: showing {MAX_RESULTS} of {total_matches} "
                f"matches ({hidden} hidden). Use a more specific path or pattern.)"
            )

        return "\n".join(result_lines)
    except asyncio.TimeoutError:
        if proc is not None:
            proc.kill()
            try:
                await proc.communicate()
            except Exception:
                pass
        return "Grep timed out after 30s"
    except Exception as e:
        return f"Error: {e}"
    finally:
        # Explicitly close the subprocess transport so its resources are
        # released while the event loop is still running. Prevents
        # BaseSubprocessTransport.__del__ from firing later on a closed loop
        # (seen with Hypothesis property tests that create a new loop per
        # example). The isinstance guard is so unit tests that pass an
        # AsyncMock aren't affected.
        if isinstance(proc, asyncio.subprocess.Process):
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                transport.close()


grep_tool = ToolDef(
    name="grep",
    description=(
        "Search file contents using regular expressions (powered by ripgrep).\n\n"
        "- Supports full regex syntax (e.g. `log.*Error`, `def \\w+\\(`)\n"
        "- Filter files by pattern with the include parameter (e.g. `*.py`, `*.{ts,tsx}`)\n"
        "- Returns file paths and line numbers with matching content\n"
        "- Use this to find specific content instead of reading entire files\n"
        "- For finding files by name, use the glob tool instead"
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in — relative to /data, or absolute (defaults to /data root)",
            },
            "include": {
                "type": "string",
                "description": "File glob pattern to filter (e.g. '*.py', '*.{ts,tsx}')",
            },
        },
        "required": ["pattern"],
    },
    handler=grep_files,
)
