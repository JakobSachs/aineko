"""Bash execution tool."""

import asyncio
import os
import signal

from aineko.tools.registry import ToolDef

MAX_OUTPUT = 10_000  # chars


async def run_bash(command: str, timeout: int = 60, description: str = "") -> str:
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd="/data",
            start_new_session=True,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode(errors="replace")
        if len(output) > MAX_OUTPUT:
            output = (
                output[:MAX_OUTPUT] + f"\n... (truncated, {len(output)} total chars)"
            )
        exit_info = f"\n[exit code: {proc.returncode}]"
        return output + exit_info
    except asyncio.TimeoutError:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            # Drain pipes so the subprocess transport closes cleanly instead
            # of being finalized later via __del__ on a possibly-closed loop.
            try:
                await asyncio.wait_for(proc.communicate(), timeout=5)
            except Exception:
                pass
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"
    finally:
        # Explicitly close the subprocess transport before returning so its
        # resources are released while the event loop is still running. This
        # prevents BaseSubprocessTransport.__del__ from firing later on a
        # closed loop (seen with Hypothesis property tests that create a new
        # loop per example). The isinstance guard is so unit tests that pass
        # an AsyncMock aren't affected.
        if isinstance(proc, asyncio.subprocess.Process):
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                transport.close()


bash_tool = ToolDef(
    name="bash",
    description=(
        "Execute a shell command in a persistent working directory (/data).\n\n"
        "IMPORTANT: Do NOT use this for file operations — use specialized tools instead:\n"
        "- Reading files: use read_file (not cat/head/tail)\n"
        "- Editing files: use edit_file (not sed/awk)\n"
        "- Writing files: use write_file (not echo/cat heredoc)\n"
        "- Finding files: use glob (not find/ls)\n"
        "- Searching content: use grep (not grep/rg)\n\n"
        "Use this tool for: git, package managers, builds, system commands, "
        "and other terminal operations that require shell execution."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run"},
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 60)",
                "default": 60,
            },
            "description": {
                "type": "string",
                "description": "Clear 5-10 word description of what the command does",
            },
        },
        "required": ["command"],
    },
    handler=run_bash,
)
