"""Bash execution tool."""

import asyncio

from aineko.tools.registry import ToolDef

MAX_OUTPUT = 10_000  # chars


async def run_bash(command: str, timeout: int = 60, description: str = "") -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd="/data",
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
        proc.kill()
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


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
