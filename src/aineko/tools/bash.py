"""Bash execution tool."""

import asyncio

from aineko.tools.registry import ToolDef

MAX_OUTPUT = 10_000  # chars


async def run_bash(command: str, timeout: int = 60) -> str:
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
            output = output[:MAX_OUTPUT] + f"\n... (truncated, {len(output)} total chars)"
        exit_info = f"\n[exit code: {proc.returncode}]"
        return output + exit_info
    except asyncio.TimeoutError:
        proc.kill()
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


bash_tool = ToolDef(
    name="bash",
    description="Execute a shell command. Working directory is /data.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run"},
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 60)",
                "default": 60,
            },
        },
        "required": ["command"],
    },
    handler=run_bash,
)
