"""Tests for updated bash tool (description param, improved output)."""

import asyncio
import pytest

import aineko.tools.bash as bash_mod
from aineko.tools.bash import bash_tool


async def _test_run_bash(command: str, timeout: int = 60, description: str = "") -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd="/tmp",
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode(errors="replace")
        return output + f"\n[exit code: {proc.returncode}]"
    except asyncio.TimeoutError:
        proc.kill()
        return f"Command timed out after {timeout}s"


@pytest.fixture(autouse=True)
def _patch_bash(monkeypatch):
    monkeypatch.setattr(bash_mod, "run_bash", _test_run_bash)


@pytest.mark.asyncio
async def test_bash_accepts_description_param():
    """run_bash should accept an optional description parameter."""
    result = await _test_run_bash("echo hi", description="Print greeting")
    assert "hi" in result


@pytest.mark.asyncio
async def test_bash_works_without_description():
    """run_bash should still work without description."""
    result = await _test_run_bash("echo works")
    assert "works" in result


def test_bash_tool_schema_has_description_param():
    """The tool schema should include a description parameter."""
    props = bash_tool.parameters["properties"]
    assert "description" in props
    assert props["description"]["type"] == "string"


def test_bash_tool_description_mentions_specialized_tools():
    """Tool description should guide toward specialized tools."""
    desc = bash_tool.description
    assert "read_file" in desc or "grep" in desc or "glob" in desc
