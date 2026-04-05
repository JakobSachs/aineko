"""Tests for bash tool."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import aineko.tools.bash as bash_mod
from aineko.tools.bash import bash_tool, run_bash


def _fake_proc(stdout: bytes, returncode: int = 0):
    """Create a mock subprocess that returns given stdout."""
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, None)
    proc.returncode = returncode
    return proc


# --- Property-based tests ---


class TestRunBashProperties:
    @given(timeout=st.integers(min_value=1, max_value=300))
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_never_crashes_on_any_timeout(self, timeout: int):
        result = await run_bash("true", timeout=timeout)
        assert isinstance(result, str)

    @given(command=st.text(min_size=0, max_size=100))
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_never_crashes_on_arbitrary_commands(self, command: str):
        result = await run_bash(command, timeout=2)
        assert isinstance(result, str)


# --- Glue logic tests (mock the subprocess, test what run_bash does with it) ---


class TestRunBashGlue:
    @pytest.mark.asyncio
    async def test_exit_code_appended(self):
        proc = _fake_proc(b"hello\n", returncode=0)
        with patch(
            "aineko.tools.bash.asyncio.create_subprocess_shell", return_value=proc
        ):
            result = await run_bash("echo hello")
        assert "[exit code: 0]" in result
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self):
        proc = _fake_proc(b"fail\n", returncode=1)
        with patch(
            "aineko.tools.bash.asyncio.create_subprocess_shell", return_value=proc
        ):
            result = await run_bash("false")
        assert "[exit code: 1]" in result

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        big_output = b"x" * 20_000
        proc = _fake_proc(big_output, returncode=0)
        with patch(
            "aineko.tools.bash.asyncio.create_subprocess_shell", return_value=proc
        ):
            with patch.object(bash_mod, "MAX_OUTPUT", 100):
                result = await run_bash("generate_lots")
        assert "truncated" in result
        assert "20000" in result  # reports total size
        assert len(result) < 20_000

    @pytest.mark.asyncio
    async def test_binary_output_decoded(self):
        binary = bytes(range(256))
        proc = _fake_proc(binary, returncode=0)
        with patch(
            "aineko.tools.bash.asyncio.create_subprocess_shell", return_value=proc
        ):
            result = await run_bash("cat /dev/urandom")
        assert isinstance(result, str)  # didn't crash
        assert "[exit code: 0]" in result

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self):
        proc = AsyncMock()
        proc.communicate.side_effect = asyncio.TimeoutError()
        proc.kill = MagicMock()
        with patch(
            "aineko.tools.bash.asyncio.create_subprocess_shell", return_value=proc
        ):
            result = await run_bash("sleep 999", timeout=1)
        assert "timed out" in result.lower()
        proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_output(self):
        proc = _fake_proc(b"", returncode=0)
        with patch(
            "aineko.tools.bash.asyncio.create_subprocess_shell", return_value=proc
        ):
            result = await run_bash("true")
        assert "[exit code: 0]" in result

    @pytest.mark.asyncio
    async def test_subprocess_creation_error(self):
        with patch(
            "aineko.tools.bash.asyncio.create_subprocess_shell",
            side_effect=OSError("no shell"),
        ):
            result = await run_bash("anything")
        assert "Error" in result
        assert "no shell" in result


# --- Tool schema tests ---


def test_bash_tool_schema_has_description_param():
    props = bash_tool.parameters["properties"]
    assert "description" in props
    assert props["description"]["type"] == "string"


def test_bash_tool_description_mentions_specialized_tools():
    assert "read_file" in bash_tool.description or "grep" in bash_tool.description
