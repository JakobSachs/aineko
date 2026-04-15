"""Tests for grep tool."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aineko.tools.grep import grep_files

# --- Property-based tests ---


class TestGrepProperties:
    @given(pattern=st.text(min_size=1, max_size=100))
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_never_crashes_on_arbitrary_pattern(self, pattern: str):
        """Any pattern string returns a string, never raises."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "file.txt").write_text("some content\n")
            with patch("aineko.tools.grep.DATA_ROOT", tmp):
                result = await grep_files(pattern)
            assert isinstance(result, str)

    @given(path=st.text(min_size=1, max_size=50))
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_never_crashes_on_arbitrary_path(self, path: str):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with patch("aineko.tools.grep.DATA_ROOT", tmp):
                result = await grep_files("test", path=path)
            assert isinstance(result, str)

    @given(include=st.text(min_size=0, max_size=50))
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_never_crashes_on_arbitrary_include(self, include: str):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "file.txt").write_text("content\n")
            with patch("aineko.tools.grep.DATA_ROOT", tmp):
                result = await grep_files("content", include=include)
            assert isinstance(result, str)


# --- Example-based tests ---


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    import aineko.tools.grep as mod

    monkeypatch.setattr(mod, "DATA_ROOT", tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_grep_finds_pattern(data_dir):
    (data_dir / "hello.py").write_text("import os\nimport sys\nprint('hello')\n")
    result = await grep_files("import")
    assert "hello.py" in result
    assert "Line" in result
    assert "import os" in result


@pytest.mark.asyncio
async def test_grep_shows_line_numbers(data_dir):
    (data_dir / "test.py").write_text("aaa\nbbb\nccc\nbbb\n")
    result = await grep_files("bbb")
    assert "Line 2" in result
    assert "Line 4" in result


@pytest.mark.asyncio
async def test_grep_groups_by_file(data_dir):
    (data_dir / "a.py").write_text("foo\n")
    (data_dir / "b.py").write_text("foo\n")
    result = await grep_files("foo")
    assert "a.py" in result
    assert "b.py" in result


@pytest.mark.asyncio
async def test_grep_with_include_filter(data_dir):
    (data_dir / "code.py").write_text("target\n")
    (data_dir / "docs.md").write_text("target\n")
    result = await grep_files("target", include="*.py")
    assert "code.py" in result
    assert "docs.md" not in result


@pytest.mark.asyncio
async def test_grep_with_path(data_dir):
    sub = data_dir / "src"
    sub.mkdir()
    (sub / "main.py").write_text("target\n")
    (data_dir / "other.py").write_text("target\n")
    result = await grep_files("target", path="src")
    assert "main.py" in result
    assert "other.py" not in result


@pytest.mark.asyncio
async def test_grep_no_results(data_dir):
    (data_dir / "empty.py").write_text("nothing here\n")
    result = await grep_files("nonexistent_pattern_xyz")
    assert "No files found" in result or "No matches" in result


@pytest.mark.asyncio
async def test_grep_regex(data_dir):
    (data_dir / "test.py").write_text("def foo():\n    pass\ndef bar():\n    pass\n")
    result = await grep_files(r"def \w+\(\)")
    assert "foo" in result
    assert "bar" in result


@pytest.mark.asyncio
async def test_grep_reports_match_count(data_dir):
    for i in range(5):
        (data_dir / f"file_{i}.py").write_text(f"match_{i}\n")
    result = await grep_files("match_")
    assert "Found" in result
    assert "5" in result


@pytest.mark.asyncio
async def test_grep_long_lines_truncated(data_dir):
    long_line = "x" * 3000
    (data_dir / "long.py").write_text(f"prefix {long_line} suffix\n")
    result = await grep_files("prefix")
    # The matched line should be truncated
    assert len(result) < 3500


@pytest.mark.asyncio
async def test_grep_max_results_truncation(data_dir, monkeypatch):
    """When matches exceed MAX_RESULTS, output is truncated with a note."""
    import aineko.tools.grep as mod

    monkeypatch.setattr(mod, "MAX_RESULTS", 5)

    for i in range(20):
        (data_dir / f"file_{i:03d}.py").write_text(f"target line {i}\n")

    result = await grep_files("target")
    assert "truncated" in result.lower() or "showing" in result.lower()


@pytest.mark.asyncio
async def test_grep_empty_output_after_parse(data_dir):
    """Lines that don't parse into file:line:content are skipped."""
    # A file whose content won't match rg's output format after grep
    (data_dir / "test.txt").write_text("no match here\n")
    result = await grep_files("zzz_nonexistent_pattern")
    assert "No files found" in result


@pytest.mark.asyncio
async def test_grep_absolute_path(tmp_path):
    """Absolute paths work anywhere in the container."""
    (tmp_path / "src.py").write_text("hello world\n")
    result = await grep_files("hello", path=str(tmp_path))
    assert "hello" in result


# --- Mocked subprocess tests for parsing edge cases ---


def _mock_rg(stdout: bytes, returncode: int = 0):
    """Create a mock subprocess with controlled rg output."""
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, b"")
    proc.returncode = returncode
    return proc


@pytest.mark.asyncio
async def test_grep_empty_stdout_after_success(data_dir):
    """rg returns exit 0 but stdout is empty/whitespace."""
    proc = _mock_rg(b"   \n  ", returncode=0)
    with patch("aineko.tools.grep.asyncio.create_subprocess_exec", return_value=proc):
        result = await grep_files("pattern")
    assert "No files found" in result


@pytest.mark.asyncio
async def test_grep_colon_fallback_parsing(data_dir):
    """Lines without | separator fall back to colon-split."""
    # rg output with colons instead of pipes
    stdout = b"/data/file.py:10:some matching content\n"
    proc = _mock_rg(stdout)
    with patch("aineko.tools.grep.asyncio.create_subprocess_exec", return_value=proc):
        result = await grep_files("matching")
    assert "Line 10" in result
    assert "some matching content" in result


@pytest.mark.asyncio
async def test_grep_unparseable_lines_skipped(data_dir):
    """Lines with <3 parts after both splits are silently skipped."""
    stdout = b"just-a-bare-line\nanother bare line\n"
    proc = _mock_rg(stdout)
    with patch("aineko.tools.grep.asyncio.create_subprocess_exec", return_value=proc):
        result = await grep_files("pattern")
    assert "No files found" in result


@pytest.mark.asyncio
async def test_grep_invalid_linenum_skipped(data_dir):
    """Lines where the line number field isn't an integer are skipped."""
    stdout = b"/data/file.py|notanum|content here\n/data/file.py|5|valid line\n"
    proc = _mock_rg(stdout)
    with patch("aineko.tools.grep.asyncio.create_subprocess_exec", return_value=proc):
        result = await grep_files("pattern")
    assert "valid line" in result
    assert "Found 1" in result


@pytest.mark.asyncio
async def test_grep_all_lines_unparseable(data_dir):
    """If every line fails parsing, by_file is empty."""
    stdout = b"garbage\nmore garbage\nno-separators-at-all\n"
    proc = _mock_rg(stdout)
    with patch("aineko.tools.grep.asyncio.create_subprocess_exec", return_value=proc):
        result = await grep_files("pattern")
    assert "No files found" in result


@pytest.mark.asyncio
async def test_grep_inner_max_results_break(data_dir, monkeypatch):
    """MAX_RESULTS cap fires mid-file when one file has many matches."""
    import aineko.tools.grep as mod

    monkeypatch.setattr(mod, "MAX_RESULTS", 3)

    # One file with 10 matches
    lines = [f"/data/big.py|{i}|match line {i}" for i in range(1, 11)]
    stdout = "\n".join(lines).encode() + b"\n"
    proc = _mock_rg(stdout)
    with patch("aineko.tools.grep.asyncio.create_subprocess_exec", return_value=proc):
        result = await grep_files("match")
    assert "showing" in result.lower() or "truncated" in result.lower()
    # Only 3 "Line N:" entries should appear
    assert result.count("Line ") == 3


@pytest.mark.asyncio
async def test_grep_timeout(data_dir):
    """TimeoutError is caught and returns a message."""
    proc = AsyncMock()
    proc.communicate.side_effect = asyncio.TimeoutError()
    proc.kill = MagicMock()
    with patch("aineko.tools.grep.asyncio.create_subprocess_exec", return_value=proc):
        result = await grep_files("pattern")
    assert "timed out" in result.lower()
    proc.kill.assert_called_once()
