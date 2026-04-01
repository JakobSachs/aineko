"""Tests for grep tool."""

import pytest

from aineko.tools.grep import grep_files


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
