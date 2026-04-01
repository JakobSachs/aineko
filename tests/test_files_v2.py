"""Tests for updated file tool output formatting (OpenCode-style)."""

import pytest

from aineko.tools.files import read_file, edit_file


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    import aineko.tools.files as mod

    monkeypatch.setattr(mod, "DATA_ROOT", tmp_path)
    return tmp_path


# --- read_file line numbers ---


@pytest.mark.asyncio
async def test_read_file_has_line_numbers(data_dir):
    """Output should have 'N: content' format like OpenCode."""
    (data_dir / "hello.py").write_text("import os\nimport sys\nprint('hi')\n")
    result = await read_file("hello.py")
    assert "1: import os" in result
    assert "2: import sys" in result
    assert "3: print('hi')" in result


@pytest.mark.asyncio
async def test_read_file_xml_tags(data_dir):
    """Output should have <path>, <type>, <content> tags."""
    (data_dir / "hello.py").write_text("code\n")
    result = await read_file("hello.py")
    assert "<path>" in result
    assert "hello.py</path>" in result
    assert "<type>file</type>" in result
    assert "<content>" in result
    assert "</content>" in result


@pytest.mark.asyncio
async def test_read_file_offset_with_line_numbers(data_dir):
    """Offset should show correct line numbers."""
    lines = "\n".join(f"line {i}" for i in range(1, 11)) + "\n"
    (data_dir / "lines.txt").write_text(lines)
    result = await read_file("lines.txt", offset=5)
    assert "5: line 5" in result
    assert "6: line 6" in result
    # Should NOT have lines 1-4
    assert "1: line 1" not in result


@pytest.mark.asyncio
async def test_read_file_shows_continuation_hint(data_dir):
    """When not showing all lines, hint about using offset."""
    lines = "\n".join(f"line {i}" for i in range(1, 301)) + "\n"
    (data_dir / "big.txt").write_text(lines)
    result = await read_file("big.txt", limit=10)
    assert "offset=" in result.lower() or "Use offset" in result


@pytest.mark.asyncio
async def test_read_file_end_of_file_message(data_dir):
    """Small file should say total lines at end."""
    (data_dir / "small.txt").write_text("a\nb\nc\n")
    result = await read_file("small.txt")
    assert "3" in result  # total line count mentioned somewhere


@pytest.mark.asyncio
async def test_read_file_long_line_truncated(data_dir):
    """Lines > 2000 chars should be truncated."""
    long_line = "x" * 3000
    (data_dir / "long.txt").write_text(long_line + "\n")
    result = await read_file("long.txt")
    # Line should be cut and have truncation indicator
    assert "truncated" in result.lower() or len(result) < 3100


@pytest.mark.asyncio
async def test_read_directory_lists_entries(data_dir):
    """Reading a directory should list entries with / for subdirs."""
    sub = data_dir / "project"
    sub.mkdir()
    (sub / "file.py").write_text("code")
    (sub / "subdir").mkdir()
    (sub / "readme.md").write_text("docs")

    result = await read_file("project")
    assert "<type>directory</type>" in result
    assert "file.py" in result
    assert "subdir/" in result
    assert "readme.md" in result


# --- edit_file with replace_all ---


@pytest.mark.asyncio
async def test_edit_file_replace_all(data_dir):
    """replace_all=True should replace all occurrences."""
    (data_dir / "doc.md").write_text("foo bar\nfoo baz\nfoo qux\n")
    result = await edit_file("doc.md", "foo", "replaced", replace_all=True)
    assert "Edited" in result or "replaced" in result.lower()
    content = (data_dir / "doc.md").read_text()
    assert content == "replaced bar\nreplaced baz\nreplaced qux\n"
    assert "foo" not in content


@pytest.mark.asyncio
async def test_edit_file_replace_all_false_still_rejects_ambiguous(data_dir):
    """Without replace_all, multiple matches should still fail."""
    (data_dir / "doc.md").write_text("foo bar\nfoo bar\n")
    result = await edit_file("doc.md", "foo bar", "baz")
    assert "2 times" in result
