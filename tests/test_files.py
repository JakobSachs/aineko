"""Tests for file read/write/edit tools."""

import pytest

from aineko.tools.files import read_file, write_file, edit_file


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point DATA_ROOT at a temp directory."""
    import aineko.tools.files as mod
    monkeypatch.setattr(mod, "DATA_ROOT", tmp_path)
    return tmp_path


# --- read_file ---

@pytest.mark.asyncio
async def test_read_file(data_dir):
    (data_dir / "hello.txt").write_text("line1\nline2\nline3\n")
    result = await read_file("hello.txt")
    assert "line1" in result
    assert "line3" in result


@pytest.mark.asyncio
async def test_read_file_with_offset(data_dir):
    (data_dir / "lines.txt").write_text("a\nb\nc\nd\ne\n")
    result = await read_file("lines.txt", offset=3)
    assert result.startswith("c\n")
    assert "a\n" not in result


@pytest.mark.asyncio
async def test_read_file_with_limit(data_dir):
    (data_dir / "lines.txt").write_text("a\nb\nc\nd\ne\n")
    result = await read_file("lines.txt", limit=2)
    assert "a\n" in result
    assert "b\n" in result
    assert "c" not in result


@pytest.mark.asyncio
async def test_read_file_with_offset_and_limit(data_dir):
    (data_dir / "lines.txt").write_text("a\nb\nc\nd\ne\n")
    result = await read_file("lines.txt", offset=2, limit=2)
    assert result == "b\nc\n"


@pytest.mark.asyncio
async def test_read_file_not_found(data_dir):
    result = await read_file("nope.txt")
    assert "Error" in result


# --- write_file ---

@pytest.mark.asyncio
async def test_write_file(data_dir):
    result = await write_file("new.txt", "hello world")
    assert "Wrote" in result
    assert (data_dir / "new.txt").read_text() == "hello world"


@pytest.mark.asyncio
async def test_write_file_creates_dirs(data_dir):
    result = await write_file("sub/dir/file.md", "content")
    assert "Wrote" in result
    assert (data_dir / "sub/dir/file.md").read_text() == "content"


# --- edit_file ---

@pytest.mark.asyncio
async def test_edit_file_replaces_text(data_dir):
    (data_dir / "doc.md").write_text("hello world\ngoodbye world\n")
    result = await edit_file("doc.md", "hello world", "hi there")
    assert "replaced" in result.lower() or "Edited" in result
    assert (data_dir / "doc.md").read_text() == "hi there\ngoodbye world\n"


@pytest.mark.asyncio
async def test_edit_file_delete_text(data_dir):
    (data_dir / "doc.md").write_text("keep this\nremove this\nkeep too\n")
    result = await edit_file("doc.md", "remove this\n", "")
    assert "Edited" in result
    assert (data_dir / "doc.md").read_text() == "keep this\nkeep too\n"


@pytest.mark.asyncio
async def test_edit_file_not_found(data_dir):
    result = await edit_file("missing.md", "old", "new")
    assert "not found" in result


@pytest.mark.asyncio
async def test_edit_file_old_text_not_found(data_dir):
    (data_dir / "doc.md").write_text("actual content here\n")
    result = await edit_file("doc.md", "nonexistent text", "new")
    assert "not found" in result.lower()
    # Should show file preview to help the model
    assert "actual content" in result


@pytest.mark.asyncio
async def test_edit_file_ambiguous_match(data_dir):
    (data_dir / "doc.md").write_text("foo bar\nfoo bar\n")
    result = await edit_file("doc.md", "foo bar", "baz")
    assert "2 times" in result
    # File should be unchanged
    assert (data_dir / "doc.md").read_text() == "foo bar\nfoo bar\n"


@pytest.mark.asyncio
async def test_edit_file_multiline(data_dir):
    (data_dir / "doc.md").write_text("start\nold line 1\nold line 2\nend\n")
    result = await edit_file("doc.md", "old line 1\nold line 2", "new line 1\nnew line 2\nnew line 3")
    assert "Edited" in result
    assert (data_dir / "doc.md").read_text() == "start\nnew line 1\nnew line 2\nnew line 3\nend\n"


# --- path traversal ---

@pytest.mark.asyncio
async def test_read_blocks_traversal(data_dir):
    result = await read_file("../../etc/passwd")
    assert "Error" in result


@pytest.mark.asyncio
async def test_write_blocks_traversal(data_dir):
    result = await write_file("../../tmp/evil.txt", "pwned")
    assert "Error" in result


@pytest.mark.asyncio
async def test_edit_blocks_traversal(data_dir):
    result = await edit_file("../../etc/passwd", "root", "hacked")
    assert "Error" in result
