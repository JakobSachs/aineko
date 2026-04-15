"""Tests for file read/write/edit tools."""

import string

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aineko.tools.files import (
    DATA_ROOT,
    _resolve_path,
    read_file,
    write_file,
    edit_file,
)

# --- Property-based tests ---


class TestResolvePathProperties:
    @given(
        name=st.text(
            alphabet=st.sampled_from(list(string.ascii_letters + string.digits + "-_")),
            min_size=1,
            max_size=50,
        )
    )
    def test_simple_names_stay_inside_data(self, name: str):
        resolved = _resolve_path(name)
        assert str(resolved).startswith(str(DATA_ROOT))

    @given(
        depth=st.integers(min_value=1, max_value=20),
        name=st.text(
            alphabet=st.sampled_from(list(string.ascii_letters)),
            min_size=1,
            max_size=10,
        ),
    )
    def test_relative_path_resolves_without_error(self, depth: int, name: str):
        # Traversal is now allowed anywhere in the container
        evil_path = "../" * depth + name
        result = _resolve_path(evil_path)
        assert result.is_absolute()


class TestEditFileProperties:
    @given(
        content=st.text(
            alphabet=st.characters(exclude_characters="\r", exclude_categories=("Cs",)),
            min_size=10,
            max_size=500,
        ),
        old_start=st.integers(min_value=0),
        old_len=st.integers(min_value=1, max_value=20),
        new_text=st.text(
            alphabet=st.characters(exclude_characters="\r", exclude_categories=("Cs",)),
            min_size=0,
            max_size=50,
        ),
    )
    @settings(max_examples=30)
    @pytest.mark.asyncio
    async def test_single_replace_changes_exactly_one(
        self,
        content: str,
        old_start: int,
        old_len: int,
        new_text: str,
    ):
        """When old_text appears exactly once, replacement succeeds."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        old_start = old_start % max(1, len(content))
        old_end = min(old_start + old_len, len(content))
        old_text = content[old_start:old_end]
        assume(len(old_text) > 0)
        assume(content.count(old_text) == 1)
        assume(old_text != new_text)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            test_file = tmp / "test.txt"
            test_file.write_text(content)

            with patch("aineko.tools.files.DATA_ROOT", tmp):
                result = await edit_file("test.txt", old_text, new_text)
            assert "Error" not in result
            assert new_text in test_file.read_text()


# --- Example-based tests ---


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
    assert "1: line1" in result
    assert "3: line3" in result


@pytest.mark.asyncio
async def test_read_file_with_offset(data_dir):
    (data_dir / "lines.txt").write_text("a\nb\nc\nd\ne\n")
    result = await read_file("lines.txt", offset=3)
    assert "3: c" in result
    assert "1: a" not in result


@pytest.mark.asyncio
async def test_read_file_with_limit(data_dir):
    (data_dir / "lines.txt").write_text("a\nb\nc\nd\ne\n")
    result = await read_file("lines.txt", limit=2)
    assert "1: a" in result
    assert "2: b" in result
    assert "3: c" not in result


@pytest.mark.asyncio
async def test_read_file_with_offset_and_limit(data_dir):
    (data_dir / "lines.txt").write_text("a\nb\nc\nd\ne\n")
    result = await read_file("lines.txt", offset=2, limit=2)
    assert "2: b" in result
    assert "3: c" in result
    assert "offset=" in result.lower()


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
    assert "Edited" in result
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
    assert "actual content" in result


@pytest.mark.asyncio
async def test_edit_file_ambiguous_match(data_dir):
    (data_dir / "doc.md").write_text("foo bar\nfoo bar\n")
    result = await edit_file("doc.md", "foo bar", "baz")
    assert "2 times" in result
    assert (data_dir / "doc.md").read_text() == "foo bar\nfoo bar\n"


@pytest.mark.asyncio
async def test_edit_file_multiline(data_dir):
    (data_dir / "doc.md").write_text("start\nold line 1\nold line 2\nend\n")
    result = await edit_file(
        "doc.md", "old line 1\nold line 2", "new line 1\nnew line 2\nnew line 3"
    )
    assert "Edited" in result
    assert (
        data_dir / "doc.md"
    ).read_text() == "start\nnew line 1\nnew line 2\nnew line 3\nend\n"


# --- replace_all ---


@pytest.mark.asyncio
async def test_edit_file_replace_all(data_dir):
    (data_dir / "doc.md").write_text("foo bar\nfoo baz\nfoo qux\n")
    result = await edit_file("doc.md", "foo", "replaced", replace_all=True)
    assert "Edited" in result
    content = (data_dir / "doc.md").read_text()
    assert content == "replaced bar\nreplaced baz\nreplaced qux\n"


# --- output formatting ---


@pytest.mark.asyncio
async def test_read_file_xml_tags(data_dir):
    (data_dir / "hello.py").write_text("code\n")
    result = await read_file("hello.py")
    assert "<path>hello.py</path>" in result
    assert "<type>file</type>" in result
    assert "<content>" in result


@pytest.mark.asyncio
async def test_read_file_long_line_truncated(data_dir):
    (data_dir / "long.txt").write_text("x" * 3000 + "\n")
    result = await read_file("long.txt")
    assert "truncated" in result.lower()


@pytest.mark.asyncio
async def test_read_directory(data_dir):
    sub = data_dir / "project"
    sub.mkdir()
    (sub / "file.py").write_text("code")
    (sub / "subdir").mkdir()
    result = await read_file("project")
    assert "<type>directory</type>" in result
    assert "file.py" in result
    assert "subdir/" in result


@pytest.mark.asyncio
async def test_read_directory_truncated(data_dir):
    """Directory listing with more entries than limit shows truncation message."""
    sub = data_dir / "bigdir"
    sub.mkdir()
    for i in range(50):
        (sub / f"file_{i:03d}.txt").write_text("x")
    result = await read_file("bigdir", limit=5)
    assert "Showing" in result
    assert "50" in result


@pytest.mark.asyncio
async def test_read_file_max_read_cap(data_dir, monkeypatch):
    """File content exceeding MAX_READ is truncated."""
    import aineko.tools.files as mod

    monkeypatch.setattr(mod, "MAX_READ", 200)
    (data_dir / "huge.txt").write_text("x" * 500 + "\n")
    result = await read_file("huge.txt")
    assert "truncated" in result.lower()


# --- path traversal ---


@pytest.mark.asyncio
async def test_read_binary_by_extension(data_dir):
    (data_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    result = await read_file("image.png")
    assert "Binary file" in result
    assert "png" in result


@pytest.mark.asyncio
async def test_read_binary_by_content(data_dir):
    (data_dir / "mystery.dat").write_bytes(bytes(range(256)) * 10)
    result = await read_file("mystery.dat")
    assert "Binary file" in result


@pytest.mark.asyncio
async def test_read_absolute_path(tmp_path):
    """Absolute paths work anywhere in the container."""
    f = tmp_path / "secret.txt"
    f.write_text("hello from absolute path")
    result = await read_file(str(f))
    assert "hello from absolute path" in result


@pytest.mark.asyncio
async def test_write_absolute_path(tmp_path):
    """Writes to absolute paths work."""
    f = tmp_path / "out.txt"
    result = await write_file(str(f), "written via absolute path")
    assert "Error" not in result
    assert f.read_text() == "written via absolute path"


@pytest.mark.asyncio
async def test_edit_absolute_path(tmp_path):
    """Edits to absolute paths work."""
    f = tmp_path / "edit_me.txt"
    f.write_text("old content")
    result = await edit_file(str(f), "old content", "new content")
    assert "Error" not in result
    assert f.read_text() == "new content"
