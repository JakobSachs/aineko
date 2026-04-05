"""Tests for memory recall tool."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aineko.tools.memory import _list_topics, _search, _read_note, _memory_recall
import aineko.tools.memory as memory_mod

# --- Property-based tests ---


class TestMemoryRecallProperties:
    @given(
        action=st.text(max_size=20),
        keyword=st.text(max_size=50),
        path=st.text(max_size=50),
    )
    @settings(max_examples=30)
    @pytest.mark.asyncio
    async def test_never_crashes(self, action: str, keyword: str, path: str):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(memory_mod, "MEMORY_DIR", Path(td)):
                result = await _memory_recall(action, keyword=keyword, path=path)
                assert isinstance(result, str)

    @given(
        path=st.text(
            alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyz/.\\-_")),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=30)
    def test_read_note_never_leaks_outside(self, path: str):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Write a secret file outside memory dir
            secret = tmp / "secret.txt"
            secret.write_text("TOP SECRET")
            mem_dir = tmp / "memory"
            mem_dir.mkdir()

            with patch.object(memory_mod, "MEMORY_DIR", mem_dir):
                result = _read_note(path)
                assert "TOP SECRET" not in result


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_mod, "MEMORY_DIR", tmp_path)

    # Set up a memory structure
    (tmp_path / "memory.md").write_text("# memory\n- favorite number: 67\n")
    prefs = tmp_path / "preferences"
    prefs.mkdir()
    (prefs / "style.md").write_text("no emojis\nlowercase\n")
    projects = tmp_path / "projects"
    projects.mkdir()
    (projects / "thesis.md").write_text("gpu performance modeling\nstall analysis\n")

    return tmp_path


def test_list_topics(memory_dir):
    result = _list_topics()
    assert "memory.md" in result
    assert "preferences/" in result
    assert "style.md" in result
    assert "projects/" in result
    assert "thesis.md" in result


def test_list_topics_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_mod, "MEMORY_DIR", tmp_path)
    result = _list_topics()
    assert "No memories" in result


def test_search_found(memory_dir):
    result = _search("gpu")
    assert "thesis.md" in result
    assert "gpu" in result.lower()


def test_search_not_found(memory_dir):
    result = _search("quantum")
    assert "No memories matching" in result


def test_search_case_insensitive(memory_dir):
    result = _search("GPU")
    assert "thesis.md" in result


def test_read_note(memory_dir):
    result = _read_note("preferences/style.md")
    assert "no emojis" in result
    assert "lowercase" in result


def test_read_note_not_found(memory_dir):
    result = _read_note("nonexistent.md")
    assert "not found" in result.lower()


def test_read_note_blocks_traversal(memory_dir):
    result = _read_note("../../etc/passwd")
    # Should not return actual file contents — either "not found", "denied", or "outside"
    assert (
        "not found" in result.lower()
        or "denied" in result.lower()
        or "outside" in result.lower()
    )


def test_list_topics_no_dir(tmp_path, monkeypatch):
    """Non-existent MEMORY_DIR returns empty message."""
    monkeypatch.setattr(memory_mod, "MEMORY_DIR", tmp_path / "nonexistent")
    result = _list_topics()
    assert "empty" in result.lower() or "no" in result.lower()


def test_search_no_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_mod, "MEMORY_DIR", tmp_path / "nonexistent")
    result = _search("anything")
    assert "No memories" in result


def test_read_note_oserror(memory_dir, monkeypatch):
    """Unreadable file returns error string."""
    bad_file = memory_dir / "bad.md"
    bad_file.write_text("content")
    bad_file.chmod(0o000)
    result = _read_note("bad.md")
    assert "error" in result.lower() or "denied" in result.lower()
    bad_file.chmod(0o644)  # restore for cleanup


def test_search_skips_unreadable_files(memory_dir):
    """Unreadable .md files during search are silently skipped."""
    bad_file = memory_dir / "unreadable.md"
    bad_file.write_text("secret keyword")
    bad_file.chmod(0o000)
    result = _search("secret")
    # Should not crash, and should not find the unreadable file
    assert "unreadable.md" not in result
    bad_file.chmod(0o644)


def test_read_note_traversal_via_symlink(tmp_path, monkeypatch):
    """A symlink that points outside MEMORY_DIR is blocked by is_relative_to."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    monkeypatch.setattr(memory_mod, "MEMORY_DIR", mem_dir)

    # Create a secret file outside memory dir
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")

    # Symlink from inside memory dir to outside
    link = mem_dir / "escape.md"
    link.symlink_to(secret)

    result = _read_note("escape.md")
    assert "TOP SECRET" not in result
    assert "denied" in result.lower() or "outside" in result.lower()


# --- _memory_recall dispatcher tests ---


@pytest.mark.asyncio
async def test_memory_recall_list(memory_dir):
    result = await _memory_recall("list")
    assert "memory.md" in result


@pytest.mark.asyncio
async def test_memory_recall_search(memory_dir):
    result = await _memory_recall("search", keyword="gpu")
    assert "thesis.md" in result


@pytest.mark.asyncio
async def test_memory_recall_search_missing_keyword(memory_dir):
    result = await _memory_recall("search")
    assert "required" in result.lower()


@pytest.mark.asyncio
async def test_memory_recall_read(memory_dir):
    result = await _memory_recall("read", path="preferences/style.md")
    assert "no emojis" in result


@pytest.mark.asyncio
async def test_memory_recall_read_missing_path(memory_dir):
    result = await _memory_recall("read")
    assert "required" in result.lower()


@pytest.mark.asyncio
async def test_memory_recall_unknown_action(memory_dir):
    result = await _memory_recall("delete")
    assert "Unknown action" in result
