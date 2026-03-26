"""Tests for memory recall tool."""

import pytest

from aineko.tools.memory import _list_topics, _search, _read_note
import aineko.tools.memory as memory_mod


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
