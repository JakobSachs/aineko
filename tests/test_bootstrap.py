"""Tests for workspace bootstrap file loader."""

from __future__ import annotations

import pytest

from aineko import bootstrap
from aineko.bootstrap import (
    BOOTSTRAP_FILES,
    MAX_FILE_BYTES,
    load_bootstrap_files,
    render_bootstrap_section,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    bootstrap._cache.clear()
    bootstrap._deprecation_warned.clear()
    yield
    bootstrap._cache.clear()
    bootstrap._deprecation_warned.clear()


def test_load_empty_dir_returns_nothing(tmp_path):
    assert load_bootstrap_files(tmp_path) == []


def test_load_finds_known_files(tmp_path):
    (tmp_path / "SOUL.md").write_text("soul content")
    (tmp_path / "AGENTS.md").write_text("agents content")
    files = load_bootstrap_files(tmp_path)
    names = [f.name for f in files]
    assert names == ["SOUL.md", "AGENTS.md"]
    assert files[0].content == "soul content"


def test_load_skips_unknown_files(tmp_path):
    (tmp_path / "random.md").write_text("nope")
    (tmp_path / "SOUL.md").write_text("yes")
    files = load_bootstrap_files(tmp_path)
    assert [f.name for f in files] == ["SOUL.md"]


def test_load_preserves_declared_order(tmp_path):
    for name in BOOTSTRAP_FILES:
        (tmp_path / name).write_text(name)
    files = load_bootstrap_files(tmp_path)
    assert [f.name for f in files] == list(BOOTSTRAP_FILES)


def test_legacy_soul_fallback(tmp_path):
    (tmp_path / "soul.md").write_text("legacy soul")
    files = load_bootstrap_files(tmp_path)
    assert len(files) == 1
    assert files[0].name == "SOUL.md"
    assert files[0].content == "legacy soul"


def test_legacy_memory_fallback(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.md").write_text("legacy index")
    files = load_bootstrap_files(tmp_path)
    assert len(files) == 1
    assert files[0].name == "MEMORY.md"
    assert files[0].content == "legacy index"


def test_canonical_wins_over_legacy(tmp_path):
    (tmp_path / "soul.md").write_text("legacy")
    (tmp_path / "SOUL.md").write_text("canonical")
    files = load_bootstrap_files(tmp_path)
    assert files[0].content == "canonical"


def test_oversize_file_truncated(tmp_path, caplog):
    big = "x" * (MAX_FILE_BYTES + 100)
    (tmp_path / "SOUL.md").write_text(big)
    with caplog.at_level("WARNING"):
        files = load_bootstrap_files(tmp_path)
    assert len(files[0].content) == MAX_FILE_BYTES
    assert any("exceeds" in r.message for r in caplog.records)


def test_cache_reuses_on_unchanged_mtime(tmp_path):
    p = tmp_path / "SOUL.md"
    p.write_text("v1")
    first = load_bootstrap_files(tmp_path)[0]
    second = load_bootstrap_files(tmp_path)[0]
    assert first is second


def test_cache_invalidates_on_mtime_change(tmp_path):
    import os

    p = tmp_path / "SOUL.md"
    p.write_text("v1")
    first = load_bootstrap_files(tmp_path)[0]
    p.write_text("v2")
    os.utime(p, (first.path.stat().st_atime, first.path.stat().st_mtime + 10))
    second = load_bootstrap_files(tmp_path)[0]
    assert second.content == "v2"
    assert first is not second


def test_render_section_format(tmp_path):
    (tmp_path / "SOUL.md").write_text("hello\n")
    files = load_bootstrap_files(tmp_path)
    rendered = render_bootstrap_section(files[0])
    assert rendered == "## SOUL.md\n\nhello"
