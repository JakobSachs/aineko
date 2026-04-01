"""Tests for tool output truncation layer."""

import pytest

from aineko.tools.truncate import truncate_output, MAX_LINES, MAX_BYTES


def test_short_output_unchanged():
    """Output under limits passes through unchanged."""
    text = "line1\nline2\nline3"
    result, meta = truncate_output(text)
    assert result == text
    assert meta["truncated"] is False
    assert "output_path" not in meta


def test_max_lines_constant():
    assert MAX_LINES == 2000


def test_max_bytes_constant():
    assert MAX_BYTES == 50 * 1024


def test_truncate_by_line_count(tmp_path, monkeypatch):
    """Output exceeding MAX_LINES is truncated and saved to file."""
    import aineko.tools.truncate as mod

    monkeypatch.setattr(mod, "TRUNCATION_DIR", tmp_path)
    monkeypatch.setattr(mod, "MAX_LINES", 10)

    lines = [f"line {i}" for i in range(50)]
    text = "\n".join(lines)
    result, meta = truncate_output(text)

    assert meta["truncated"] is True
    assert "output_path" in meta
    # The saved file should contain the full output
    saved = open(meta["output_path"]).read()
    assert saved == text
    # The result should contain the preview (first 10 lines)
    assert "line 0" in result
    assert "line 9" in result
    # Should mention truncation
    assert "truncated" in result.lower() or "lines" in result.lower()
    # Should hint to use read_file
    assert "read_file" in result


def test_truncate_by_byte_size(tmp_path, monkeypatch):
    """Output exceeding MAX_BYTES is truncated."""
    import aineko.tools.truncate as mod

    monkeypatch.setattr(mod, "TRUNCATION_DIR", tmp_path)
    monkeypatch.setattr(mod, "MAX_BYTES", 100)

    text = "x" * 500
    result, meta = truncate_output(text)

    assert meta["truncated"] is True
    assert "output_path" in meta
    assert len(result) < len(text)


def test_truncation_file_contains_full_output(tmp_path, monkeypatch):
    """The saved file always has the complete original output."""
    import aineko.tools.truncate as mod

    monkeypatch.setattr(mod, "TRUNCATION_DIR", tmp_path)
    monkeypatch.setattr(mod, "MAX_LINES", 5)

    lines = [f"line {i}" for i in range(20)]
    text = "\n".join(lines)
    _, meta = truncate_output(text)

    saved = open(meta["output_path"]).read()
    assert saved == text


def test_empty_output():
    """Empty string passes through."""
    result, meta = truncate_output("")
    assert result == ""
    assert meta["truncated"] is False
