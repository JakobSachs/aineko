"""Tests for tool output truncation layer."""

from hypothesis import given, settings
from hypothesis import strategies as st

from aineko.tools.truncate import truncate_output, MAX_LINES, MAX_BYTES

_nasty_text = st.text(
    alphabet=st.characters(categories=("L", "M", "N", "P", "S", "Z", "C")),
    min_size=0,
    max_size=500,
)


# --- Property-based tests ---


class TestTruncateOutputProperties:
    @given(text=_nasty_text)
    @settings(max_examples=50)
    def test_never_crashes(self, text: str):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with patch("aineko.tools.truncate.TRUNCATION_DIR", Path(tempfile.mkdtemp())):
            truncate_output(text)

    @given(text=st.text(min_size=0, max_size=100))
    def test_short_text_not_truncated(self, text: str):
        result, meta = truncate_output(text)
        if text:
            assert meta["truncated"] is False
            assert result == text

    @given(text=_nasty_text)
    @settings(max_examples=30)
    def test_metadata_always_has_truncated_key(self, text: str):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with patch("aineko.tools.truncate.TRUNCATION_DIR", Path(tempfile.mkdtemp())):
            _, meta = truncate_output(text)
            assert "truncated" in meta

    @given(
        line=st.text(
            alphabet=st.characters(categories=("L", "N")), min_size=1, max_size=20
        ),
        count=st.integers(min_value=MAX_LINES + 1, max_value=MAX_LINES + 50),
    )
    @settings(max_examples=10)
    def test_line_count_respected(self, line: str, count: int):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with patch("aineko.tools.truncate.TRUNCATION_DIR", Path(tempfile.mkdtemp())):
            text = (line + "\n") * count
            result, meta = truncate_output(text)
            assert meta["truncated"] is True
            preview = result.split("\n\n... ")[0]
            assert preview.count("\n") <= MAX_LINES


# --- Example-based tests ---


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
