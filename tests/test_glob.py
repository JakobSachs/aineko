"""Tests for glob tool."""

import pytest

from aineko.tools.glob import glob_files


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    import aineko.tools.glob as mod

    monkeypatch.setattr(mod, "DATA_ROOT", tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_glob_finds_files(data_dir):
    (data_dir / "foo.py").write_text("code")
    (data_dir / "bar.py").write_text("code")
    (data_dir / "readme.md").write_text("docs")
    result = await glob_files("*.py")
    assert "foo.py" in result
    assert "bar.py" in result
    assert "readme.md" not in result


@pytest.mark.asyncio
async def test_glob_recursive(data_dir):
    sub = data_dir / "src"
    sub.mkdir()
    (sub / "main.py").write_text("code")
    (data_dir / "setup.py").write_text("code")
    result = await glob_files("**/*.py")
    assert "main.py" in result
    assert "setup.py" in result


@pytest.mark.asyncio
async def test_glob_with_path(data_dir):
    sub = data_dir / "src"
    sub.mkdir()
    (sub / "main.py").write_text("code")
    (data_dir / "other.py").write_text("code")
    result = await glob_files("*.py", path="src")
    assert "main.py" in result
    assert "other.py" not in result


@pytest.mark.asyncio
async def test_glob_no_results(data_dir):
    result = await glob_files("*.xyz")
    assert "No files found" in result


@pytest.mark.asyncio
async def test_glob_truncates_at_100(data_dir):
    """Should cap results at 100 with a truncation note."""
    for i in range(150):
        (data_dir / f"file_{i:03d}.txt").write_text("x")
    result = await glob_files("*.txt")
    # Should mention truncation
    assert "100" in result
    # The non-note lines should be <= 100
    file_lines = [
        line
        for line in result.strip().splitlines()
        if not line.startswith("(") and line.strip()
    ]
    assert len(file_lines) <= 100


@pytest.mark.asyncio
async def test_glob_returns_absolute_paths(data_dir):
    (data_dir / "test.py").write_text("code")
    result = await glob_files("*.py")
    # Paths should be absolute
    for line in result.strip().splitlines():
        if line and not line.startswith("("):
            assert line.startswith("/")


@pytest.mark.asyncio
async def test_glob_absolute_path(tmp_path):
    """Absolute paths work anywhere in the container."""
    (tmp_path / "a.py").write_text("")
    result = await glob_files("*.py", path=str(tmp_path))
    assert str(tmp_path / "a.py") in result


@pytest.mark.asyncio
async def test_glob_invalid_pattern(data_dir):
    """Invalid glob pattern returns error, not exception."""
    result = await glob_files("[invalid")
    assert isinstance(result, str)
