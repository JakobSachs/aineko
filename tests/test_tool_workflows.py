"""Workflow tests — chaining tools the way the LLM actually uses them.

Tests that glob → read → edit → read roundtrips work, that grep results
point to real lines, and that write → read is lossless. Property-tested
where the content is arbitrary.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    rule,
    invariant,
)

import aineko.tools.files as files_mod
import aineko.tools.glob as glob_mod
import aineko.tools.grep as grep_mod
from aineko.tools.files import read_file, write_file, edit_file
from aineko.tools.glob import glob_files
from aineko.tools.grep import grep_files

# Text that roundtrips cleanly through write_text/read_text — printable chars + newlines.
_file_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "M", "N", "P", "S"),
        whitelist_characters=" \t\n",
    ),
    min_size=1,
    max_size=1000,
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A shared /data root for all tool modules."""
    monkeypatch.setattr(files_mod, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(glob_mod, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(grep_mod, "DATA_ROOT", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Scenario: write → read roundtrip
# ---------------------------------------------------------------------------


class TestWriteReadRoundtrip:
    @given(content=_file_text)
    @settings(max_examples=30)
    @pytest.mark.asyncio
    async def test_write_then_read_preserves_content(self, content: str):
        """Any writable content survives a write → read roundtrip."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            with patch.object(files_mod, "DATA_ROOT", Path(td)):
                await write_file("roundtrip.txt", content)
                result = await read_file("roundtrip.txt")

            for line in content.splitlines():
                preview = line[:2000]
                if preview.strip():
                    assert preview in result


# ---------------------------------------------------------------------------
# Scenario: write → glob → read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_glob_read(workspace):
    """Write files, find them with glob, read the found file."""
    await write_file("src/app.py", "def main():\n    print('hello')\n")
    await write_file("src/utils.py", "def helper():\n    pass\n")
    await write_file("README.md", "# Project\n")

    # Glob for Python files
    found = await glob_files("**/*.py")
    assert "app.py" in found
    assert "utils.py" in found
    assert "README.md" not in found

    # Extract a path from glob output and read it
    glob_lines = [line for line in found.strip().splitlines() if line.startswith("/")]
    assert len(glob_lines) >= 1

    # Read using path relative to DATA_ROOT
    rel_path = glob_lines[0].replace(str(workspace) + "/", "")
    content = await read_file(rel_path)
    assert "def " in content


# ---------------------------------------------------------------------------
# Scenario: write → grep → read the matching file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_grep_read(workspace):
    """Write files, grep for a pattern, verify the line numbers are correct."""
    source = "import os\nimport sys\n\ndef process():\n    return os.getcwd()\n"
    await write_file("worker.py", source)

    # Grep for 'import'
    grep_result = await grep_files("import")
    assert "worker.py" in grep_result
    assert "Line 1" in grep_result
    assert "Line 2" in grep_result

    # Read the file and verify grep's line numbers match
    read_result = await read_file("worker.py")
    assert "1: import os" in read_result
    assert "2: import sys" in read_result


# ---------------------------------------------------------------------------
# Scenario: write → read → edit → read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_read_edit_read(workspace):
    """Full edit workflow: write, read to find text, edit, verify."""
    original = "name = 'old_value'\ncount = 42\n"
    await write_file("config.py", original)

    # Read to see current state
    read1 = await read_file("config.py")
    assert "old_value" in read1

    # Edit — use the raw text, NOT the line-numbered version
    edit_result = await edit_file(
        "config.py", "name = 'old_value'", "name = 'new_value'"
    )
    assert "Error" not in edit_result

    # Read again to verify
    read2 = await read_file("config.py")
    assert "new_value" in read2
    assert "old_value" not in read2
    # Unchanged line is preserved
    assert "count = 42" in read2


class TestWriteEditRoundtrip:
    @given(
        original=_file_text.filter(lambda s: len(s) >= 20),
        replacement=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "M", "N", "P", "S"),
                whitelist_characters=" \t\n",
            ),
            min_size=1,
            max_size=50,
        ),
        slice_start=st.integers(min_value=0),
        slice_len=st.integers(min_value=1, max_value=15),
    )
    @settings(max_examples=25)
    @pytest.mark.asyncio
    async def test_edit_preserves_surrounding_content(
        self,
        original,
        replacement,
        slice_start,
        slice_len,
    ):
        """Editing a unique substring preserves everything else."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        slice_start = slice_start % max(1, len(original))
        slice_end = min(slice_start + slice_len, len(original))
        target = original[slice_start:slice_end]
        assume(len(target) > 0)
        assume(original.count(target) == 1)
        assume(target != replacement)

        with tempfile.TemporaryDirectory() as td:
            with patch.object(files_mod, "DATA_ROOT", Path(td)):
                await write_file("test.txt", original)
                result = await edit_file("test.txt", target, replacement)
                assume("Error" not in result)

                read_result = await read_file("test.txt")

            # Verify each surviving line appears in read output
            # (read_file adds "N: " prefixes, so check line-by-line)
            # Use str.replace to match what edit_file actually does
            expected = original.replace(target, replacement, 1)
            for line in expected.splitlines():
                trimmed = line[:2000]  # read_file truncates long lines
                if trimmed.strip():
                    assert trimmed in read_result


# ---------------------------------------------------------------------------
# Scenario: grep → edit (find-and-replace driven by search)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grep_then_edit(workspace):
    """Use grep to find something, then edit it out."""
    await write_file("app.py", "# TODO: remove this\ndef main():\n    pass\n")

    # Find the TODO
    grep_result = await grep_files("TODO")
    assert "app.py" in grep_result
    assert "remove this" in grep_result

    # Edit it out
    edit_result = await edit_file("app.py", "# TODO: remove this\n", "")
    assert "Error" not in edit_result

    # Verify it's gone
    grep_after = await grep_files("TODO")
    assert "No files found" in grep_after


# ---------------------------------------------------------------------------
# Scenario: multi-file workflow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_file_rename_workflow(workspace):
    """Rename a function across multiple files using grep → edit."""
    await write_file("lib.py", "def old_name():\n    return 42\n")
    await write_file("main.py", "from lib import old_name\nresult = old_name()\n")

    # Find all references
    grep_result = await grep_files("old_name")
    assert "lib.py" in grep_result
    assert "main.py" in grep_result

    # Rename in both files
    for fname in ["lib.py", "main.py"]:
        result = await edit_file(fname, "old_name", "new_name", replace_all=True)
        assert "Error" not in result

    # Verify old name is gone everywhere
    grep_after = await grep_files("old_name")
    assert "No files found" in grep_after

    # Verify new name is present
    grep_new = await grep_files("new_name")
    assert "lib.py" in grep_new
    assert "main.py" in grep_new


# ---------------------------------------------------------------------------
# Scenario: glob → grep (narrow search to matching files)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glob_then_grep(workspace):
    """Use glob to find files, then grep within a subdirectory."""
    (workspace / "src").mkdir()
    await write_file("src/server.py", "def handle_request():\n    pass\n")
    await write_file("src/client.py", "def send_request():\n    pass\n")
    await write_file("docs/api.md", "# request handling\n")

    # Glob to see what's in src/
    found = await glob_files("*.py", path="src")
    assert "server.py" in found
    assert "client.py" in found

    # Grep only in src/
    grep_result = await grep_files("request", path="src")
    assert "server.py" in grep_result
    assert "client.py" in grep_result
    assert "api.md" not in grep_result


# ---------------------------------------------------------------------------
# Stateful: hypothesis generates arbitrary tool sequences
# ---------------------------------------------------------------------------


# Filenames the state machine can use (small fixed set keeps things focused)
_filenames = st.sampled_from(["a.txt", "b.txt", "c.py", "sub/d.txt"])

# Content strategy — printable + newlines, kept short
_short_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P"),
        whitelist_characters=" \n",
    ),
    min_size=1,
    max_size=200,
)


class ToolWorkflowMachine(RuleBasedStateMachine):
    """Model-based test: run arbitrary tool sequences, check invariants.

    The 'model' is a plain dict tracking what each file should contain.
    After every step, we verify the real tools agree with the model.
    """

    def __init__(self):
        super().__init__()
        self._td = tempfile.TemporaryDirectory()
        self._root = Path(self._td.name)
        self._patches = [
            patch.object(files_mod, "DATA_ROOT", self._root),
            patch.object(glob_mod, "DATA_ROOT", self._root),
            patch.object(grep_mod, "DATA_ROOT", self._root),
        ]
        for p in self._patches:
            p.start()
        # Model: path → expected content
        self.files: dict[str, str] = {}

    def teardown(self):
        for p in self._patches:
            p.stop()
        self._td.cleanup()

    def _run(self, coro):
        """Run an async tool call synchronously."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @rule(name=_filenames, content=_short_text)
    def write(self, name, content):
        """Write a file and update the model."""
        result = self._run(write_file(name, content))
        assert "Error" not in result
        self.files[name] = content

    @rule(name=_filenames)
    def read(self, name):
        """Read a file and check it matches the model."""
        result = self._run(read_file(name))
        if name in self.files:
            # Every line of the model content should appear in the read output
            for line in self.files[name].splitlines():
                stripped = line[:2000]
                if stripped.strip():
                    assert stripped in result
        else:
            assert "Error" in result

    @rule(name=_filenames, old=_short_text, new=_short_text)
    def edit(self, name, old, new):
        """Try an edit and update the model if it succeeds."""
        if name not in self.files:
            return
        content = self.files[name]
        if old not in content or content.count(old) != 1 or old == new:
            return  # skip edits that would fail
        result = self._run(edit_file(name, old, new))
        if "Error" not in result:
            self.files[name] = content.replace(old, new, 1)

    @rule()
    def glob_py(self):
        """Glob for *.py and verify known .py files appear."""
        result = self._run(glob_files("**/*.py"))
        for name in self.files:
            if name.endswith(".py"):
                # glob returns absolute paths
                assert name.split("/")[-1] in result

    @invariant()
    def files_exist_on_disk(self):
        """Every file in the model should exist on disk with correct content."""
        for name, expected in self.files.items():
            path = self._root / name
            assert path.exists(), f"Model has {name} but file missing on disk"
            assert path.read_text() == expected


# Cap: 10 steps per run, 20 runs
TestToolStateful = ToolWorkflowMachine.TestCase
TestToolStateful.settings = settings(max_examples=20, stateful_step_count=10)
