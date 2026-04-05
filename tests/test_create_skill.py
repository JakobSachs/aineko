"""Tests for create_skill tool."""

import string
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aineko.tools.create_skill import create_skill
import aineko.tools.create_skill as mod

# --- Property-based tests ---


class TestCreateSkillProperties:
    @given(name=st.text(min_size=1, max_size=50))
    @settings(max_examples=30)
    @pytest.mark.asyncio
    async def test_never_crashes(self, name: str):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(mod, "SKILLS_DIR", Path(td)):
                result = await create_skill(name, "desc")
                assert isinstance(result, str)

    @given(
        name=st.text(
            alphabet=st.sampled_from(list(string.ascii_letters + " ")),
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=30)
    @pytest.mark.asyncio
    async def test_slug_is_lowercase_no_spaces(self, name: str):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(mod, "SKILLS_DIR", Path(td)):
                await create_skill(name, "desc")
                slug = name.lower().replace(" ", "-")
                skill_dir = Path(td) / slug
                if skill_dir.exists():
                    assert " " not in skill_dir.name
                    assert skill_dir.name == skill_dir.name.lower()


# --- Example-based tests ---


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    d = tmp_path / "skills"
    d.mkdir()
    monkeypatch.setattr(mod, "SKILLS_DIR", d)
    return d


@pytest.mark.asyncio
async def test_creates_skill_directory(skills_dir):
    result = await create_skill("Weather", "Get weather info")
    assert "Created" in result
    assert (skills_dir / "weather").is_dir()
    assert (skills_dir / "weather" / "scripts").is_dir()
    assert (skills_dir / "weather" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_skill_md_content(skills_dir):
    await create_skill("My Tool", "Does things", body="Use it wisely.")
    content = (skills_dir / "my-tool" / "SKILL.md").read_text()
    assert "name: My Tool" in content
    assert "description: Does things" in content
    assert "Use it wisely." in content


@pytest.mark.asyncio
async def test_slug_generation(skills_dir):
    await create_skill("Hello World Tool", "desc")
    assert (skills_dir / "hello-world-tool").is_dir()


@pytest.mark.asyncio
async def test_duplicate_skill_rejected(skills_dir):
    await create_skill("Dup", "first")
    result = await create_skill("Dup", "second")
    assert "already exists" in result


@pytest.mark.asyncio
async def test_empty_body_default(skills_dir):
    await create_skill("Minimal", "bare bones")
    content = (skills_dir / "minimal" / "SKILL.md").read_text()
    assert "name: Minimal" in content
