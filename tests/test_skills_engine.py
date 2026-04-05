"""Tests for SkillsEngine — discovery, loading, and hot-reload."""

from pathlib import Path

import pytest

from aineko.skills.engine import SkillsEngine


@pytest.fixture
def skills_dir(tmp_path):
    return tmp_path / "skills"


@pytest.fixture
def engine(skills_dir):
    return SkillsEngine(skills_dir)


def _create_skill(
    skills_dir: Path, name: str, description: str = "", body: str = ""
) -> Path:
    """Helper to write a SKILL.md file with frontmatter."""
    skill_path = skills_dir / name
    skill_path.mkdir(parents=True, exist_ok=True)
    md = skill_path / "SKILL.md"
    md.write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}")
    return md


class TestLoadAll:
    def test_creates_dir_if_missing(self, engine, skills_dir):
        assert not skills_dir.exists()
        engine.load_all()
        assert skills_dir.exists()

    def test_loads_skills(self, engine, skills_dir):
        _create_skill(skills_dir, "weather", "Get weather info", "Use the weather API.")
        _create_skill(skills_dir, "deploy", "Deploy the app", "Run deploy script.")

        engine.load_all()

        assert engine.get("weather") is not None
        assert engine.get("deploy") is not None
        assert engine.get("nonexistent") is None

    def test_version_increments(self, engine, skills_dir):
        skills_dir.mkdir(parents=True)
        assert engine.version == 0
        engine.load_all()
        assert engine.version == 1
        engine.load_all()
        assert engine.version == 2

    def test_empty_dir(self, engine, skills_dir):
        skills_dir.mkdir(parents=True)
        engine.load_all()
        assert engine.summaries() == []

    def test_clears_old_skills_on_reload(self, engine, skills_dir):
        _create_skill(skills_dir, "old-skill", "old")
        engine.load_all()
        assert engine.get("old-skill") is not None

        # Remove old skill, add new one
        import shutil

        shutil.rmtree(skills_dir / "old-skill")
        _create_skill(skills_dir, "new-skill", "new")
        engine.load_all()

        assert engine.get("old-skill") is None
        assert engine.get("new-skill") is not None


class TestLoadSkill:
    def test_parses_frontmatter(self, engine, skills_dir):
        _create_skill(skills_dir, "greet", "Say hello", "Hello world body")
        engine.load_all()

        skill = engine.get("greet")
        assert skill.name == "greet"
        assert skill.description == "Say hello"
        assert skill.body == "Hello world body"

    def test_defaults_name_to_dir(self, engine, skills_dir):
        """If frontmatter has no name, use the directory name."""
        skill_path = skills_dir / "my-tool"
        skill_path.mkdir(parents=True)
        (skill_path / "SKILL.md").write_text("---\ndescription: a tool\n---\nbody")

        engine.load_all()
        skill = engine.get("my-tool")
        assert skill is not None
        assert skill.name == "my-tool"

    def test_malformed_frontmatter_skipped(self, engine, skills_dir):
        skill_path = skills_dir / "broken"
        skill_path.mkdir(parents=True)
        # Write invalid YAML frontmatter
        (skill_path / "SKILL.md").write_text("---\n: invalid: yaml: {{{\n---\nbody")

        engine.load_all()
        # Should not crash, just skip
        assert engine.get("broken") is None


class TestSummaries:
    def test_returns_name_and_description(self, engine, skills_dir):
        _create_skill(skills_dir, "alpha", "First skill")
        _create_skill(skills_dir, "beta", "Second skill")
        engine.load_all()

        summaries = engine.summaries()
        assert len(summaries) == 2
        names = {s["name"] for s in summaries}
        assert names == {"alpha", "beta"}
        for s in summaries:
            assert "name" in s
            assert "description" in s


class TestReload:
    def test_reload_calls_load_all(self, engine, skills_dir):
        skills_dir.mkdir(parents=True)
        engine.load_all()
        v1 = engine.version

        engine.reload()
        assert engine.version == v1 + 1
