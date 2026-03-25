"""Skill discovery, loading, and file watching."""

import logging
from dataclasses import dataclass
from pathlib import Path

import frontmatter

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    body: str  # full SKILL.md content (loaded on demand)


class SkillsEngine:
    def __init__(self, skills_dir: Path) -> None:
        self._dir = skills_dir
        self._skills: dict[str, Skill] = {}
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def load_all(self) -> None:
        """Scan skills directory and load all SKILL.md files."""
        self._skills.clear()
        if not self._dir.exists():
            self._dir.mkdir(parents=True, exist_ok=True)
            return

        for skill_md in self._dir.glob("*/SKILL.md"):
            self._load_skill(skill_md)

        self._version += 1
        logger.info("Loaded %d skills (v%d)", len(self._skills), self._version)

    def _load_skill(self, path: Path) -> None:
        try:
            post = frontmatter.load(path)
            name = post.get("name", path.parent.name)
            description = post.get("description", "")
            self._skills[name] = Skill(
                name=name,
                description=description,
                path=path,
                body=post.content,
            )
        except Exception:
            logger.exception("Failed to load skill from %s", path)

    def summaries(self) -> list[dict[str, str]]:
        """Return name+description for system prompt (level 1 — cheap)."""
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def reload(self) -> None:
        """Full reload — called by file watcher on change."""
        self.load_all()

    async def watch(self) -> None:
        """Watch skills directory for changes and hot-reload."""
        from watchfiles import awatch

        logger.info("Watching %s for skill changes", self._dir)
        async for _changes in awatch(self._dir):
            logger.info("Skills directory changed, reloading...")
            self.reload()
