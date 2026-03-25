"""Tool: scaffold a new skill directory with SKILL.md."""

from pathlib import Path

from aineko.tools.registry import ToolDef

DATA_ROOT = Path("/data")
SKILLS_DIR = DATA_ROOT / "skills"

TEMPLATE = """\
---
name: {name}
description: {description}
---

{body}
"""


async def create_skill(name: str, description: str, body: str = "") -> str:
    """Create a new skill directory with SKILL.md."""
    slug = name.lower().replace(" ", "-")
    skill_dir = SKILLS_DIR / slug
    if skill_dir.exists():
        return f"Error: skill directory already exists: {skill_dir}"

    try:
        skill_dir.mkdir(parents=True)
        (skill_dir / "scripts").mkdir()

        content = TEMPLATE.format(name=name, description=description, body=body)
        (skill_dir / "SKILL.md").write_text(content)

        return f"Created skill '{name}' at {skill_dir}"
    except Exception as e:
        return f"Error creating skill: {e}"


create_skill_tool = ToolDef(
    name="create_skill",
    description=(
        "Scaffold a new skill directory under /data/skills/ with a SKILL.md template. "
        "Use this to create new skills that persist across restarts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Human-readable skill name"},
            "description": {
                "type": "string",
                "description": "Short description of what the skill does",
            },
            "body": {
                "type": "string",
                "description": "Markdown body for the SKILL.md (instructions, examples, etc.)",
                "default": "",
            },
        },
        "required": ["name", "description"],
    },
    handler=create_skill,
)
