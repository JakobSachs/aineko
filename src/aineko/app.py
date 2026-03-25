"""FastAPI application with lifespan — starts all services."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from aineko.config import Settings
from aineko.context import trim_messages
from aineko.cron.scheduler import CronScheduler
from aineko.db import create_tables, dispose_engine, get_session, init_engine
from aineko.heartbeat.runner import HeartbeatRunner
from aineko.kimi.client import KimiClient
from aineko.matrix.client import MatrixConnector
from aineko.routes.health import router as health_router
from aineko.schemas.message import IncomingMessage
from aineko.skills.engine import SkillsEngine
from aineko.cron.runner import run_cron_job
from aineko.tools.bash import bash_tool
from aineko.tools.create_skill import create_skill_tool
from aineko.tools.files import read_file_tool, write_file_tool
from aineko.tools.registry import ToolRegistry
from aineko.tools.memory import memory_recall_tool
from aineko.tools.web_search import web_search_tool

logger = logging.getLogger(__name__)


def build_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(bash_tool)
    registry.register(read_file_tool)
    registry.register(write_file_tool)
    registry.register(web_search_tool)
    registry.register(create_skill_tool)
    registry.register(memory_recall_tool)
    return registry


_DEFAULT_SOUL = (
    "You are aineko, a personal AI assistant.\n\n"
    "You can execute bash commands, read/write files, and search the web.\n"
    "You run inside a container; /data is your persistent storage.\n\n"
    "## Tools\n\n"
    "Call tools when you need to take action. "
    "Prefer concrete answers over asking the user to do things themselves.\n"
)


def build_system_prompt(skills: SkillsEngine, soul_path: Path, memory_dir: Path) -> str:
    # Load soul from file, create default if missing
    if soul_path.exists():
        soul = soul_path.read_text()
    else:
        soul = _DEFAULT_SOUL
        soul_path.write_text(soul)

    summaries = skills.summaries()
    if summaries:
        lines = [f"- **{s['name']}**: {s['description']}" for s in summaries]
        soul += "\n\n## Available Skills\n" + "\n".join(lines)

    # Inject memory index if it exists
    memory_index = memory_dir / "memory.md"
    if memory_index.exists():
        soul += "\n\n---\n\n" + memory_index.read_text()

    return soul


async def handle_message(
    msg: IncomingMessage,
    kimi: KimiClient,
    tools: ToolRegistry,
    skills: SkillsEngine,
    matrix: MatrixConnector,
    soul_path: Path,
    memory_dir: Path,
    max_context_tokens: int,
) -> None:
    """Core message handler: load session, run agent, send reply."""
    from sqlalchemy import select

    from aineko.models.message import Message, Role, Session

    async for db in get_session():
        # Get or create session for this room
        result = await db.execute(select(Session).where(Session.room_id == msg.room_id))
        session = result.scalar_one_or_none()
        if session is None:
            session = Session(room_id=msg.room_id)
            db.add(session)
            await db.flush()

        # Save incoming message
        db.add(Message(
            session_id=session.id,
            role=Role.USER,
            content=msg.body,
        ))
        await db.flush()

        # Load conversation history
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session.id)
            .order_by(Message.created_at)
        )
        history = result.scalars().all()

        # Build messages for Kimi
        messages = [{"role": "system", "content": build_system_prompt(skills, soul_path, memory_dir)}]
        for m in history:
            messages.append({"role": m.role.value, "content": m.content})

        # Trim to fit context window
        messages = trim_messages(messages, max_context_tokens)

        # Run agent loop
        response = await kimi.chat_loop(messages, tools)

        # Save assistant response
        db.add(Message(
            session_id=session.id,
            role=Role.ASSISTANT,
            content=response.content,
            token_count=response.usage.get("total_tokens"),
        ))
        await db.commit()

        # Send to Matrix — split on --- separators for natural chat feel
        if response.content:
            parts = [p.strip() for p in response.content.split("\n---\n") if p.strip()]
            for part in parts:
                await matrix.send_message(msg.room_id, part)
                if len(parts) > 1:
                    await asyncio.sleep(0.6)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings: Settings = app.state.settings

    # Database
    init_engine(settings.db_url)
    await create_tables()
    logger.info("Database ready: %s", settings.db_url)

    # Skills
    skills = SkillsEngine(settings.skills_dir)
    skills.load_all()

    # Tools
    tools = build_tools()

    # Kimi
    kimi = KimiClient(settings.kimi)

    # Soul + Memory
    soul_path = settings.data_dir / "soul.md"
    memory_dir = settings.data_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    max_ctx = settings.kimi.max_context_tokens

    # Matrix
    matrix = MatrixConnector(settings.matrix, store_path=settings.data_dir / "crypto_store")
    matrix.on_message(lambda msg: handle_message(msg, kimi, tools, skills, matrix, soul_path, memory_dir, max_ctx))

    # Cron
    cron = CronScheduler()

    # Heartbeat
    heartbeat = HeartbeatRunner(settings.heartbeat, settings.heartbeat_file)

    # Wire cron runner
    sys_prompt = build_system_prompt(skills, soul_path, memory_dir)
    cron.set_runner(
        lambda job: run_cron_job(job, kimi, tools, skills, matrix, sys_prompt)
    )

    # Store on app state for route access
    app.state.skills = skills
    app.state.kimi = kimi
    app.state.matrix = matrix
    app.state.cron = cron
    app.state.heartbeat = heartbeat

    # Start background tasks
    bg_tasks: list[asyncio.Task[None]] = []
    bg_tasks.append(asyncio.create_task(matrix.start(), name="matrix-sync"))
    bg_tasks.append(asyncio.create_task(skills.watch(), name="skills-watcher"))

    if settings.cron.enabled:
        async for db in get_session():
            await cron.load_jobs(db)
        cron.start()

    # Heartbeat periodic task
    if settings.heartbeat.enabled:
        heartbeat_room = settings.heartbeat.room or (
            settings.matrix.room_list[0] if settings.matrix.room_list else ""
        )

        async def heartbeat_loop() -> None:
            import asyncio as _asyncio

            interval = settings.heartbeat.every_minutes * 60
            while True:
                await _asyncio.sleep(interval)
                try:
                    if not heartbeat.should_run():
                        continue

                    tasks_content = heartbeat.get_tasks()
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {
                            "role": "user",
                            "content": (
                                "Heartbeat check-in. Review your tasks and take "
                                "any appropriate action.\n\n" + tasks_content
                            ),
                        },
                    ]
                    response = await kimi.chat_loop(messages, tools)

                    if response.content and heartbeat.should_deliver(response.content):
                        if heartbeat_room:
                            await matrix.send_message(heartbeat_room, response.content)
                except Exception:
                    logger.exception("Heartbeat tick failed")

        bg_tasks.append(asyncio.create_task(heartbeat_loop(), name="heartbeat"))

    logger.info("aineko started")

    yield

    # Shutdown
    logger.info("aineko shutting down...")
    cron.stop()
    await matrix.stop()
    await kimi.close()
    for task in bg_tasks:
        task.cancel()
    await dispose_engine()
    logger.info("aineko stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    app = FastAPI(title="aineko", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(health_router)

    return app
