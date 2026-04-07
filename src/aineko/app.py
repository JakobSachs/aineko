"""FastAPI application with lifespan — starts all services."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from aineko.compaction import compact_messages, ingest_before_compaction, should_compact
from aineko.config import Settings
from aineko.context import trim_messages
from aineko.cron.scheduler import CronScheduler
from aineko.db import (
    create_tables,
    dispose_engine,
    get_session,
    init_engine,
)
import aineko.db as _db
from aineko.handler import (
    build_request_tools,
    format_tool_footer,
    handle_command,
    load_conversation,
    persist_response,
)
from aineko.heartbeat.loop import heartbeat_tick_loop
from aineko.heartbeat.runner import HeartbeatRunner
from aineko.kimi.client import ChatResponse, KimiClient
from aineko.matrix.client import MatrixConnector
from aineko.models.message import Message, Role
from aineko.routes.health import router as health_router
from aineko.schemas.message import IncomingMessage
from aineko.skills.engine import SkillsEngine
from aineko.cron.runner import run_cron_job
from aineko.tools.bash import bash_tool
from aineko.tools.create_skill import create_skill_tool
from aineko.tools.files import edit_file_tool, read_file_tool, write_file_tool
from aineko.tools.glob import glob_tool
from aineko.tools.grep import grep_tool
from aineko.tools.registry import ToolRegistry
from aineko.tools.memory import memory_tool
from aineko.tools import calendar as calendar_mod
from aineko.tools import web_search as web_search_mod
from aineko.tools.calendar import read_calendar_tool
from aineko.tools.search_chat import search_chat_tool
from aineko.tools.tool_history import search_tool_history_tool
from aineko.tools.background_task import BackgroundTaskManager
from aineko.tools.web_search import web_search_tool

logger = logging.getLogger(__name__)


def build_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(bash_tool)
    registry.register(read_file_tool)
    registry.register(write_file_tool)
    registry.register(edit_file_tool)
    registry.register(glob_tool)
    registry.register(grep_tool)
    registry.register(web_search_tool)
    registry.register(create_skill_tool)
    registry.register(memory_tool)
    registry.register(search_tool_history_tool)
    registry.register(search_chat_tool)
    registry.register(read_calendar_tool)
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
    bg_tasks: BackgroundTaskManager | None = None,
    compaction_keep_recent: int = 4,
) -> None:
    """Core message handler: load session, run agent, send reply."""
    from sqlalchemy import delete as sa_delete

    assert _db.async_session_factory is not None

    async with _db.async_session_factory() as db:
        if await handle_command(db, msg, matrix):
            return

        sent_messages: list[str] = []
        request_tools: ToolRegistry = build_request_tools(
            tools, matrix, msg.room_id, sent_messages, bg_tasks
        )

        sys_prompt: str = build_system_prompt(skills, soul_path, memory_dir)
        session, user_msg, messages = await load_conversation(db, msg, sys_prompt)

        # Compact if conversation is getting long
        if should_compact(messages, max_context_tokens):
            logger.info("conversation approaching context limit, compacting")
            from sqlalchemy import select

            result = await db.execute(
                select(Message)
                .where(Message.session_id == session.id)
                .order_by(Message.created_at)
            )
            history: list[Message] = list(result.scalars().all())

            # Ingest messages about to be compacted into long-term memory
            ingest_before_compaction(
                messages, session.id, session.room_id, compaction_keep_recent
            )

            messages, summary_text = await compact_messages(
                messages,
                kimi,
                keep_recent=compaction_keep_recent,
            )
            if summary_text:
                if old_msg_ids := [m.id for m in history[:-compaction_keep_recent]]:
                    await db.execute(
                        sa_delete(Message).where(Message.id.in_(old_msg_ids))
                    )
                db.add(
                    Message(
                        session_id=session.id,
                        role=Role.SYSTEM,
                        content=summary_text,
                    )
                )
                await db.flush()

        messages = trim_messages(messages, max_context_tokens)

        response: ChatResponse = await kimi.chat_loop(messages, request_tools)

        if response.content:
            footer: str = format_tool_footer(response.tool_history)
            await matrix.send_message(msg.room_id, response.content + footer)

        await persist_response(db, session, user_msg, response, sent_messages)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings: Settings = app.state.settings

    # Database
    init_engine(settings.db_url)
    await create_tables()
    logger.info("Database ready: %s", settings.db_url)

    # Skills
    skills: SkillsEngine = SkillsEngine(settings.skills_dir)
    skills.load_all()

    # Tools
    tools: ToolRegistry = build_tools()
    web_search_mod.brave_api_key = settings.brave_api_key
    calendar_mod.caldav_url = settings.caldav.url
    calendar_mod.caldav_username = settings.caldav.username
    calendar_mod.caldav_password = settings.caldav.password

    # Kimi
    kimi: KimiClient = KimiClient(settings.kimi)

    # Soul + Memory
    soul_path: Path = settings.data_dir / "soul.md"
    memory_dir: Path = settings.data_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    # Run persistent setup script if it exists
    setup_script: Path = settings.data_dir / "setup.sh"
    if setup_script.exists():
        import subprocess

        logger.info("Running /data/setup.sh...")
        result = subprocess.run(
            ["sh", str(setup_script)], capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            logger.info("setup.sh completed successfully")
        else:
            logger.warning(
                "setup.sh failed (exit %d): %s", result.returncode, result.stderr[:500]
            )
    max_ctx: int = settings.kimi.max_context_tokens
    keep_recent: int = settings.kimi.compaction_keep_recent

    # Background task manager
    bg_task_mgr: BackgroundTaskManager = BackgroundTaskManager()

    # Matrix
    matrix: MatrixConnector = MatrixConnector(
        settings.matrix, store_path=settings.data_dir / "crypto_store"
    )
    matrix.on_message(
        lambda msg: handle_message(
            msg,
            kimi,
            tools,
            skills,
            matrix,
            soul_path,
            memory_dir,
            max_ctx,
            bg_task_mgr,
            compaction_keep_recent=keep_recent,
        )
    )

    # Cron
    cron: CronScheduler = CronScheduler()

    # Heartbeat
    heartbeat: HeartbeatRunner = HeartbeatRunner(
        settings.heartbeat, settings.heartbeat_file
    )

    # Wire cron runner
    sys_prompt: str = build_system_prompt(skills, soul_path, memory_dir)
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
        heartbeat_room: str = settings.heartbeat.room or (
            settings.matrix.room_list[0] if settings.matrix.room_list else ""
        )
        interval: int = settings.heartbeat.every_minutes * 60
        bg_tasks.append(
            asyncio.create_task(
                heartbeat_tick_loop(
                    heartbeat,
                    kimi,
                    tools,
                    matrix,
                    heartbeat_room,
                    sys_prompt,
                    interval=interval,
                ),
                name="heartbeat",
            )
        )

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

    from aineko.logging import setup_logging

    setup_logging(settings.log_level)

    app = FastAPI(title="aineko", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(health_router)

    return app
