"""FastAPI application with lifespan — starts all services."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from aineko.bootstrap import (
    SessionType,
    load_bootstrap_files,
    render_bootstrap_section,
)
from aineko.compaction import compact_messages, run_memory_flush, should_compact
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
from aineko.rss.poller import rss_cleanup_loop, rss_poll_loop
from aineko.heartbeat.runner import HeartbeatRunner
from aineko.kimi.client import ChatResponse, KimiClient
from aineko.matrix.client import MatrixConnector
from aineko.models.message import Message, Role, Session, ToolLog
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
from aineko.tools.memory import init_memory_dir, memory_tool
from aineko.tools import calendar as calendar_mod
from aineko.tools import web_search as web_search_mod
from aineko.tools.calendar import read_calendar_tool
from aineko.tools.rss import query_rss_tool
from aineko.tools.search_chat import search_chat_tool
from aineko.tools.tool_history import search_tool_history_tool
from aineko.tools.background_task import BackgroundTaskManager
from aineko.tools.web_search import web_search_tool
from aineko.tools.web_fetch import web_fetch_tool
from aineko.tools import image as image_mod
from aineko.tools.image import image_tool

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
    registry.register(web_fetch_tool)
    registry.register(image_tool)
    registry.register(create_skill_tool)
    registry.register(memory_tool)
    registry.register(search_tool_history_tool)
    registry.register(query_rss_tool)
    registry.register(search_chat_tool)
    registry.register(read_calendar_tool)
    return registry


_IDENTITY_LINE = "You are aineko, a personal AI assistant."


def build_system_prompt(
    skills: SkillsEngine,
    data_dir: Path,
    session_type: SessionType = "main",
) -> str:
    """Assemble the system prompt from bootstrap files + runtime sections.

    Order: identity → SOUL → AGENTS → TOOLS → USER → HEARTBEAT → MEMORY
    (each bootstrap file rendered as `## {name}`) → skills.
    """
    sections: list[str] = [_IDENTITY_LINE]

    for bf in load_bootstrap_files(data_dir, session_type=session_type):
        sections.append(render_bootstrap_section(bf))

    summaries = skills.summaries()
    if summaries:
        lines = [f"- **{s['name']}**: {s['description']}" for s in summaries]
        sections.append("## Available Skills\n" + "\n".join(lines))

    return "\n\n".join(sections)


async def handle_message(
    msg: IncomingMessage,
    kimi: KimiClient,
    tools: ToolRegistry,
    skills: SkillsEngine,
    matrix: MatrixConnector,
    data_dir: Path,
    max_context_tokens: int,
    bg_tasks: BackgroundTaskManager | None = None,
    compaction_keep_recent: int = 4,
    cron: CronScheduler | None = None,
    interject_queue: "asyncio.Queue[str] | None" = None,
) -> None:
    """Core message handler: load session, run agent, send reply."""
    from sqlalchemy import delete as sa_delete
    from sqlalchemy import update as sa_update

    assert _db.async_session_factory is not None

    sent_messages: list[str] = []
    request_tools: ToolRegistry = build_request_tools(
        tools, matrix, msg.room_id, sent_messages, bg_tasks, kimi, cron
    )
    sys_prompt: str = build_system_prompt(skills, data_dir)

    # Scope 1a: command check, load conversation, collect history ids (if
    # compaction needed). Release DB connection before any LLM call so we
    # don't pin pool slots or hold idle-in-transaction across network I/O.
    async with _db.async_session_factory() as db:
        if await handle_command(db, msg, matrix):
            return

        session, user_msg, messages = await load_conversation(db, msg, sys_prompt)
        session_id: int = session.id
        user_msg_id: int = user_msg.id
        last_input_tokens: int | None = session.last_input_tokens

        from aineko.context import estimate_tokens

        estimated_ctx = sum(estimate_tokens(m.get("content")) for m in messages)
        logger.info(
            "context size",
            extra={
                "event": "context_size",
                "estimated_tokens": estimated_ctx,
                "last_input_tokens": last_input_tokens,
                "max_tokens": max_context_tokens,
                "msg_count": len(messages),
            },
        )

        needs_compaction: bool = should_compact(
            messages, max_context_tokens, last_input_tokens=last_input_tokens
        )
        history_ids: list[int] = []
        if needs_compaction:
            from sqlalchemy import select

            result = await db.execute(
                select(Message.id)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at, Message.id)
            )
            history_ids = list(result.scalars().all())

    # Compaction LLM calls run with no DB handle held. Writes are deferred
    # to scope 2 so that compaction state (deleted history + summary row)
    # commits atomically with the assistant response. If the main LLM call
    # below fails, compaction is discarded too, preserving retry consistency.
    summary_text: str | None = None
    old_removed: int = 0
    if needs_compaction:
        logger.info("conversation approaching context limit, compacting")
        if kimi._settings.memory_flush_enabled:
            await run_memory_flush(messages, kimi, request_tools)
        messages, summary_text, old_removed = await compact_messages(
            messages,
            kimi,
            keep_recent=compaction_keep_recent,
        )

    messages = trim_messages(messages, max_context_tokens)

    async def on_intermediate(text: str) -> None:
        if text not in sent_messages:
            await matrix.send_message(msg.room_id, text)
            sent_messages.append(text)

    # Main LLM call — no DB connection held. If it raises, scope 2 is
    # skipped and no compaction writes happen.
    response: ChatResponse = await kimi.chat_loop(
        messages,
        request_tools,
        on_intermediate=on_intermediate,
        interject_queue=interject_queue,
    )

    if response.content and not msg.suppress_text_response:
        footer: str = format_tool_footer(response.tool_history)
        await matrix.send_message(msg.room_id, response.content + footer)

    # Scope 2: compaction writes + assistant response + tool logs in one
    # transaction. persist_response commits at the end.
    async with _db.async_session_factory() as db:
        if summary_text:
            if old_msg_ids := history_ids[:old_removed]:
                await db.execute(
                    sa_delete(ToolLog).where(ToolLog.message_id.in_(old_msg_ids))
                )
                await db.execute(sa_delete(Message).where(Message.id.in_(old_msg_ids)))
            db.add(
                Message(
                    session_id=session_id,
                    role=Role.SYSTEM,
                    content=summary_text,
                )
            )
        input_tokens = response.usage.get("input_tokens")
        if input_tokens is not None:
            await db.execute(
                sa_update(Session)
                .where(Session.id == session_id)
                .values(last_input_tokens=int(input_tokens))
            )
        await persist_response(db, session_id, user_msg_id, response, sent_messages)


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
    image_mod._settings = settings.llm
    calendar_mod.caldav_url = settings.caldav.url
    calendar_mod.caldav_username = settings.caldav.username
    calendar_mod.caldav_password = settings.caldav.password

    # LLM
    kimi: KimiClient = KimiClient(settings.llm)

    # Memory dir (daily logs)
    memory_dir: Path = settings.data_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    init_memory_dir(memory_dir)

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
    max_ctx: int = settings.llm.max_context_tokens
    keep_recent: int = settings.llm.compaction_keep_recent

    # Background task manager
    bg_task_mgr: BackgroundTaskManager = BackgroundTaskManager()

    # Matrix
    matrix: MatrixConnector = MatrixConnector(
        settings.matrix, store_path=settings.data_dir / "crypto_store"
    )

    # Cron (constructed before handler registration so tools can see it)
    cron: CronScheduler = CronScheduler()

    matrix.on_message(
        lambda msg, interject: handle_message(
            msg,
            kimi,
            tools,
            skills,
            matrix,
            settings.data_dir,
            max_ctx,
            bg_task_mgr,
            compaction_keep_recent=keep_recent,
            cron=cron,
            interject_queue=interject,
        )
    )

    # Heartbeat
    heartbeat: HeartbeatRunner = HeartbeatRunner(
        settings.heartbeat, settings.heartbeat_file
    )

    # Wire cron runner (cron uses its own prompt mode later; for now, main)
    sys_prompt: str = build_system_prompt(
        skills, settings.data_dir, session_type="cron"
    )
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
        interval: int = settings.heartbeat.every_minutes * 60
        bg_tasks.append(
            asyncio.create_task(
                heartbeat_tick_loop(
                    heartbeat,
                    kimi,
                    tools,
                    matrix,
                    settings.matrix.room_id,
                    sys_prompt,
                    interval=interval,
                ),
                name="heartbeat",
            )
        )

    # RSS poller
    if settings.rss.enabled:
        import json

        from aineko.config import RssFeedConfig

        rss_feeds: list[RssFeedConfig] = []
        if settings.rss_feeds_file.exists():
            raw = json.loads(settings.rss_feeds_file.read_text())
            rss_feeds = [RssFeedConfig(**f) for f in raw]
        else:
            logger.warning("RSS enabled but %s not found", settings.rss_feeds_file)

        if rss_feeds:
            bg_tasks.append(
                asyncio.create_task(
                    rss_poll_loop(
                        matrix,
                        settings.matrix.room_id,
                        rss_feeds,
                        settings.rss.poll_interval,
                    ),
                    name="rss-poller",
                )
            )
            logger.info(
                "RSS poller started: %d feed(s), room=%s",
                len(rss_feeds),
                settings.matrix.room_id,
            )

        bg_tasks.append(asyncio.create_task(rss_cleanup_loop(), name="rss-cleanup"))

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
