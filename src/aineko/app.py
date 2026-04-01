"""FastAPI application with lifespan — starts all services."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from aineko.compaction import compact_messages, should_compact
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
from aineko.tools.files import edit_file_tool, read_file_tool, write_file_tool
from aineko.tools.glob import glob_tool
from aineko.tools.grep import grep_tool
from aineko.tools.registry import ToolRegistry
from aineko.tools.memory import memory_recall_tool
from aineko.tools import web_search as web_search_mod
from aineko.tools.search_chat import search_chat_tool
from aineko.tools.tool_history import search_tool_history_tool
from aineko.tools.background_task import BackgroundTaskManager
from aineko.tools.messaging import (
    make_background_task_tools,
    make_send_file_tool,
    make_send_message_tool,
)
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
    registry.register(memory_recall_tool)
    registry.register(search_tool_history_tool)
    registry.register(search_chat_tool)
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
) -> None:
    """Core message handler: load session, run agent, send reply."""
    from sqlalchemy import select

    from aineko.models.message import Message, Role, Session, ToolLog

    # Handle /reset command
    if msg.body.strip().lower() in ("/reset", "/clear", "/forget"):
        async for db in get_session():
            result = await db.execute(
                select(Session).where(Session.room_id == msg.room_id)
            )
            session = result.scalar_one_or_none()
            if session:
                await db.execute(
                    select(Message)
                    .where(Message.session_id == session.id)
                    .with_for_update()
                )
                from sqlalchemy import delete

                await db.execute(
                    delete(Message).where(Message.session_id == session.id)
                )
                await db.commit()
            await matrix.send_message(msg.room_id, "conversation cleared, fresh start")
        return

    # Track messages sent via the send_message tool
    sent_messages: list[str] = []

    # Build per-request tools with context-bound messaging tools
    request_tools = ToolRegistry()
    for tool_def in tools._tools.values():
        request_tools.register(tool_def)
    request_tools.register(make_send_message_tool(matrix, msg.room_id, sent_messages))
    request_tools.register(make_send_file_tool(matrix, msg.room_id))

    if bg_tasks is not None:
        for tool_def in make_background_task_tools(bg_tasks, msg.room_id, matrix):
            request_tools.register(tool_def)

    async for db in get_session():
        # Get or create session for this room
        result = await db.execute(select(Session).where(Session.room_id == msg.room_id))
        session = result.scalar_one_or_none()
        if session is None:
            session = Session(room_id=msg.room_id)
            db.add(session)
            await db.flush()

        # Save incoming message
        user_msg = Message(
            session_id=session.id,
            role=Role.USER,
            content=msg.body,
        )
        db.add(user_msg)
        await db.flush()

        # Load conversation history
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session.id)
            .order_by(Message.created_at)
        )
        history = result.scalars().all()

        # Build messages for Kimi
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(skills, soul_path, memory_dir),
            }
        ]
        import json as _json

        for m in history:
            if m.role == Role.ASSISTANT and m.tool_name == "__has_tool_calls__":
                # Reconstruct Anthropic content blocks from stored JSON
                try:
                    blocks = _json.loads(m.content)
                    messages.append({"role": "assistant", "content": blocks})
                except _json.JSONDecodeError:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": m.content}],
                        }
                    )
            elif m.role == Role.TOOL and m.tool_name == "__tool_results__":
                # Reconstruct tool_result user message from stored JSON
                try:
                    blocks = _json.loads(m.content)
                    messages.append({"role": "user", "content": blocks})
                except _json.JSONDecodeError:
                    pass  # skip corrupted tool results
            elif m.role == Role.ASSISTANT:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": m.content}],
                    }
                )
            elif m.role == Role.SYSTEM:
                messages.append({"role": "system", "content": m.content})
            else:
                messages.append({"role": m.role.value, "content": m.content})

        # If current message has an image, make the last user message multimodal
        if msg.image_b64 and msg.image_mime:
            last_user = messages[-1]
            last_user["content"] = [
                {"type": "text", "text": last_user["content"]},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{msg.image_mime};base64,{msg.image_b64}"
                    },
                },
            ]

        # Compact if conversation is getting long (LLM-powered summary)
        if should_compact(messages, max_context_tokens):
            logger.info("conversation approaching context limit, compacting")
            messages = await compact_messages(messages, kimi)

            # Persist the compaction: delete old messages, insert summary
            # Find messages that were compacted (everything before recent ones)
            from sqlalchemy import delete as sa_delete

            old_msg_ids = [m.id for m in history[:-4]] if len(history) > 4 else []
            if old_msg_ids:
                await db.execute(sa_delete(Message).where(Message.id.in_(old_msg_ids)))
                # Save the compaction summary to DB
                summary_msg = [
                    m
                    for m in messages
                    if m.get("role") == "system" and "compacted" in m.get("content", "")
                ]
                if summary_msg:
                    db.add(
                        Message(
                            session_id=session.id,
                            role=Role.SYSTEM,
                            content=summary_msg[0]["content"],
                        )
                    )
                await db.flush()

        # Trim as safety net (in case compaction wasn't enough)
        messages = trim_messages(messages, max_context_tokens)

        # Run agent loop
        response = await kimi.chat_loop(messages, request_tools)

        # Auto-send final response if model has text content
        if response.content:
            await matrix.send_message(msg.room_id, response.content)

        # Persist conversation in Anthropic-reconstructable format.
        # If tools were called, save: assistant (with tool_use refs) + tool results + final text.
        # This lets us rebuild proper content blocks when loading history.
        import json as _json

        if response.tool_history:
            # Save assistant message with tool call metadata as JSON content blocks
            assistant_blocks: list[dict[str, Any]] = []
            # Include any intermediate send_message content
            for sm in sent_messages:
                assistant_blocks.append({"type": "text", "text": sm})
            for rec in response.tool_history:
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": f"persisted_{rec.tool_name}_{id(rec)}",
                        "name": rec.tool_name,
                        "input": _json.loads(rec.arguments) if rec.arguments else {},
                    }
                )
            if response.content:
                assistant_blocks.append({"type": "text", "text": response.content})
            db.add(
                Message(
                    session_id=session.id,
                    role=Role.ASSISTANT,
                    # Store as JSON so we can reconstruct content blocks on load
                    content=_json.dumps(assistant_blocks),
                    token_count=response.usage.get("total_tokens"),
                    tool_name="__has_tool_calls__",  # marker
                )
            )

            # Save tool results as a single "tool" role message with JSON content blocks
            result_blocks: list[dict[str, Any]] = []
            for rec in response.tool_history:
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": f"persisted_{rec.tool_name}_{id(rec)}",
                        "content": rec.result[:2000],  # cap stored results
                    }
                )
            db.add(
                Message(
                    session_id=session.id,
                    role=Role.TOOL,
                    content=_json.dumps(result_blocks),
                    tool_name="__tool_results__",  # marker
                )
            )
        else:
            # No tool calls — save plain assistant message
            all_visible = sent_messages.copy()
            if response.content:
                all_visible.append(response.content)
            visible = "\n".join(all_visible) if all_visible else ""
            if visible:
                db.add(
                    Message(
                        session_id=session.id,
                        role=Role.ASSISTANT,
                        content=visible,
                        token_count=response.usage.get("total_tokens"),
                    )
                )

        # Persist tool call history
        for rec in response.tool_history:
            db.add(
                ToolLog(
                    session_id=session.id,
                    message_id=user_msg.id,
                    tool_name=rec.tool_name,
                    arguments=rec.arguments,
                    result=rec.result,
                )
            )

        await db.commit()


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
    web_search_mod.brave_api_key = settings.brave_api_key

    # Kimi
    kimi = KimiClient(settings.kimi)

    # Soul + Memory
    soul_path = settings.data_dir / "soul.md"
    memory_dir = settings.data_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    # Run persistent setup script if it exists
    setup_script = settings.data_dir / "setup.sh"
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
    max_ctx = settings.kimi.max_context_tokens

    # Background task manager
    bg_task_mgr = BackgroundTaskManager()

    # Matrix
    matrix = MatrixConnector(
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
        )
    )

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

                    # Give heartbeat a send_message tool so it can proactively reach out
                    hb_tools = ToolRegistry()
                    for tool_def in tools._tools.values():
                        hb_tools.register(tool_def)
                    hb_tools.register(
                        make_send_message_tool(matrix, heartbeat_room, [])
                    )

                    response = await kimi.chat_loop(messages, hb_tools)

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

    from aineko.logging import setup_logging

    setup_logging(settings.log_level)

    app = FastAPI(title="aineko", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(health_router)

    return app
