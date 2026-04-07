"""Tests for background task spawning and result injection."""

import asyncio
from datetime import datetime, timezone

import pytest

import aineko.tools.bash as bash_mod
from aineko.queue import MessageQueue
from aineko.schemas.message import IncomingMessage
from aineko.tools.background_task import BackgroundTaskManager

# The real run_bash hardcodes cwd=/data which doesn't exist in test env.
# Replace with a version that uses /tmp.
_orig_run_bash = bash_mod.run_bash


async def _test_run_bash(command: str, timeout: int = 60) -> str:
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd="/tmp",
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode(errors="replace")
        return output + f"\n[exit code: {proc.returncode}]"
    except asyncio.TimeoutError:
        if proc is not None:
            proc.kill()
            try:
                await proc.communicate()
            except Exception:
                pass
        return f"Command timed out after {timeout}s"
    finally:
        # Release the subprocess transport eagerly so its finalizer doesn't
        # fire later on an already-closed event loop.
        if isinstance(proc, asyncio.subprocess.Process):
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                transport.close()


@pytest.fixture(autouse=True)
def _patch_bash(monkeypatch):
    monkeypatch.setattr(bash_mod, "run_bash", _test_run_bash)


def _msg(body: str, room: str = "!room:test") -> IncomingMessage:
    return IncomingMessage(
        room_id=room,
        sender="@user:test",
        body=body,
        timestamp=datetime.now(timezone.utc),
        event_id=f"evt_{id(body)}",
    )


@pytest.mark.asyncio
async def test_spawn_runs_command():
    """Spawned task executes and captures output."""
    mgr = BackgroundTaskManager()
    record = mgr.spawn(command="echo hello_bg", label="echo test", room_id="!r:t")
    assert not record.finished
    await record.asyncio_task
    assert record.finished
    assert "hello_bg" in record.result


@pytest.mark.asyncio
async def test_spawn_returns_immediately():
    """spawn() returns before the command finishes."""
    mgr = BackgroundTaskManager()
    record = mgr.spawn(command="sleep 5", label="slow", room_id="!r:t", timeout=10)
    # Should be running, not finished
    assert not record.finished
    assert record.asyncio_task is not None
    # Cancel so test doesn't hang
    record.asyncio_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await record.asyncio_task


@pytest.mark.asyncio
async def test_inject_fn_called_on_completion():
    """The inject callback fires with task_id, label, and result."""
    injected: list[tuple[str, str, str]] = []

    async def fake_inject(task_id: str, label: str, result: str) -> None:
        injected.append((task_id, label, result))

    mgr = BackgroundTaskManager()
    record = mgr.spawn(
        command="echo injected_output",
        label="inject test",
        room_id="!r:t",
        inject_fn=fake_inject,
    )
    await record.asyncio_task
    assert len(injected) == 1
    assert injected[0][0] == record.id
    assert injected[0][1] == "inject test"
    assert "injected_output" in injected[0][2]


@pytest.mark.asyncio
async def test_inject_fn_called_on_error():
    """The inject callback fires even when the command fails."""
    injected: list[tuple[str, str, str]] = []

    async def fake_inject(task_id: str, label: str, result: str) -> None:
        injected.append((task_id, label, result))

    mgr = BackgroundTaskManager()
    record = mgr.spawn(
        command="nonexistent_cmd_xyz_42",
        label="failing task",
        room_id="!r:t",
        inject_fn=fake_inject,
    )
    await record.asyncio_task
    assert record.finished
    assert len(injected) == 1


@pytest.mark.asyncio
async def test_list_tasks():
    """list_tasks shows running and finished tasks."""
    mgr = BackgroundTaskManager()
    r1 = mgr.spawn(command="echo one", label="first", room_id="!r:t")
    r2 = mgr.spawn(command="echo two", label="second", room_id="!r:t")
    # Both should appear
    tasks = mgr.list_tasks()
    assert len(tasks) == 2
    ids = {t["id"] for t in tasks}
    assert r1.id in ids
    assert r2.id in ids

    # Wait for completion
    await r1.asyncio_task
    await r2.asyncio_task
    tasks = mgr.list_tasks()
    assert all(t["finished"] for t in tasks)


@pytest.mark.asyncio
async def test_cleanup_finished():
    """cleanup_finished removes completed tasks."""
    mgr = BackgroundTaskManager()
    r1 = mgr.spawn(command="echo done", label="cleanup", room_id="!r:t")
    await r1.asyncio_task
    assert mgr.cleanup_finished() == 1
    assert mgr.list_tasks() == []


@pytest.mark.asyncio
async def test_timeout_kills_task(monkeypatch):
    """Task that exceeds timeout reports timeout."""

    async def slow_bash(command: str, timeout: int = 60, **kw) -> str:
        # Simulate what run_bash does on timeout — wait, then return message
        await asyncio.sleep(0.05)
        return f"Command timed out after {timeout}s"

    monkeypatch.setattr(bash_mod, "run_bash", slow_bash)
    mgr = BackgroundTaskManager()
    record = mgr.spawn(command="sleep 999", label="timeout", room_id="!r:t", timeout=1)
    await record.asyncio_task
    assert record.finished
    assert "timed out" in record.result.lower()


@pytest.mark.asyncio
async def test_get_task():
    """get() returns a record by ID, or None."""
    mgr = BackgroundTaskManager()
    r = mgr.spawn(command="echo hi", label="test", room_id="!r:t")
    assert mgr.get(r.id) is r
    assert mgr.get("nonexistent") is None
    await r.asyncio_task


@pytest.mark.asyncio
async def test_list_tasks_elapsed_minutes():
    """Elapsed time over 60s is shown in minutes."""
    mgr = BackgroundTaskManager()
    r = mgr.spawn(command="echo hi", label="old", room_id="!r:t")
    # Backdate the creation time
    r.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    tasks = mgr.list_tasks()
    assert any("m" in t["elapsed"] for t in tasks)
    r.asyncio_task.cancel()
    try:
        await r.asyncio_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_spawn_error_path():
    """When _run raises, result captures the error and inject_fn is still called."""
    injected = []

    async def fake_inject(task_id, label, result):
        injected.append(result)

    mgr = BackgroundTaskManager()
    # Patch run_bash to raise inside the task
    import aineko.tools.bash as bmod

    original = bmod.run_bash

    async def exploding_bash(command, timeout=60, **kw):
        raise RuntimeError("kaboom")

    bmod.run_bash = exploding_bash
    try:
        r = mgr.spawn(
            command="anything", label="boom", room_id="!r:t", inject_fn=fake_inject
        )
        await r.asyncio_task
        assert r.finished
        assert "kaboom" in r.result
        assert len(injected) == 1
    finally:
        bmod.run_bash = original


@pytest.mark.asyncio
async def test_result_injected_into_queue():
    """End-to-end: task result flows through inject_message into the message queue."""
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        received.append(msg)

    q = MessageQueue(handler)

    async def inject_via_queue(task_id: str, label: str, result: str) -> None:
        body = f"[Background task `{label}` (id: {task_id}) completed]\n\n{result}"
        msg = IncomingMessage(
            room_id="!room:test",
            sender="background_task",
            body=body,
            timestamp=datetime.now(timezone.utc),
            event_id=f"$bg-{task_id}",
        )
        await q.enqueue(msg)

    mgr = BackgroundTaskManager()
    record = mgr.spawn(
        command="echo e2e_result",
        label="e2e",
        room_id="!room:test",
        inject_fn=inject_via_queue,
    )
    await record.asyncio_task
    # Wait for debounce + processing
    await asyncio.sleep(2.5)

    assert len(received) == 1
    assert "e2e_result" in received[0].body
    assert "Background task" in received[0].body
    assert received[0].sender == "background_task"
