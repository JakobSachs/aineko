"""Tests for HeartbeatRunner — pre/post filtering logic."""

from unittest.mock import MagicMock, patch

import pytest

from aineko.config import HeartbeatSettings
from aineko.heartbeat.runner import HeartbeatRunner


@pytest.fixture
def settings():
    return HeartbeatSettings(
        enabled=True,
        every_minutes=30,
        active_hours_start="08:00",
        active_hours_end="23:00",
        room="!room:test",
    )


@pytest.fixture
def heartbeat_file(tmp_path):
    return tmp_path / "HEARTBEAT.md"


@pytest.fixture
def runner(settings, heartbeat_file):
    return HeartbeatRunner(settings, heartbeat_file)


def _mock_local_time(hour, minute=0):
    """Create a mock datetime.now() that returns a fake local time via .astimezone()."""
    local = MagicMock()
    local.hour = hour
    local.minute = minute
    mock_dt = MagicMock()
    utc_now = MagicMock()
    utc_now.astimezone.return_value = local
    mock_dt.now.return_value = utc_now
    return mock_dt


class TestShouldRun:
    def test_disabled(self, heartbeat_file):
        s = HeartbeatSettings(enabled=False)
        r = HeartbeatRunner(s, heartbeat_file)
        assert r.should_run() is False

    def test_agent_busy(self, runner):
        assert runner.should_run(agent_busy=True) is False

    def test_outside_active_hours(self, runner, heartbeat_file):
        heartbeat_file.write_text("- check logs")
        with patch("aineko.heartbeat.runner.datetime", _mock_local_time(3, 0)):
            assert runner.should_run() is False

    def test_no_tasks(self, runner):
        assert runner.should_run() is False

    def test_empty_heartbeat_file(self, runner, heartbeat_file):
        heartbeat_file.write_text("")
        assert runner.should_run() is False

    def test_all_conditions_met(self, runner, heartbeat_file):
        heartbeat_file.write_text("- check the logs")
        with patch("aineko.heartbeat.runner.datetime", _mock_local_time(12, 0)):
            assert runner.should_run() is True


class TestHasTasks:
    def test_file_missing(self, runner):
        assert runner._has_tasks() is False

    def test_empty_file(self, runner, heartbeat_file):
        heartbeat_file.write_text("")
        assert runner._has_tasks() is False

    def test_only_headers(self, runner, heartbeat_file):
        heartbeat_file.write_text("# Tasks\n## Sub-section\n")
        assert runner._has_tasks() is False

    def test_only_empty_checkboxes(self, runner, heartbeat_file):
        heartbeat_file.write_text("- [ ]\n- [x]\n")
        assert runner._has_tasks() is False

    def test_real_task(self, runner, heartbeat_file):
        heartbeat_file.write_text("# Tasks\n- [ ] check server logs\n")
        assert runner._has_tasks() is True

    def test_only_html_comments(self, runner, heartbeat_file):
        heartbeat_file.write_text(
            "# Tasks\n<!-- aineko reads this every 30m -->\n<!-- say HEARTBEAT_OK -->\n"
        )
        assert runner._has_tasks() is False

    def test_non_checkbox_line(self, runner, heartbeat_file):
        heartbeat_file.write_text("check stuff\n")
        assert runner._has_tasks() is True


class TestGetTasks:
    def test_file_missing(self, runner):
        assert runner.get_tasks() == ""

    def test_reads_content(self, runner, heartbeat_file):
        heartbeat_file.write_text("- check logs\n- deploy")
        assert runner.get_tasks() == "- check logs\n- deploy"


class TestShouldDeliver:
    def test_heartbeat_ok_suppressed(self, runner):
        assert runner.should_deliver("HEARTBEAT_OK") is False

    def test_heartbeat_ok_case_insensitive(self, runner):
        assert runner.should_deliver("  heartbeat_ok  ") is False

    def test_duplicate_response_suppressed(self, runner):
        assert runner.should_deliver("All clear") is True
        assert runner.should_deliver("All clear") is False

    def test_different_responses_delivered(self, runner):
        assert runner.should_deliver("Alert: disk full") is True
        assert runner.should_deliver("Alert: CPU high") is True

    def test_last_response_tracked(self, runner):
        runner.should_deliver("first")
        runner.should_deliver("second")
        assert runner.should_deliver("first") is True


class TestInActiveHours:
    def test_within_hours(self, runner):
        with patch("aineko.heartbeat.runner.datetime", _mock_local_time(12, 0)):
            assert runner._in_active_hours() is True

    def test_before_start(self, runner):
        with patch("aineko.heartbeat.runner.datetime", _mock_local_time(5, 0)):
            assert runner._in_active_hours() is False

    def test_after_end(self, runner):
        with patch("aineko.heartbeat.runner.datetime", _mock_local_time(23, 30)):
            assert runner._in_active_hours() is False

    def test_at_exact_start(self, runner):
        with patch("aineko.heartbeat.runner.datetime", _mock_local_time(8, 0)):
            assert runner._in_active_hours() is True

    def test_at_exact_end(self, runner):
        with patch("aineko.heartbeat.runner.datetime", _mock_local_time(23, 0)):
            assert runner._in_active_hours() is True
