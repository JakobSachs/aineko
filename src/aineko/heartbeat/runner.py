"""Heartbeat runner with smart pre-filtering."""

import logging
from pathlib import Path

from aineko.config import HeartbeatSettings
from aineko.time import now_local

logger = logging.getLogger(__name__)


class HeartbeatRunner:
    def __init__(self, settings: HeartbeatSettings, heartbeat_file: Path) -> None:
        self._settings = settings
        self._file = heartbeat_file
        self._last_response: str = ""

    def should_run(self, agent_busy: bool = False) -> bool:
        """Pre-filter: decide if we should invoke the agent."""
        if not self._settings.enabled:
            return False

        if agent_busy:
            logger.debug("Heartbeat skipped: agent busy")
            return False

        if not self._in_active_hours():
            logger.debug("Heartbeat skipped: outside active hours")
            return False

        if not self._has_tasks():
            logger.debug("Heartbeat skipped: no tasks in HEARTBEAT.md")
            return False

        return True

    def get_tasks(self) -> str:
        """Read HEARTBEAT.md content."""
        if not self._file.exists():
            return ""
        return self._file.read_text()

    def should_deliver(self, response: str) -> bool:
        """Post-filter: decide if the agent's response should be sent to Matrix."""
        normalized = response.strip().upper()

        if normalized == "HEARTBEAT_OK":
            return False

        if response.strip() == self._last_response:
            logger.debug("Heartbeat suppressed: duplicate response")
            return False

        self._last_response = response.strip()
        return True

    def _in_active_hours(self) -> bool:
        now = now_local()
        start_h, start_m = map(int, self._settings.active_hours_start.split(":"))
        end_h, end_m = map(int, self._settings.active_hours_end.split(":"))
        current = now.hour * 60 + now.minute
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        return start <= current <= end

    def _has_tasks(self) -> bool:
        content = self.get_tasks()
        if not content:
            return False
        # Strip markdown headers, whitespace, empty checkboxes
        lines = content.strip().splitlines()
        for line in lines:
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("#")
                and not stripped.startswith("<!--")
                and stripped not in ("- [ ]", "- [x]")
            ):
                return True
        return False
