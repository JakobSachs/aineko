"""Tests for structured JSON logging."""

import json
import logging

from aineko.logging import StructuredFormatter


def test_basic_format():
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello world",
        args=(),
        exc_info=None,
    )
    line = formatter.format(record)
    data = json.loads(line)
    assert data["msg"] == "hello world"
    assert data["level"] == "info"
    assert data["logger"] == "test"
    assert "ts" in data


def test_extra_fields():
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="tool call",
        args=(),
        exc_info=None,
    )
    record.event = "tool_call"
    record.tool = "bash"
    record.tool_args = {"command": "ls"}
    line = formatter.format(record)
    data = json.loads(line)
    assert data["event"] == "tool_call"
    assert data["tool"] == "bash"
    assert data["tool_args"] == {"command": "ls"}


def test_missing_extra_fields_omitted():
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="simple",
        args=(),
        exc_info=None,
    )
    line = formatter.format(record)
    data = json.loads(line)
    assert "event" not in data
    assert "tool" not in data


def test_exception_included():
    formatter = StructuredFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="failed",
        args=(),
        exc_info=exc_info,
    )
    line = formatter.format(record)
    data = json.loads(line)
    assert "exception" in data
    assert "boom" in data["exception"]
