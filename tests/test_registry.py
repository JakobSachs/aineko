"""Tests for tool registry."""

import pytest

from aineko.tools.registry import ToolDef, ToolRegistry


async def _echo(text: str) -> str:
    return f"echo: {text}"


async def _add(a: int, b: int) -> str:
    return str(a + b)


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(
        ToolDef(
            name="echo",
            description="Echo text back",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=_echo,
        )
    )
    r.register(
        ToolDef(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            handler=_add,
        )
    )
    return r


def test_register_and_get(registry):
    assert registry.get("echo") is not None
    assert registry.get("echo").name == "echo"
    assert registry.get("nonexistent") is None


def test_schemas(registry):
    schemas = registry.schemas()
    assert len(schemas) == 2
    names = {s["function"]["name"] for s in schemas}
    assert names == {"echo", "add"}
    assert schemas[0]["type"] == "function"


@pytest.mark.asyncio
async def test_call_success(registry):
    result = await registry.call("echo", {"text": "hello"})
    assert result == "echo: hello"


@pytest.mark.asyncio
async def test_call_unknown_tool(registry):
    result = await registry.call("nonexistent", {})
    assert "unknown tool" in result.lower()


@pytest.mark.asyncio
async def test_call_error_handling(registry):
    """Handler that raises returns error string."""

    async def _fail() -> str:
        raise ValueError("broken")

    registry.register(
        ToolDef(
            name="fail",
            description="Always fails",
            parameters={"type": "object", "properties": {}},
            handler=_fail,
        )
    )
    result = await registry.call("fail", {})
    assert "Error" in result
    assert "broken" in result
