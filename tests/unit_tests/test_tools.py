"""Unit tests for tool functions (no LLM/API calls)."""

from __future__ import annotations

from react_agent.tools import (
    get_all_tools,
    get_tool_by_name,
    python_repl,
    run_mini_react_loop,
)


class TestPythonRepl:
    """Sandboxed Python REPL tool."""

    def test_basic_calculation(self) -> None:
        result = python_repl("2 + 2")
        assert result == "4"

    def test_math_module_available(self) -> None:
        result = python_repl("math.sqrt(16)")
        assert result == "4.0"

    def test_builtin_sum(self) -> None:
        result = python_repl("sum([1, 2, 3, 4, 5])")
        assert result == "15"

    def test_float_division(self) -> None:
        result = python_repl("10 / 3")
        assert "3.33" in result

    def test_list_comprehension(self) -> None:
        result = python_repl("[x * 2 for x in range(5)]")
        assert result == "[0, 2, 4, 6, 8]"

    def test_error_division_by_zero(self) -> None:
        result = python_repl("1 / 0")
        assert result.startswith("Error:")

    def test_syntax_error(self) -> None:
        result = python_repl("def foo():")
        assert result.startswith("Error:")

    def test_open_blocked(self) -> None:
        """open() should not be available (sandbox)."""
        result = python_repl("open('/etc/passwd')")
        assert result.startswith("Error:")

    def test_import_blocked(self) -> None:
        """__import__ should not be available (sandbox)."""
        result = python_repl("__import__('os')")
        assert result.startswith("Error:")

    def test_exec_blocked(self) -> None:
        """exec should not be available (sandbox)."""
        result = python_repl("exec('print(1)')")
        assert result.startswith("Error:")

    def test_empty_string(self) -> None:
        result = python_repl("")
        assert result.startswith("Error:")  # eval('') raises SyntaxError


class TestToolRegistry:
    """Tool registry functions."""

    def test_get_all_tools_returns_builtins(self) -> None:
        tools = get_all_tools()
        tool_names = {t.__name__ for t in tools}
        assert "search" in tool_names
        assert "python_repl" in tool_names
        assert "retrieve" in tool_names

    def test_get_tool_by_name_found(self) -> None:
        tool = get_tool_by_name("python_repl")
        assert tool is not None
        assert callable(tool)

    def test_get_tool_by_name_not_found(self) -> None:
        tool = get_tool_by_name("nonexistent_tool_xyz")
        assert tool is None

    def test_get_tool_by_name_search(self) -> None:
        tool = get_tool_by_name("search")
        assert tool is not None
        assert callable(tool)


class TestMiniReActLoop:
    """run_mini_react_loop shared helper."""

    def test_loop_stops_when_no_tool_calls(self) -> None:
        """If the model returns no tool_calls, loop exits immediately."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        async def _run() -> None:
            mock_model = AsyncMock()
            mock_response = MagicMock()
            mock_response.tool_calls = []  # No tools ⇒ stop immediately
            mock_response.content = "final answer"
            mock_model.ainvoke.return_value = mock_response

            messages = [MagicMock(content="test prompt")]
            result = await run_mini_react_loop(mock_model, [], messages, max_rounds=3)

            # Should only call once (no retry needed)
            assert mock_model.ainvoke.call_count == 1
            assert len(result) == 2  # original + response
            assert result[-1].content == "final answer"

        asyncio.run(_run())

    def test_function_is_callable(self) -> None:
        """Verify that run_mini_react_loop is callable and importable."""
        assert callable(run_mini_react_loop)
