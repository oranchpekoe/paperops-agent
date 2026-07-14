"""Unit tests for state definitions (no external dependencies)."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from react_agent.state import InputState, MainState


class TestMainStateDefaults:
    """All MainState fields should have sensible defaults."""

    def test_mode_default_empty(self) -> None:
        state = MainState()
        assert state.mode == ""

    def test_plan_steps_default_empty(self) -> None:
        state = MainState()
        assert state.plan_steps == []

    def test_step_results_default_empty(self) -> None:
        state = MainState()
        assert state.step_results == []

    def test_reflection_iteration_default_zero(self) -> None:
        state = MainState()
        assert state.reflection_iteration == 0

    def test_supervisor_iteration_default_zero(self) -> None:
        state = MainState()
        assert state.supervisor_iteration == 0

    def test_supervisor_next_specialist_default_empty(self) -> None:
        state = MainState()
        assert state.supervisor_next_specialist == ""

    def test_benchmark_mode_default_false(self) -> None:
        state = MainState()
        assert state.benchmark_mode is False

    def test_user_query_default_empty(self) -> None:
        state = MainState()
        assert state.user_query == ""

    def test_recalled_facts_default_empty(self) -> None:
        state = MainState()
        assert state.recalled_facts == ""


class TestInputState:
    """InputState tests — message handling."""

    def test_messages_default_empty(self) -> None:
        state = InputState()
        assert state.messages == []

    def test_messages_can_be_set(self) -> None:
        msg = HumanMessage(content="hello")
        state = InputState(messages=[msg])
        assert len(state.messages) == 1
        assert state.messages[0].content == "hello"


class TestBenchmarkMode:
    """benchmark_mode field should be settable and readable."""

    def test_benchmark_mode_true(self) -> None:
        state = MainState(benchmark_mode=True)
        assert state.benchmark_mode is True

    def test_benchmark_mode_can_be_toggled(self) -> None:
        state = MainState()
        assert state.benchmark_mode is False
        state.benchmark_mode = True
        assert state.benchmark_mode is True
