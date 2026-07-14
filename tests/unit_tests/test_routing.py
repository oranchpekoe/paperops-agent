"""Unit tests for routing logic (no LLM calls — pure logic tests)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from react_agent.memory import (
    _BENCHMARK_SIGNALS,
    _looks_like_benchmark,
    set_benchmark_mode,
    _in_benchmark_mode,
)
from react_agent.modes.supervisor import SupervisorDecision, _parse_text_decision


# ---------------------------------------------------------------------------
# Supervisor text-parsing fallback
# ---------------------------------------------------------------------------


class TestParseTextDecision:
    """Text-based supervisor decision parsing (fallback path)."""

    def test_research_prefix(self) -> None:
        d = _parse_text_decision("RESEARCH\nSearch for GDP data of Japan")
        assert isinstance(d, SupervisorDecision)
        assert d.action == "RESEARCH"
        assert "GDP" in d.task

    def test_execute_prefix(self) -> None:
        d = _parse_text_decision("EXECUTE\nCalculate 5% of GDP")
        assert isinstance(d, SupervisorDecision)
        assert d.action == "EXECUTE"
        assert "Calculate" in d.task

    def test_analyse_prefix(self) -> None:
        d = _parse_text_decision("ANALYSE\nEvaluate the search results")
        assert isinstance(d, SupervisorDecision)
        assert d.action == "ANALYSE"

    def test_answer_prefix(self) -> None:
        d = _parse_text_decision("ANSWER\nHere is the final result")
        assert isinstance(d, SupervisorDecision)
        assert d.action == "ANSWER"

    def test_keyword_research(self) -> None:
        """Unstructured text containing 'research' → RESEARCH."""
        d = _parse_text_decision("I think we should search the web for more data")
        assert isinstance(d, SupervisorDecision)
        assert d.action == "RESEARCH"

    def test_keyword_execute(self) -> None:
        """Unstructured text containing 'calculate' → EXECUTE."""
        d = _parse_text_decision("We need to compute the total sum now")
        assert isinstance(d, SupervisorDecision)
        assert d.action == "EXECUTE"

    def test_keyword_analyse(self) -> None:
        """Unstructured text containing 'evaluate' → ANALYSE."""
        d = _parse_text_decision("Let's reason about this carefully")
        assert isinstance(d, SupervisorDecision)
        assert d.action == "ANALYSE"

    def test_default_fallback(self) -> None:
        """Anything that doesn't match keywords → ANSWER."""
        d = _parse_text_decision("The sky is blue")
        assert isinstance(d, SupervisorDecision)
        assert d.action == "ANSWER"


# ---------------------------------------------------------------------------
# Benchmark detection — graph layer
# ---------------------------------------------------------------------------


class TestBenchmarkQueryDetection:
    """_looks_like_benchmark in graph.py."""

    def test_tokyo_trip_benchmark(self) -> None:
        assert _looks_like_benchmark(
            "Plan a 3-day trip to Tokyo for a first-time visitor."
        ) is True

    def test_capital_of_france_benchmark(self) -> None:
        assert _looks_like_benchmark("What is the capital of France?") is True

    def test_compound_interest_benchmark(self) -> None:
        assert _looks_like_benchmark(
            "Calculate the compound interest on $10,000 at 5% annual rate over 10 years, compounded monthly."
        ) is True

    def test_normal_query_not_detected(self) -> None:
        assert _looks_like_benchmark("How to make a plan to travel Beijing tomorrow") is False

    def test_normal_conversation_not_detected(self) -> None:
        assert _looks_like_benchmark("Hello, how are you?") is False

    def test_empty_query(self) -> None:
        assert _looks_like_benchmark("") is False

    def test_none_query(self) -> None:
        assert _looks_like_benchmark(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Benchmark detection — memory/tool layer
# ---------------------------------------------------------------------------


class TestLooksLikeBenchmark:
    """_looks_like_benchmark in memory.py."""

    def test_tokyo_benchmark_signal(self) -> None:
        assert _looks_like_benchmark(
            "plan a 3-day trip to tokyo for a first-time visitor"
        ) is True

    def test_capital_france_signal(self) -> None:
        assert _looks_like_benchmark("what is the capital of france") is True

    def test_normal_fact_not_flagged(self) -> None:
        assert _looks_like_benchmark(
            "The user prefers short answers"
        ) is False

    def test_empty_text(self) -> None:
        assert _looks_like_benchmark("") is False

    def test_none_text(self) -> None:
        assert _looks_like_benchmark(None) is False  # type: ignore[arg-type]

    def test_all_signals_are_lowercase(self) -> None:
        """All benchmark signals should be lowercase for reliable matching."""
        for signal in _BENCHMARK_SIGNALS:
            assert signal == signal.lower(), f"Signal not lowercase: {signal!r}"
