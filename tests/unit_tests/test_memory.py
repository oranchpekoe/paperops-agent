"""Unit tests for memory module (benchmark detection + ContextVar logic).

Full Chroma-backed tests (store/recall/dedup) require a running embedding API
and Chroma installation.  They are marked with ``@pytest.mark.integration``
and skipped by default in unit test runs.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from react_agent.memory import (
    _BENCHMARK_SIGNALS,
    _benchmark_mode,
    _in_benchmark_mode,
    extract_facts,
    remember,
    set_benchmark_mode,
)


class TestBenchmarkModeContextVar:
    """ContextVar-based benchmark mode flag."""

    def setup_method(self) -> None:
        """Ensure benchmark mode is off before each test."""
        set_benchmark_mode(False)

    def test_default_is_false(self) -> None:
        assert _in_benchmark_mode() is False

    def test_set_to_true(self) -> None:
        set_benchmark_mode(True)
        assert _in_benchmark_mode() is True

    def test_set_to_false(self) -> None:
        set_benchmark_mode(True)
        assert _in_benchmark_mode() is True
        set_benchmark_mode(False)
        assert _in_benchmark_mode() is False

    def test_contextvar_token(self) -> None:
        """ContextVar should be set and readable in same context."""
        token = _benchmark_mode.set(True)
        try:
            assert _in_benchmark_mode() is True
        finally:
            _benchmark_mode.reset(token)
        assert _in_benchmark_mode() is False


class TestRememberToolBenchmarkGuard:
    """remember() tool rejects storage in benchmark mode."""

    def setup_method(self) -> None:
        set_benchmark_mode(False)

    def test_rejects_benchmark_signal(self) -> None:
        """Even without benchmark_mode, pattern-matched facts are rejected."""
        result = asyncio.run(remember(
            "plan a 3-day trip to tokyo for a first-time visitor"
        ))
        assert "⚠️" in result or "NOT stored" in result

    def test_rejects_in_benchmark_mode(self) -> None:
        """With benchmark_mode=True, even normal facts are rejected."""
        set_benchmark_mode(True)
        try:
            result = asyncio.run(remember("The user prefers short answers"))
            assert "NOT stored" in result
        finally:
            set_benchmark_mode(False)

    def test_benchmark_mode_trumps_all(self) -> None:
        """benchmark_mode=True blocks everything, regardless of content."""
        set_benchmark_mode(True)
        try:
            result = asyncio.run(remember("The sky is blue"))
            assert "NOT stored" in result
        finally:
            set_benchmark_mode(False)


class TestExtractFactsBenchmarkGuard:
    """extract_facts() skips extraction in benchmark mode."""

    def setup_method(self) -> None:
        set_benchmark_mode(False)

    def test_in_benchmark_mode_returns_empty(self) -> None:
        """In benchmark mode, extract_facts returns [] without calling LLM."""
        from unittest.mock import AsyncMock

        from langchain_core.messages import AIMessage, HumanMessage

        async def _run() -> None:
            set_benchmark_mode(True)
            try:
                mock_model = AsyncMock()
                messages = [
                    HumanMessage(content="I am a student"),
                    AIMessage(content="Got it, you're a student"),
                ]
                facts = await extract_facts(messages, mock_model, max_facts=5)
                assert facts == []
                # Model should NOT be called — benchmark_mode triggers early return
                mock_model.ainvoke.assert_not_called()
            finally:
                set_benchmark_mode(False)

        asyncio.run(_run())


class TestBenchmarkSignals:
    """Benchmark signal list validation."""

    def test_no_empty_signals(self) -> None:
        for signal in _BENCHMARK_SIGNALS:
            assert signal.strip(), "Empty signal found"

    def test_all_lowercase(self) -> None:
        for signal in _BENCHMARK_SIGNALS:
            assert signal == signal.lower(), f"Signal not lowercase: {signal!r}"

    def test_signal_count(self) -> None:
        """Should match benchmark signal count (at least 14)."""
        assert len(_BENCHMARK_SIGNALS) >= 14


# ---------------------------------------------------------------------------
# Integration tests (require Chroma + embedding API)
# These are skipped in default `pytest tests/unit_tests/` runs.
# Run with: pytest tests/unit_tests/test_memory.py -m integration -v
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMemoryStoreIntegration:
    """Tests that require a running embedding API and Chroma."""

    @pytest.fixture(autouse=True)
    def _check_env(self) -> None:
        """Skip if embedding API is not configured."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_store_and_recall(self) -> None:
        """Store a fact and recall it."""

        async def _run() -> None:
            from react_agent.memory import _ensure_memory_loaded

            store = await _ensure_memory_loaded()
            if store is None:
                pytest.skip("Memory store unavailable")

            # Store a unique fact
            fact = "pytest-test-user-likes-coffee-xyz"
            fact_id = await store.store(fact, dedup=False)
            assert fact_id

            # Recall it
            results = await store.recall("coffee preferences", k=5)
            contents = [r["content"] for r in results]
            assert any("coffee" in c for c in contents), f"Fact not found in: {contents}"

            # Clean up
            await store.delete(fact_id)

        asyncio.run(_run())

    def test_store_dedup(self) -> None:
        """Duplicate facts should be skipped when dedup=True."""

        async def _run() -> None:
            from react_agent.memory import _ensure_memory_loaded

            store = await _ensure_memory_loaded()
            if store is None:
                pytest.skip("Memory store unavailable")

            fact = "pytest-test-user-is-a-pilot-xyz"
            id1 = await store.store(fact, dedup=False)
            id2 = await store.store(fact, dedup=True)
            assert id1 == id2

            # Clean up
            await store.delete(id1)

        asyncio.run(_run())

    def test_clear_all(self) -> None:
        """clear_all should return count of removed docs."""

        async def _run() -> None:
            from react_agent.memory import _ensure_memory_loaded

            store = await _ensure_memory_loaded()
            if store is None:
                pytest.skip("Memory store unavailable")

            await store.store("pytest-temp-clear-test-fact", dedup=False)
            count = await store.clear_all()
            assert count >= 1

        asyncio.run(_run())

    def test_clear_contaminated(self) -> None:
        """clear_contaminated should remove matching patterns."""

        async def _run() -> None:
            from react_agent.memory import _ensure_memory_loaded

            store = await _ensure_memory_loaded()
            if store is None:
                pytest.skip("Memory store unavailable")

            await store.store("pytest-tokyo-trip-benchmark-test-xyz", dedup=False)
            removed = await store.clear_contaminated(["tokyo-trip-benchmark"])
            assert removed >= 1

        asyncio.run(_run())
