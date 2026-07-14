"""Test the three-layer memory system: short-term (checkpointer),
summary (compress_context), and long-term (Chroma recall/store).

Tests are split into:
- **Unit tests** (no API keys): pure logic, mock LLMs
- **Integration tests** (require API keys): real graph checkpointer behaviour
"""

from __future__ import annotations

import asyncio
import os

import pytest

from react_agent.memory import compress_context


# ============================================================================
# Unit tests — context compression (mock LLM, no API keys)
# ============================================================================


class TestCompressContext:
    """compress_context() — old messages → summary paragraph via LLM."""

    @staticmethod
    def _make_messages(count: int, prefix: str = "msg") -> list:
        """Build *count* alternating Human / AI messages for testing."""
        from langchain_core.messages import AIMessage, HumanMessage

        msgs = []
        for i in range(count):
            cls = HumanMessage if i % 2 == 0 else AIMessage
            msgs.append(cls(content=f"{prefix}-{i:03d}"))
        return msgs

    def test_below_threshold_returns_unchanged(self) -> None:
        """When len <= 20, the list comes back unchanged (no LLM call)."""
        msgs = self._make_messages(15)

        async def _run():
            # Create a mock model that should NOT be called
            from unittest.mock import AsyncMock

            mock_model = AsyncMock()
            result = await compress_context(msgs, mock_model)
            # Model should never be called — below threshold
            mock_model.ainvoke.assert_not_called()
            # List should be identical
            assert result is msgs

        asyncio.run(_run())

    def test_above_threshold_triggers_compression(self) -> None:
        """When len > 20, old messages are replaced by a summary."""
        msgs = self._make_messages(25)

        async def _run():
            from unittest.mock import AsyncMock

            from langchain_core.messages import AIMessage, SystemMessage

            mock_model = AsyncMock()
            mock_response = AIMessage(content="This is a compressed summary of the conversation.")
            mock_model.ainvoke.return_value = mock_response

            result = await compress_context(msgs, mock_model)
            # Model should be called exactly once
            mock_model.ainvoke.assert_called_once()
            # Result should be shorter than input
            assert len(result) < len(msgs), f"Expected compression: {len(result)} < {len(msgs)}"
            # First message should be a SystemMessage (the summary)
            assert isinstance(result[0], SystemMessage), f"Expected SystemMessage, got {type(result[0]).__name__}"
            assert "compressed summary" in str(result[0].content)
            # Last 10 messages should be unchanged
            assert len(result) == 11  # 1 summary + 10 recent

        asyncio.run(_run())

    def test_compression_preserves_recent_messages(self) -> None:
        """The most recent ``keep_last`` messages stay verbatim."""
        msgs = self._make_messages(30)
        keep_last = 8

        async def _run():
            from unittest.mock import AsyncMock

            from langchain_core.messages import AIMessage

            mock_model = AsyncMock()
            mock_model.ainvoke.return_value = AIMessage(content="Summary.")

            result = await compress_context(msgs, mock_model, keep_last=keep_last)
            # Result: [summary] + last 8 messages
            assert len(result) == 1 + keep_last
            # Last 8 messages should be identical to original
            for i in range(1, len(result)):
                orig = msgs[len(msgs) - keep_last + (i - 1)]
                assert result[i] is orig, f"Recent message #{i} should be same object reference"

        asyncio.run(_run())

    def test_compress_handles_model_error(self) -> None:
        """When the LLM call fails, compression still returns a usable list."""
        msgs = self._make_messages(25)

        async def _run():
            from unittest.mock import AsyncMock

            from langchain_core.messages import SystemMessage

            mock_model = AsyncMock()
            mock_model.ainvoke.side_effect = RuntimeError("API unavailable")

            result = await compress_context(msgs, mock_model)
            # Should NOT raise — graceful degradation
            assert len(result) == 11  # 1 fallback summary + 10 recent
            assert isinstance(result[0], SystemMessage)
            assert "earlier messages omitted" in str(result[0].content).lower()

        asyncio.run(_run())


# ============================================================================
# Integration tests — checkpointer + multi-turn (require API keys)
# ============================================================================


@pytest.mark.integration
class TestCheckpointerMultiTurn:
    """Multi-turn conversation with MemorySaver checkpointer.

    These tests require a working LLM (OPENAI_API_KEY or equivalent).
    Run with: pytest tests/test_memory_layers.py -m integration -v
    """

    @pytest.fixture(autouse=True)
    def _check_env(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_messages_accumulate_with_thread_id(self) -> None:
        """Messages within the same thread_id should accumulate across calls."""
        import uuid

        from langchain_core.messages import HumanMessage

        from react_agent.graph import graph

        thread_id = f"test-acc-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}

        async def _run():
            # Turn 1: ask a simple question
            result1 = await graph.ainvoke(
                {"messages": [HumanMessage(content="What is 2 + 2? Answer in one word.")]},
                config=config,
            )
            count1 = len(result1["messages"])
            assert count1 >= 2, f"Turn 1 should have at least 2 messages, got {count1}"

            # Turn 2: follow up (same thread)
            result2 = await graph.ainvoke(
                {"messages": [HumanMessage(content="Multiply that answer by 3. One number only.")]},
                config=config,
            )
            count2 = len(result2["messages"])
            # Messages should have accumulated: turn1_messages + turn2_new_messages
            assert count2 > count1, (
                f"Turn 2 ({count2}) should have more messages than turn 1 ({count1}) — "
                f"checkpointer should accumulate across calls"
            )

            # The final answer should contain "12" (knowing 2+2=4, ×3=12)
            final = str(result2["messages"][-1].content).lower()
            assert "12" in final, f"Expected '12' in final answer, got: {final!r}"

        asyncio.run(_run())

    def test_different_threads_are_isolated(self) -> None:
        """Different thread_ids should have independent state."""
        import uuid

        from langchain_core.messages import HumanMessage

        from react_agent.graph import graph

        thread_a = f"test-iso-a-{uuid.uuid4().hex[:8]}"
        thread_b = f"test-iso-b-{uuid.uuid4().hex[:8]}"

        async def _run():
            # Thread A: ask about a topic
            result_a = await graph.ainvoke(
                {"messages": [HumanMessage(content="Say 'hello' in French.")]},
                config={"configurable": {"thread_id": thread_a}},
            )
            # Should contain "bonjour" somewhere
            assert "bonjour" in str(result_a["messages"][-1].content).lower()

            # Thread B: ask something completely different
            result_b = await graph.ainvoke(
                {"messages": [HumanMessage(content="What is the capital of Japan? One word.")]},
                config={"configurable": {"thread_id": thread_b}},
            )
            # Thread B should NOT have thread A's messages
            b_messages_text = " ".join(
                str(m.content) for m in result_b["messages"]
            )
            assert "bonjour" not in b_messages_text.lower(), (
                "Thread B should not contain Thread A's conversation"
            )
            assert "tokyo" in str(result_b["messages"][-1].content).lower()

        asyncio.run(_run())

    def test_checkpointer_persists_state_across_calls(self) -> None:
        """Verify that state truly persists across multiple ainvoke calls."""
        import uuid

        from langchain_core.messages import HumanMessage

        from react_agent.graph import graph

        thread_id = f"test-persist-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}

        async def _run():
            result1 = await graph.ainvoke(
                {"messages": [HumanMessage(content="My name is Alice. What is 1+1?")]},
                config=config,
            )
            msgs1 = result1["messages"]
            assert "2" in str(msgs1[-1].content)
            assert any("Alice" in str(m.content) for m in msgs1)

            # Same thread — state should carry "Alice" forward
            result2 = await graph.ainvoke(
                {"messages": [HumanMessage(content="What is my name? One word only.")]},
                config=config,
            )
            final = str(result2["messages"][-1].content).lower()
            assert "alice" in final, f"Should remember name 'Alice', got: {final!r}"

        asyncio.run(_run())
