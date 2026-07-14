"""Streaming output test — verifies astream_events() works end-to-end.

Requires a running LLM API (uses MODEL env var).  Not run as part of the
default unit test suite.
"""

from __future__ import annotations

import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

import asyncio

from dotenv import load_dotenv

load_dotenv()

from react_agent.stream import stream_events, stream_tokens


async def test_stream_events_simple_query() -> None:
    """stream_events() should yield at least one event."""
    events = []
    async for event in stream_events("What is 2+2? Answer in one word."):
        events.append(event)

    # We should have at least some events (chain starts, LLM streams, chain ends)
    assert len(events) > 0, "No events received"
    print(f"  → Total events: {len(events)}")


async def test_stream_tokens_yields_text() -> None:
    """stream_tokens() should yield text tokens."""
    tokens = []
    async for token in stream_tokens("Say 'hello world' and nothing else."):
        tokens.append(token)

    assert len(tokens) > 0, "No tokens received"
    full_text = "".join(tokens)
    print(f"  → Streamed text ({len(tokens)} tokens): {full_text[:100]}")


async def test_stream_tokens_planning_query() -> None:
    """stream_tokens() should work with a multi-step query too."""
    tokens = []
    async for token in stream_tokens(
        "What is the capital of France? Just say the city name."
    ):
        tokens.append(token)

    full_text = "".join(tokens)
    print(f"  → Streamed text ({len(tokens)} tokens): {full_text[:200]}")
    assert len(full_text) > 0


async def main() -> None:
    """Run all streaming tests and report results."""
    print("=" * 60)
    print("Streaming Output Tests")
    print("=" * 60)
    print(f"MODEL: {os.environ.get('MODEL', 'not set')}")
    print()

    tests = [
        ("stream_events simple query", test_stream_events_simple_query),
        ("stream_tokens yields text", test_stream_tokens_yields_text),
        ("stream_tokens planning query", test_stream_tokens_planning_query),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"▶ {name}...")
        try:
            await test_fn()
            print("  ✓ PASS\n")
            passed += 1
        except Exception as exc:
            print(f"  ✗ FAIL: {exc}\n")
            failed += 1

    print(f"{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
