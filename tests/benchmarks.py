"""Benchmark dataset for the Multi-Mode Agent Framework.

Each case defines a query and the expected behaviour — mode routing,
tool usage, and answer content.  The eval runner (:file:`run_evals.py`)
executes every case and scores the agent against these expectations.

Adding a new benchmark
----------------------
1. Add a ``BenchmarkCase`` to the ``BENCHMARKS`` list.
2. Set ``category`` to one of: ``routing``, ``tool_use``, ``quality``,
   ``multi_step``, ``memory``.
3. Run ``python tests/run_evals.py`` to see if the agent passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkCase:
    """A single evaluation case for the agent."""

    id: str
    """Short kebab-case identifier (e.g. ``simple-fact``)."""

    query: str
    """The user query to send to the agent."""

    category: str
    """What aspect this case tests:
    ``routing``, ``tool_use``, ``quality``, ``multi_step``, ``memory``."""

    expected_mode: str
    """Which mode the router should select:
    ``react`` | ``reflection`` | ``plan_solve`` | ``supervisor``."""

    expected_tools: list[str] = field(default_factory=list)
    """Tools the agent should call (empty list = no tools expected)."""

    forbidden_tools: list[str] = field(default_factory=list)
    """Tools the agent must NOT call (e.g. search on a math question)."""

    expected_keywords: list[str] = field(default_factory=list)
    """Keywords or phrases that should appear in the final answer.
    At least 60% must match for the quality check to pass."""

    expected_keywords_any: list[str] = field(default_factory=list)
    """Alternative keywords — at least ONE from this list must appear.
    Useful for open-ended questions where the exact phrasing varies."""

    min_messages: int = 2
    """Minimum total messages expected (proxy for multi-step reasoning).
    Set > 2 for cases that should involve tool calls or reflection cycles."""

    description: str = ""
    """Human-readable explanation of what this case validates."""


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

BENCHMARKS: list[BenchmarkCase] = [
    # ── Category: routing ──────────────────────────────────────────────
    BenchmarkCase(
        id="routing-simple-fact",
        query="What is the capital of France?",
        category="routing",
        expected_mode="react",
        expected_tools=[],
        expected_keywords=["Paris"],
        min_messages=2,
        description="Simple factual lookup → react, no tools needed.",
    ),
    BenchmarkCase(
        id="routing-writing",
        query="Write a short analysis: is Python good for AI development?",
        category="routing",
        expected_mode="reflection",
        expected_tools=[],
        expected_keywords=["Python", "AI", "library", "machine learning", "deep learning"],
        expected_keywords_any=["AI", "artificial intelligence", "machine learning"],
        min_messages=3,
        description="Writing/analysis task → reflection with self-critique.",
    ),
    BenchmarkCase(
        id="routing-planning",
        query="Plan a 3-day trip to Tokyo for a first-time visitor.",
        category="routing",
        expected_mode="plan_solve",
        expected_tools=[],
        expected_keywords=["Tokyo", "day", "visit", "itinerary"],
        expected_keywords_any=["Tokyo", "Japan"],
        min_messages=2,
        description="Multi-step planning → plan_solve with decomposition.",
    ),
    BenchmarkCase(
        id="routing-multi-domain",
        query="Research the GDP of Japan in 2024 and calculate what 5% of it is.",
        category="routing",
        expected_mode="supervisor",
        expected_tools=["search"],
        expected_keywords=["GDP", "Japan", "2024", "5%", "trillion", "billion"],
        expected_keywords_any=["GDP", "Japan", "5%", "percent"],
        min_messages=5,
        description="Search + calculation → supervisor with multi-specialist delegation.",
    ),
    BenchmarkCase(
        id="routing-math-word-problem",
        query="A train leaves Station A at 60 km/h. Another train leaves Station B, 300 km away, at 90 km/h toward Station A. When do they meet?",
        category="routing",
        expected_mode="plan_solve",
        expected_tools=["python_repl"],
        expected_keywords=["hour", "km", "time"],
        expected_keywords_any=["meet", "time", "hour"],
        min_messages=3,
        description="Math word problem → plan_solve with python_repl tool use.",
    ),

    # ── Category: tool_use ─────────────────────────────────────────────
    BenchmarkCase(
        id="tool-search-current-events",
        query="Who won the most recent FIFA World Cup before 2026?",
        category="tool_use",
        expected_mode="react",
        expected_tools=["search"],
        forbidden_tools=["python_repl"],
        expected_keywords_any=["Argentina", "2022", "World Cup"],
        min_messages=3,
        description="Current information → must use search tool.",
    ),
    BenchmarkCase(
        id="tool-calculation",
        query="Calculate the compound interest on $10,000 at 5% annual rate over 10 years, compounded monthly.",
        category="tool_use",
        expected_mode="plan_solve",  # Complex math → plan_solve after router prompt refinement
        expected_tools=["python_repl"],
        forbidden_tools=["search"],
        expected_keywords=["16", "interest"],
        expected_keywords_any=["16470", "16470.09", "6470", "16,470"],
        min_messages=3,
        description="Numeric calculation → plan_solve with python_repl, not search.",
    ),
    BenchmarkCase(
        id="tool-no-unnecessary-search",
        query="What is 2 + 2?",
        category="tool_use",
        expected_mode="react",
        expected_tools=[],
        forbidden_tools=["search"],
        expected_keywords=["4", "four"],
        min_messages=2,
        description="Trivial question → no tools should be called.",
    ),

    # ── Category: quality ──────────────────────────────────────────────
    BenchmarkCase(
        id="quality-code-review",
        query="Review this code for bugs: `def divide(a, b): return a / b`",
        category="quality",
        expected_mode="reflection",
        expected_tools=[],
        expected_keywords=["zero", "division", "error", "check", "exception", "handle"],
        expected_keywords_any=["zero", "division by zero", "ZeroDivisionError"],
        min_messages=3,
        description="Code review → must mention division-by-zero issue.",
    ),
    BenchmarkCase(
        id="quality-structured-answer",
        query="Compare Python and JavaScript for web development in 3-4 bullet points.",
        category="quality",
        expected_mode="reflection",
        expected_tools=[],
        expected_keywords=["Python", "JavaScript"],
        expected_keywords_any=["Python", "JavaScript", "framework"],
        min_messages=3,
        description="Structured comparison → answer should cover both languages.",
    ),

    # ── Category: multi_step ───────────────────────────────────────────
    BenchmarkCase(
        id="multi-search-then-calculate",
        query="Search for the current population of China, then calculate what 1% of that number is.",
        category="multi_step",
        expected_mode="supervisor",
        expected_tools=["search", "python_repl"],
        expected_keywords_any=["billion", "million", "population"],
        min_messages=5,
        description="Search → calculate pipeline → supervisor with researcher + executor.",
    ),
    BenchmarkCase(
        id="multi-decomposition",
        query="I want to start learning machine learning. Create a 4-week study plan for a Python programmer.",
        category="multi_step",
        expected_mode="plan_solve",
        expected_tools=[],
        expected_keywords=["week", "Python", "learn", "machine learning"],
        expected_keywords_any=["week", "study plan", "curriculum", "schedule"],
        min_messages=2,
        description="Educational plan → plan_solve with step decomposition.",
    ),

    # ── Category: memory ───────────────────────────────────────────────
    BenchmarkCase(
        id="memory-explicit-remember",
        query="Remember this: I am a dual-degree master's student in computer science and finance.",
        category="memory",
        expected_mode="react",
        expected_tools=["remember"],
        expected_keywords_any=["Stored", "memory", "remember"],
        min_messages=2,
        description="Explicit memory → agent should call the remember tool.",
    ),
    BenchmarkCase(
        id="memory-recall-test",
        query="What degree am I pursuing?",
        category="memory",
        expected_mode="react",
        expected_tools=["recall"],
        expected_keywords_any=["master", "dual", "computer science", "finance"],
        min_messages=2,
        description="Memory recall → agent should use recall tool to find stored facts.",
    ),
]

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def summary() -> dict:
    """Return aggregate counts by category."""
    cats: dict[str, int] = {}
    for b in BENCHMARKS:
        cats[b.category] = cats.get(b.category, 0) + 1
    return {
        "total": len(BENCHMARKS),
        "by_category": cats,
        "expected_modes": sorted(set(b.expected_mode for b in BENCHMARKS)),
    }


if __name__ == "__main__":
    s = summary()
    print(f"Total benchmarks: {s['total']}")
    print(f"By category: {s['by_category']}")
    print(f"Expected modes: {s['expected_modes']}")
    for b in BENCHMARKS:
        print(f"  [{b.category}] {b.id}: {b.description}")
