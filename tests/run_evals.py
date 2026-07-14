r"""Evaluation runner for the Multi-Mode Agent Framework.

Runs every benchmark defined in :file:`benchmarks.py` against the live agent,
scores the results across four dimensions, and prints a formatted report.

Usage
-----
.. code-block:: bash

    # Run all benchmarks
    python tests/run_evals.py

    # Run a single category
    python tests/run_evals.py --category routing

    # Output JSON report (for CI / dashboards)
    python tests/run_evals.py --json

Scoring dimensions
------------------
* **Route** (30 %) — did the mode router pick the expected mode?
* **Tools** (25 %) — were expected tools called and forbidden tools avoided?
* **Quality** (25 %) — do expected keywords appear in the final answer?
* **Depth** (20 %) — does the message count indicate sufficient reasoning?
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage

from react_agent.graph import graph
from react_agent.memory import set_benchmark_mode  # ContextVar — disables remember tool

# ---------------------------------------------------------------------------
# Quiet mode — suppress HTTP noise during evals
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(message)s",
    stream=sys.stdout,
)
# Keep our own logger
_log = logging.getLogger("evals")
_log.setLevel(logging.INFO)
# Also need to let the graph's trace logger work at WARNING level
logging.getLogger("trace").setLevel(logging.WARNING)

from benchmarks import BENCHMARKS, BenchmarkCase  # noqa: E402

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Per-benchmark evaluation result."""

    case: BenchmarkCase
    passed: bool = False
    score: float = 0.0  # 0–100

    # Raw observations
    actual_mode: str = ""
    tools_called: list[str] = field(default_factory=list)
    tools_forbidden_called: list[str] = field(default_factory=list)
    final_answer: str = ""
    message_count: int = 0
    latency_seconds: float = 0.0

    # Dimension scores
    score_route: float = 0.0
    score_tools: float = 0.0
    score_quality: float = 0.0
    score_depth: float = 0.0

    # Detail for debugging
    error: str = ""


@dataclass
class EvalReport:
    """Aggregate report across all benchmarks."""

    results: list[EvalResult] = field(default_factory=list)
    total_score: float = 0.0
    total_passed: int = 0
    total_failed: int = 0
    total_time: float = 0.0

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.total_passed / len(self.results) * 100

    def by_category(self) -> dict[str, list[EvalResult]]:
        cats: dict[str, list[EvalResult]] = {}
        for r in self.results:
            cats.setdefault(r.case.category, []).append(r)
        return cats


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

ROUTE_WEIGHT = 0.30
TOOLS_WEIGHT = 0.25
QUALITY_WEIGHT = 0.25
DEPTH_WEIGHT = 0.20
PASS_THRESHOLD = 60.0  # a benchmark "passes" at 60 %


def _score_route(case: BenchmarkCase, actual_mode: str) -> float:
    """Score mode routing — exact match = 100, otherwise 0."""
    return 100.0 if actual_mode == case.expected_mode else 0.0


def _score_tools(case: BenchmarkCase, tools_called: list[str], forbidden_called: list[str]) -> float:
    """Score tool usage.

    - Each expected tool called: +N points
    - Each forbidden tool called: −N points
    - No expected tools specified: full score if no forbidden tools called
    """
    if not case.expected_tools and not case.forbidden_tools:
        return 100.0  # no tool expectations → skip

    score = 100.0

    # Penalise forbidden tools
    if case.forbidden_tools:
        penalty_per_forbidden = 100.0 / max(len(case.forbidden_tools), 1)
        for t in forbidden_called:
            # Allow search if it was also in expected_tools
            if t not in case.expected_tools:
                score -= penalty_per_forbidden

    # Reward expected tools (only matters if some are expected)
    if case.expected_tools:
        hit_count = sum(1 for t in case.expected_tools if t in tools_called)
        tool_score = (hit_count / len(case.expected_tools)) * 100.0
        # Blend: 50 % expected-tool coverage, 50 % no-forbidden penalty
        score = (tool_score + max(0.0, score)) / 2.0

    return max(0.0, min(100.0, score))


def _score_quality(case: BenchmarkCase, final_answer: str) -> float:
    """Score answer quality via keyword matching.

    ``expected_keywords``: at least 60 % must match.
    ``expected_keywords_any``: at least 1 must match (if specified).
    """
    answer_lower = final_answer.lower()

    # Mandatory keywords
    if case.expected_keywords:
        hits = sum(1 for kw in case.expected_keywords if kw.lower() in answer_lower)
        mandatory_score = (hits / len(case.expected_keywords)) * 100.0
    else:
        mandatory_score = 100.0

    # "Any" keywords
    if case.expected_keywords_any:
        any_hit = any(kw.lower() in answer_lower for kw in case.expected_keywords_any)
        any_score = 100.0 if any_hit else 0.0
    else:
        any_score = 100.0

    return (mandatory_score + any_score) / 2.0


def _score_depth(case: BenchmarkCase, message_count: int) -> float:
    """Score reasoning depth — more messages = more tool/reasoning steps."""
    if message_count >= case.min_messages:
        return 100.0
    return (message_count / case.min_messages) * 100.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_one(case: BenchmarkCase) -> EvalResult:
    """Execute a single benchmark case and score it."""
    result = EvalResult(case=case)
    _log.info("  ▶ %s", case.id)

    t0 = time.perf_counter()

    try:
        # Enable benchmark isolation at two layers:
        # 1. State.benchmark_mode → graph nodes (extract_memory) skip storage
        # 2. ContextVar → tool functions (remember, extract_facts) reject storage
        set_benchmark_mode(True)
        raw = await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": case.query}],
                "benchmark_mode": True,
            },
            config={"configurable": {"thread_id": f"eval-{case.id}"}},
        )
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.latency_seconds = time.perf_counter() - t0
        _log.error("    ✗ ERROR: %s", result.error)
        return result

    result.latency_seconds = time.perf_counter() - t0
    messages = raw.get("messages", [])

    # ── Extract observations ────────────────────────────────────────
    result.actual_mode = raw.get("mode", "unknown")
    result.message_count = len(messages)

    # Collect tool calls from all AIMessages
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "unknown")
                result.tools_called.append(tool_name)

    # Forbidden tools check
    result.tools_forbidden_called = [
        t for t in result.tools_called
        if t in case.forbidden_tools and t not in case.expected_tools
    ]

    # Final answer = last message content
    if messages:
        last = messages[-1]
        result.final_answer = str(last.content) if hasattr(last, "content") else str(last)

    # ── Score ────────────────────────────────────────────────────────
    result.score_route = _score_route(case, result.actual_mode)
    result.score_tools = _score_tools(case, result.tools_called, result.tools_forbidden_called)
    result.score_quality = _score_quality(case, result.final_answer)
    result.score_depth = _score_depth(case, result.message_count)

    result.score = (
        ROUTE_WEIGHT * result.score_route
        + TOOLS_WEIGHT * result.score_tools
        + QUALITY_WEIGHT * result.score_quality
        + DEPTH_WEIGHT * result.score_depth
    )
    result.passed = result.score >= PASS_THRESHOLD

    # ── One-line log ─────────────────────────────────────────────────
    status = "✓" if result.passed else "✗"
    _log.info(
        "    %s  score=%5.1f%%  route=%s→%s  tools=%s  msgs=%d  %.1fs",
        status,
        result.score,
        case.expected_mode,
        result.actual_mode,
        result.tools_called or "(none)",
        result.message_count,
        result.latency_seconds,
    )
    if not result.passed:
        _log.info("         route=%.0f tools=%.0f quality=%.0f depth=%.0f",
                  result.score_route, result.score_tools,
                  result.score_quality, result.score_depth)

    return result


async def run_all(category_filter: Optional[str] = None) -> EvalReport:
    """Run all (or filtered) benchmarks sequentially and return a report."""
    cases = BENCHMARKS
    if category_filter:
        cases = [c for c in BENCHMARKS if c.category == category_filter]

    _log.info("=" * 60)
    _log.info("Multi-Mode Agent Framework — Evaluation Suite")
    _log.info("Benchmarks: %d  |  Pass threshold: %.0f%%", len(cases), PASS_THRESHOLD)
    _log.info("=" * 60)

    results: list[EvalResult] = []
    t0 = time.perf_counter()

    for case in cases:
        result = await run_one(case)
        results.append(result)

    total_time = time.perf_counter() - t0
    report = EvalReport(
        results=results,
        total_score=sum(r.score for r in results) / len(results) if results else 0.0,
        total_passed=sum(1 for r in results if r.passed),
        total_failed=sum(1 for r in results if not r.passed),
        total_time=total_time,
    )

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def print_report(report: EvalReport) -> None:
    """Print a human-readable evaluation report."""
    print(f"\n{'=' * 70}")
    print("📊 EVALUATION REPORT")
    print(f"{'=' * 70}")
    print(f"Total benchmarks:  {len(report.results)}")
    print(f"Passed:            {report.total_passed}  ✓")
    print(f"Failed:            {report.total_failed}  ✗")
    print(f"Pass rate:         {report.pass_rate:.1f} %")
    print(f"Average score:     {report.total_score:.1f} %")
    print(f"Total wall time:   {report.total_time:.1f} s")
    print()

    # By category
    by_cat = report.by_category()
    if by_cat:
        print(f"{'Category':<20} {'Count':>5} {'Passed':>7} {'Avg Score':>10}")
        print("-" * 44)
        for cat, results in sorted(by_cat.items()):
            passed = sum(1 for r in results if r.passed)
            avg = sum(r.score for r in results) / len(results) if results else 0
            print(f"{cat:<20} {len(results):>5} {passed:>7} {avg:>9.1f}%")

    # Detail table
    print(f"\n{'─' * 90}")
    print(f"{'ID':<35} {'Category':<12} {'Mode':<12} {'Score':>6} {'Result'}")
    print(f"{'─' * 90}")
    for r in report.results:
        mode_str = f"{r.case.expected_mode}→{r.actual_mode}"
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(
            f"{r.case.id:<35} {r.case.category:<12} {mode_str:<12} {r.score:>5.1f}% {status}"
        )
        if not r.passed:
            _print_failure_detail(r)

    print(f"{'─' * 90}")

    # Summary verdict
    if report.pass_rate >= 80:
        verdict = "✅ READY — Agent passes most benchmarks."
    elif report.pass_rate >= 60:
        verdict = "⚠️  NEEDS WORK — Several benchmarks failed, review the details."
    else:
        verdict = "❌ FAILING — Significant gaps, investigate before presenting."

    print(f"\n{verdict}\n")


def _print_failure_detail(r: EvalResult) -> None:
    """Print why a benchmark failed."""
    reasons: list[str] = []
    if r.score_route < 100:
        reasons.append(f"route mismatch (expected {r.case.expected_mode}, got {r.actual_mode})")
    if r.score_tools < 100:
        if r.tools_forbidden_called:
            reasons.append(f"forbidden tools called: {r.tools_forbidden_called}")
        if r.case.expected_tools and not any(t in r.tools_called for t in r.case.expected_tools):
            reasons.append(f"expected tools not called: {r.case.expected_tools}")
    if r.score_quality < 50:
        reasons.append(f"keywords missing (answer: {r.final_answer[:80]}...)")
    if r.score_depth < 100:
        reasons.append(f"shallow ({r.message_count} msgs, expected ≥{r.case.min_messages})")
    if r.error:
        reasons.append(f"error: {r.error}")

    for reason in reasons:
        print(f"         ↳ {reason}")


def print_json_report(report: EvalReport) -> None:
    """Print the report as a JSON object (suitable for CI pipelines)."""
    out: dict = {
        "total_benchmarks": len(report.results),
        "passed": report.total_passed,
        "failed": report.total_failed,
        "pass_rate": round(report.pass_rate, 1),
        "average_score": round(report.total_score, 1),
        "total_time_seconds": round(report.total_time, 1),
        "results": [],
    }
    for r in report.results:
        out["results"].append({
            "id": r.case.id,
            "category": r.case.category,
            "passed": r.passed,
            "score": round(r.score, 1),
            "actual_mode": r.actual_mode,
            "expected_mode": r.case.expected_mode,
            "tools_called": r.tools_called,
            "message_count": r.message_count,
            "latency_seconds": round(r.latency_seconds, 1),
            "error": r.error or None,
        })
    print(json.dumps(out, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(category: Optional[str] = None, json_output: bool = False) -> None:
    report = await run_all(category_filter=category)

    if json_output:
        print_json_report(report)
    else:
        print_report(report)

    # Exit code for CI
    if report.pass_rate < 60:
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the Multi-Mode Agent Framework.")
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="Run only benchmarks in this category (routing, tool_use, quality, multi_step, memory).",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON (for CI / dashboards).",
    )
    args = parser.parse_args()
    asyncio.run(main(category=args.category, json_output=args.json))
