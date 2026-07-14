"""Plan-and-Solve agent subgraph.

Implements the *Plan → Execute → Aggregate* pattern:

1. **plan** — the LLM decomposes the user's problem into numbered steps.
2. **execute** — each step is dispatched to the LLM (with tool access).  Results
   accumulate across steps so later steps can reference earlier findings.
3. **aggregate** — all step results are combined into a polished final answer.

This mode is ideal for multi-step reasoning — math word problems, travel
planning, research synthesis, and any task with sequential dependencies.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import MainState
from react_agent.tools import TOOLS, _ensure_mcp_loaded, _mcp_tools, run_mini_react_loop
from react_agent.utils import load_chat_model, resolve_model

logger = logging.getLogger(__name__)
_trace = logging.getLogger("trace")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLAN_PROMPT = """You are a strategic planner.  Break down the user's problem into a clear, logical sequence of steps.

Rules:
- Output ONLY a numbered list of steps (one per line: "1. ...", "2. ...", etc.)
- Each step should be a single, actionable sub-task.
- **3–5 steps maximum.** Do NOT over-decompose — fewer, meatier steps work better.
- Steps should be ordered so each builds on the previous ones where needed.
- Do NOT solve the problem — just write the plan.
- Focus on the CORE aspects, not every possible detail.

User query:
{query}"""

EXECUTE_STEP_PROMPT = """You are executing step {step_num} of a multi-step plan.

Overall goal: {goal}

Full plan:
{plan_text}

Results from previous steps (truncated for relevance):
{previous_results}

Your task for this step:
{step_description}

Execute this step now.  You may use tools (search, python_repl) if they help.
**Always use python_repl for any numeric calculation** — do not do math in your head.
Provide a clear, detailed result.  If the step requires computation, show your work.
**Important:** Only call search if this step genuinely needs external/current information.
If this step can be answered from your own knowledge (except calculations), do NOT call search."""

AGGREGATE_PROMPT = """You are a synthesis expert.  Combine the following step-by-step results into one polished, comprehensive final answer.

Original question: {query}

Plan and results:
{plan_and_results}

Write a clear, well-structured final answer that addresses the original question completely.  Do not mention "Step 1", "Step 2", etc. — present the information naturally."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_plan(text: str) -> List[str]:
    """Extract numbered steps from LLM output."""
    steps: List[str] = []
    for line in text.strip().split("\n"):
        match = re.match(r"^\s*(\d+)[\.\)]\s*(.+)", line)
        if match:
            steps.append(match.group(2).strip())
    return steps if steps else [text.strip()]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def _plan(
    state: MainState, runtime: Runtime[Context]
) -> Dict:
    """Decompose the user query into an ordered list of steps."""
    _trace.info("📍 [3/5] PlanSolve._plan — 分解任务为步骤")
    model = load_chat_model(resolve_model(runtime))

    response = await model.ainvoke([
        SystemMessage(content=PLAN_PROMPT.format(query=state.user_query)),
        HumanMessage(content="Create a plan for the task described above."),
    ])

    steps = _parse_plan(str(response.content))
    _trace.info("   → 分解出 %d 个步骤: %s", len(steps), [s[:50] for s in steps])

    return {
        "plan_steps": steps,
        "current_step": 0,
        "step_results": [],
    }


async def _execute_all(
    state: MainState, runtime: Runtime[Context]
) -> Dict:
    """Execute every plan step sequentially, with tool access per step."""
    await _ensure_mcp_loaded()
    # Only core tools — memory tools (remember/recall) are intentionally
    # excluded during plan execution to avoid side effects like storing
    # facts mid-execution or wasting tokens on redundant recall calls.
    tools = list(TOOLS) + list(_mcp_tools)
    model = load_chat_model(resolve_model(runtime)).bind_tools(tools)

    plan_text = "\n".join(
        f"{i + 1}. {s}" for i, s in enumerate(state.plan_steps)
    )
    results: List[str] = []

    _trace.info("   → PlanSolve._execute_all — 开始执行 %d 个步骤", len(state.plan_steps))
    for i, step_desc in enumerate(state.plan_steps):
        # Brief pause between steps to avoid rate limits on free tiers
        if i > 0:
            await asyncio.sleep(3)

        _trace.info("      → 执行步骤 %d/%d: %s", i + 1, len(state.plan_steps), step_desc[:60])

        previous = "\n".join(
            f"Step {j + 1} result: {r}"
            for j, r in enumerate(results)
        ) or "(none — this is the first step)"

        prompt = EXECUTE_STEP_PROMPT.format(
            step_num=i + 1,
            goal=state.user_query,
            plan_text=plan_text,
            previous_results=previous,
            step_description=step_desc,
        )

        msgs = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Execute step {i + 1} as described."),
        ]

        # Mini ReAct loop — shared helper (tools.py)
        msgs = await run_mini_react_loop(model, tools, msgs, max_rounds=3)

        # Extract final answer from the loop result
        last = msgs[-1]
        results.append(str(last.content) if hasattr(last, "content") else str(last))

    return {
        "step_results": results,
        "current_step": len(state.plan_steps),
    }


async def _aggregate(
    state: MainState, runtime: Runtime[Context]
) -> Dict:
    """Synthesise all step results into a final answer."""
    _trace.info("📍 [4/5] PlanSolve._aggregate — 汇总所有步骤为最终答案")
    model = load_chat_model(resolve_model(runtime))

    plan_and_results = "\n\n".join(
        f"Step {i + 1}: {step}\nResult: {result}"
        for i, (step, result) in enumerate(
            zip(state.plan_steps, state.step_results)
        )
    )

    response = await model.ainvoke([
        SystemMessage(content=AGGREGATE_PROMPT.format(
            query=state.user_query,
            plan_and_results=plan_and_results,
        )),
        HumanMessage(content="Please synthesize the final answer."),
    ])

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Subgraph factory
# ---------------------------------------------------------------------------


def build_plan_solve_subgraph() -> StateGraph:
    """Create a compiled Plan-and-Solve subgraph."""
    builder = StateGraph(MainState)

    builder.add_node("plan", _plan)
    builder.add_node("execute_all", _execute_all)
    builder.add_node("aggregate", _aggregate)

    builder.add_edge("__start__", "plan")
    builder.add_edge("plan", "execute_all")
    builder.add_edge("execute_all", "aggregate")
    builder.add_edge("aggregate", "__end__")

    return builder.compile(name="PlanSolve")
