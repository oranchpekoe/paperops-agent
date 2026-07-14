"""Reflection agent subgraph.

Implements the *Generate → Critique → Refine* loop:

1. **generate** — produce an initial answer.
2. **reflect** — the model critiques its own answer, listing issues.
3. **refine** — the answer is rewritten addressing every critique.
4. Loop back to (2) until the critique signals "no further issues"
   or the maximum iteration count (3) is reached.

This mode shines for writing, analysis, code review, and any task where
the first answer is likely imperfect.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import MainState
from react_agent.utils import load_chat_model, resolve_model

logger = logging.getLogger(__name__)
_trace = logging.getLogger("trace")

MAX_REFLECTION_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATE_PROMPT = """You are a thoughtful AI assistant.  Give a thorough, well-reasoned answer to the user's query.  Take your time and be detailed."""

REFLECT_PROMPT = """You are a strict quality reviewer.  Critically examine the following response.

Evaluate it on:
1. **Factual accuracy** — are there any errors or hallucinations?
2. **Completeness** — what important information is missing?
3. **Clarity** — is anything confusing or poorly explained?
4. **Logic** — are there any reasoning flaws?

Be harsh and specific.  List every issue you find.

If the response is already excellent with no meaningful issues, start your reply with the exact phrase: PASS

Otherwise, start your reply with: ISSUES_FOUND
Then list each issue with a brief explanation.

Response to review:
---
{draft}
---"""

REFINE_PROMPT = """You are a meticulous editor.  Rewrite the response below, addressing EVERY issue raised in the critique.

Requirements:
- Fix all factual errors.
- Fill in missing information.
- Improve clarity and flow.
- Strengthen the logical reasoning.
- Keep the same overall structure unless the critique demands changes.

Original response:
---
{draft}
---

Critique:
---
{critique}
---

Produce the improved response now (no preamble, just the response):"""

# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def _generate(
    state: MainState, runtime: Runtime[Context]
) -> Dict:
    """Produce the initial draft."""
    _trace.info("📍 [3/5] Reflection._generate — 生成初始回答")
    model = load_chat_model(resolve_model(runtime))

    response = await model.ainvoke([
        SystemMessage(content=GENERATE_PROMPT),
        HumanMessage(content=state.user_query),
    ])

    _trace.info("   → 生成完成，回复长度: %d 字符", len(str(response.content)))

    return {
        "messages": [response],
        "reflection_iteration": 0,
    }


async def _reflect(
    state: MainState, runtime: Runtime[Context]
) -> Dict:
    """Critique the current draft."""
    _trace.info("   → Reflection._reflect — 审视第 %d 轮", state.reflection_iteration + 1)
    model = load_chat_model(resolve_model(runtime))

    # The last AIMessage is the current draft
    draft = ""
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            draft = str(msg.content)
            break

    critique_prompt = REFLECT_PROMPT.format(draft=draft)

    # Small delay between iterations to avoid rate limits on free tiers
    if state.reflection_iteration > 0:
        await asyncio.sleep(3)

    try:
        response = await model.ainvoke([
            SystemMessage(content=critique_prompt),
            HumanMessage(content="Please review the response above."),
        ])
    except Exception as e:
        logger.warning("Reflect call failed (likely rate-limited): %s — ending reflection early", e)
        # Return a "PASS" message so the router will exit the loop gracefully
        return {
            "messages": [AIMessage(content="PASS — rate limit encountered, using current draft.")],
        }

    return {
        "messages": [response],
    }


def _route_reflect(state: MainState) -> Literal["refine", "__end__"]:
    """Decide whether to refine or finish."""
    # Check iteration limit
    if state.reflection_iteration >= MAX_REFLECTION_ITERATIONS:
        _trace.info("📍 [4/5] Reflection 达到最大迭代数 %d → __end__", MAX_REFLECTION_ITERATIONS)
        return "__end__"

    # Parse the last critique to see if it says PASS
    last_msg = state.messages[-1]
    if isinstance(last_msg, AIMessage):
        text = str(last_msg.content).strip().upper()
        # Check for PASS signal
        if text.startswith("PASS") or re.search(
            r"\b(PASS|NO\s*ISSUES?|NO\s*IMPROVEMENT|LOOKS?\s*GOOD|WELL\s*DONE|EXCELLENT)\b",
            text,
        ):
            _trace.info("📍 [4/5] Reflection 审视通过（PASS）→ __end__")
            return "__end__"

    _trace.info("   → 有问题需要修改 → 路由到 _refine")
    return "refine"


async def _refine(
    state: MainState, runtime: Runtime[Context]
) -> Dict:
    """Improve the draft based on the critique."""
    _trace.info("   → Reflection._refine — 根据审视意见修改 (迭代 #%d)", state.reflection_iteration + 1)
    model = load_chat_model(resolve_model(runtime))

    # Find the draft (second-to-last AIMessage before the critique)
    messages_list = list(state.messages)
    draft = ""
    critique = ""

    # Walk backwards: critique is the last AIMessage, draft is the one before it
    found_critique = False
    for msg in reversed(messages_list):
        if isinstance(msg, AIMessage):
            if not found_critique:
                critique = str(msg.content)
                found_critique = True
            else:
                draft = str(msg.content)
                break

    refine_prompt = REFINE_PROMPT.format(draft=draft, critique=critique)

    try:
        response = await model.ainvoke([
            SystemMessage(content=refine_prompt),
            HumanMessage(content="Please produce the improved response."),
        ])
    except Exception as e:
        logger.warning("Refine call failed (likely rate-limited): %s — keeping current draft", e)
        # Return the current draft unchanged so the user at least sees something
        return {
            "messages": [AIMessage(content=draft)],
            "reflection_iteration": MAX_REFLECTION_ITERATIONS,  # force exit
        }

    new_iteration = state.reflection_iteration + 1

    return {
        "messages": [response],
        "reflection_iteration": new_iteration,
    }


# ---------------------------------------------------------------------------
# Subgraph factory
# ---------------------------------------------------------------------------


def build_reflection_subgraph() -> StateGraph:
    """Create a compiled Reflection subgraph."""
    builder = StateGraph(MainState)

    builder.add_node("generate", _generate)
    builder.add_node("reflect", _reflect)
    builder.add_node("refine", _refine)

    builder.add_edge("__start__", "generate")
    builder.add_edge("generate", "reflect")
    builder.add_conditional_edges("reflect", _route_reflect, {
        "refine": "refine",
        "__end__": "__end__",
    })
    builder.add_edge("refine", "reflect")

    return builder.compile(name="Reflection")
