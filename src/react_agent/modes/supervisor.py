"""Supervisor multi-agent subgraph.

Implements a **Supervisor-Worker** pattern where a supervisor LLM analyses the
user's task, delegates to specialist agents, reviews their output, and either
delegates again or produces a final synthesised answer.

Specialists
----------
* **researcher** — LLM with tool access (biased toward ``search``).  Gathers
  facts, finds information, retrieves context.
* **analyst** — LLM *without* tool access.  Critiques, reasons, compares,
  and evaluates.  Pure cognitive work.
* **executor** — LLM with tool access (biased toward ``python_repl``).
  Runs calculations, processes data, produces concrete outputs.

Structured Output
-----------------
Supervisor decisions use ``with_structured_output()`` with a Pydantic model,
which guarantees the LLM returns a valid action + task pair rather than
free-form text.  This eliminates fragile text-parsing heuristics.  A fallback
path is preserved for models / providers that do not support structured output.

Loop guard
----------
At most ``MAX_SUPERVISOR_ITERATIONS`` specialist invocations are allowed.
After that the supervisor is forced to produce a final answer.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Dict, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

# ``with_structured_output()`` internally stores the parsed Pydantic model on an
# AIMessage.parsed field.  LangGraph's state serializer triggers a harmless
# PydanticSerializationUnexpectedValue warning for that field.  Suppress it so
# the logs stay readable.
warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic serializer warnings.*",
    category=UserWarning,
)

from react_agent.context import Context
from react_agent.state import MainState
from react_agent.tools import _ensure_mcp_loaded, get_all_tools, run_mini_react_loop
from react_agent.utils import load_chat_model, resolve_model

logger = logging.getLogger(__name__)
_trace = logging.getLogger("trace")

MAX_SUPERVISOR_ITERATIONS = 5

# ---------------------------------------------------------------------------
# Pydantic model — guarantees structured decisions from the supervisor LLM
# ---------------------------------------------------------------------------

class SupervisorDecision(BaseModel):
    """Structured decision from the supervisor.

    The LLM is forced to output valid JSON matching this schema, eliminating
    the need to parse free-form text with regex / keyword heuristics.
    """

    action: Literal["RESEARCH", "EXECUTE", "ANALYSE", "ANSWER"] = Field(
        description=(
            "RESEARCH: the task needs web search to find facts, data, or current info. "
            "EXECUTE: the task needs a calculation or code execution. "
            "ANALYSE: the task needs reasoning, critique, or evaluation of known info. "
            "ANSWER: sufficient information gathered — deliver the final answer now."
        )
    )
    task: str = Field(
        description=(
            "If delegating (RESEARCH/EXECUTE/ANALYSE): a clear, specific sub-task "
            "instruction for the specialist.  If answering (ANSWER): the polished "
            "final answer text addressing the user's original query."
        )
    )


# ---------------------------------------------------------------------------
# Prompts (simplified — the Pydantic schema handles output format)
# ---------------------------------------------------------------------------

SUPERVISOR_DECIDE_PROMPT = """You are a supervisor agent coordinating a team of specialists to solve the user's task.

Your team:
- RESEARCHER — searches the web for facts, data, current information
- EXECUTOR — runs calculations and code via python_repl
- ANALYST — reasons, critiques, and evaluates information (no tools)

Analyse the user's query and decide the FIRST action to take.  For simple
questions that you can answer directly without any specialist help, answer now.

User query:
{query}"""

RESEARCHER_PROMPT = """You are a **research specialist**.  Your job is to gather accurate, current information to help answer a larger task.

Use the search tool to find relevant information.  Be thorough — search for specific facts, numbers, and details.

Once you have gathered enough information, write a clear summary of your findings.  Cite specific data points.

Context — the overall task being solved:
{task}

Specific sub-task for you now:
{subtask}"""

ANALYST_PROMPT = """You are an **analysis specialist**.  Your job is to reason through information, identify patterns, evaluate options, and provide insightful analysis.

**Important:** You do NOT have access to tools.  Work with the information already gathered.

Context — the overall task being solved:
{task}

Information gathered so far:
{context}

Specific analysis to perform now:
{subtask}

Provide a clear, well-reasoned analysis.  Be critical — point out gaps or uncertainties if they exist."""

EXECUTOR_PROMPT = """You are an **execution specialist**.  Your job is to perform computations, run calculations, process data, or generate structured output using the python_repl tool.

Use the python_repl tool whenever a calculation is needed.  Show your work — explain what you're computing and why.

Context — the overall task being solved:
{task}

Information available from previous steps:
{context}

Specific execution task:
{subtask}

Produce a clear, concrete result.  Include numbers, code outputs, or structured data as appropriate."""

SUPERVISOR_REVIEW_PROMPT = """You are reviewing progress on a multi-step task as the supervisor.

Original query:
{query}

Work completed by your team so far:
{history}

Decide the NEXT step.  If all necessary information has been gathered and any
required calculations have been performed, deliver the final ANSWER now.
Otherwise, delegate to the appropriate specialist."""

# ---------------------------------------------------------------------------
# Structured output helper (with text-parsing fallback)
# ---------------------------------------------------------------------------


async def _get_decision(
    model: BaseChatModel,
    messages: list,
    *,
    default_action: str = "FINISH",
) -> SupervisorDecision:
    """Call model with structured output, falling back to text parsing.

    Parameters
    ----------
    model : BaseChatModel
        The raw chat model (before ``with_structured_output``).
    messages : list
        Prompt messages to send.
    default_action : str
        Action to use when both structured output and text parsing fail.

    Returns:
    -------
    SupervisorDecision
        Always returns a valid decision — never ``None``.
    """
    # ── primary path: structured output (Pydantic schema) ──────────────
    try:
        structured = model.with_structured_output(SupervisorDecision)
        decision = await structured.ainvoke(messages)
        if isinstance(decision, SupervisorDecision):
            _trace.info("   → 结构化输出: action=%s", decision.action)
            return decision
    except Exception as exc:
        logger.debug("Structured output failed, falling back to text parsing: %s", exc)

    # ── fallback path: text-based parsing (legacy) ─────────────────────
    try:
        response = await model.ainvoke(messages)
        return _parse_text_decision(str(response.content))
    except Exception as exc:
        logger.warning("Text fallback also failed: %s — using default %s", exc, default_action)
        return SupervisorDecision(action="ANSWER", task="I wasn't able to process this request. Could you rephrase?")


def _parse_text_decision(raw: str) -> SupervisorDecision:
    r"""Parse free-text supervisor output into a ``SupervisorDecision``.

    Handles the old prompt format (e.g. ``RESEARCH\\nSearch for GDP data``)
    as well as unstructured prose that contains keywords.
    """
    raw_lower = raw.lower().strip()
    first_line = raw.strip().split("\n")[0].strip().upper()

    # Layer 1: exact prefix on first line
    if first_line.startswith("RESEARCH"):
        task = raw.strip().split("\n", 1)[1].strip() if "\n" in raw else "Research this topic"
        return SupervisorDecision(action="RESEARCH", task=task)
    if first_line.startswith("EXECUTE"):
        task = raw.strip().split("\n", 1)[1].strip() if "\n" in raw else "Run the calculation"
        return SupervisorDecision(action="EXECUTE", task=task)
    if first_line.startswith("ANALYSE") or first_line.startswith("ANALYZE"):
        task = raw.strip().split("\n", 1)[1].strip() if "\n" in raw else "Analyse the available information"
        return SupervisorDecision(action="ANALYSE", task=task)
    if first_line.startswith("ANSWER"):
        task = raw.strip().split("\n", 1)[1].strip() if "\n" in raw else raw.strip()
        return SupervisorDecision(action="ANSWER", task=task)

    # Layer 2: keyword matching for unstructured responses (case-insensitive)
    if "research" in raw_lower or "search" in raw_lower:
        return SupervisorDecision(action="RESEARCH", task=raw.strip())
    if "execut" in raw_lower or "calculat" in raw_lower or "comput" in raw_lower:
        return SupervisorDecision(action="EXECUTE", task=raw.strip())
    if "analys" in raw_lower or "reason" in raw_lower or "evaluat" in raw_lower:
        return SupervisorDecision(action="ANALYSE", task=raw.strip())

    # Layer 3: anything else → treat as final answer
    return SupervisorDecision(action="ANSWER", task=raw.strip())


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def _supervisor_decide(
    state: MainState, runtime: Runtime[Context]
) -> Dict:
    """Analyse the initial task and pick the first specialist (or finish directly)."""
    _trace.info("📍 [3/5] Supervisor._decide — 分析任务，选择第一个专家")

    model = load_chat_model(resolve_model(runtime))

    decision = await _get_decision(model, [
        SystemMessage(content=SUPERVISOR_DECIDE_PROMPT.format(query=state.user_query)),
        HumanMessage(content="Analyse this task and decide the first step."),
    ])

    # Map structured action to internal routing keys
    action_map = {
        "RESEARCH": "researcher",
        "EXECUTE": "executor",
        "ANALYSE": "analyst",
    }

    if decision.action == "ANSWER":
        _trace.info("   → 监督者决策: FINISH (直接回答)")
        return {
            "messages": [AIMessage(content=decision.task)],
            "supervisor_next_specialist": "FINISH",
            "supervisor_iteration": 0,
        }

    specialist = action_map[decision.action]
    _trace.info("   → 监督者决策: %s", specialist)

    return {
        "messages": [AIMessage(content=f"[Supervisor → {specialist}] {decision.task}")],
        "supervisor_next_specialist": specialist,
        "supervisor_iteration": 0,
    }


# ---------------------------------------------------------------------------
# Shared specialist runner — the three specialists differ only in their
# prompt template and whether they use tools.  This helper eliminates the
# duplication that previously existed across _supervisor_researcher,
# _supervisor_analyst, and _supervisor_executor.
# ---------------------------------------------------------------------------

_SPECIALIST_LABELS: dict[str, str] = {
    "researcher": "执行搜索/信息收集",
    "analyst":    "执行分析/推理",
    "executor":   "执行计算/代码",
}


async def _run_specialist(
    state: MainState,
    runtime: Runtime[Context],
    *,
    name: str,
    prompt_template: str,
    use_tools: bool = False,
) -> Dict:
    """Run a specialist and return its messages.

    Parameters
    ----------
    name :
        Specialist name (``"researcher"`` / ``"analyst"`` / ``"executor"``).
        Used for trace logging and the injected ``HumanMessage`` label.
    prompt_template :
        System prompt with ``{task}``, ``{subtask}``, and optionally
        ``{context}`` placeholders.  Extra kwargs are safely ignored by
        :meth:`str.format`.
    use_tools :
        When ``True``, bind tools and run a mini ReAct loop (researcher &
        executor).  When ``False``, a single ``model.ainvoke()`` call (analyst).
    """
    _trace.info("   → Supervisor._%s — %s", name, _SPECIALIST_LABELS.get(name, ""))

    subtask = _extract_subtask(state)
    context = _gather_context(state)

    prompt_kwargs = {
        "task": state.user_query,
        "subtask": subtask,
        "context": context,
    }
    task_prompt = prompt_template.format(**prompt_kwargs)

    label = f"{name.capitalize()} task:"

    if use_tools:
        await _ensure_mcp_loaded()
        tools = get_all_tools()
        model = load_chat_model(resolve_model(runtime)).bind_tools(tools)

        msgs = [
            SystemMessage(content=task_prompt),
            HumanMessage(content=f"{label} {subtask}"),
        ]
        msgs = await run_mini_react_loop(model, tools, msgs, max_rounds=3)
        result_msgs = [m for m in msgs[1:] if not (
            isinstance(m, HumanMessage) and str(m.content).startswith(label)
        )]
        return {"messages": result_msgs}
    else:
        model = load_chat_model(resolve_model(runtime))
        response = await model.ainvoke([
            SystemMessage(content=task_prompt),
            HumanMessage(content=f"{label} {subtask}"),
        ])
        return {"messages": [response]}


async def _supervisor_researcher(
    state: MainState, runtime: Runtime[Context]
) -> Dict:
    """Research specialist — mini ReAct loop biased toward search."""
    return await _run_specialist(
        state, runtime,
        name="researcher",
        prompt_template=RESEARCHER_PROMPT,
        use_tools=True,
    )


async def _supervisor_analyst(
    state: MainState, runtime: Runtime[Context]
) -> Dict:
    """Analyst specialist — pure reasoning, no tools."""
    return await _run_specialist(
        state, runtime,
        name="analyst",
        prompt_template=ANALYST_PROMPT,
        use_tools=False,
    )


async def _supervisor_executor(
    state: MainState, runtime: Runtime[Context]
) -> Dict:
    """Executor specialist — mini ReAct loop biased toward python_repl."""
    return await _run_specialist(
        state, runtime,
        name="executor",
        prompt_template=EXECUTOR_PROMPT,
        use_tools=True,
    )


async def _supervisor_review(
    state: MainState, runtime: Runtime[Context]
) -> Dict:
    """Review progress and decide: call another specialist or produce final answer."""
    _trace.info("   → Supervisor._review — 评估进度 (迭代 %d)", state.supervisor_iteration + 1)

    new_iteration = state.supervisor_iteration + 1

    # Rate-limit protection
    if state.supervisor_iteration > 0:
        await asyncio.sleep(3)

    # Force finish if at cap
    if new_iteration >= MAX_SUPERVISOR_ITERATIONS:
        _trace.info("   → 达到最大迭代数，强制结束")
        return {
            "supervisor_next_specialist": "FINISH",
            "supervisor_iteration": new_iteration,
            "messages": [AIMessage(
                content="[Supervisor] Maximum iterations reached — producing final answer."
            )],
        }

    model = load_chat_model(resolve_model(runtime))
    history = _gather_context(state)

    decision = await _get_decision(model, [
        SystemMessage(content=SUPERVISOR_REVIEW_PROMPT.format(
            query=state.user_query, history=history
        )),
        HumanMessage(content="Review the progress and decide the next step."),
    ])

    _trace.info("   → 监督者决策: %s (迭代 %d/%d)", decision.action, new_iteration, MAX_SUPERVISOR_ITERATIONS)

    action_map = {
        "RESEARCH": "researcher",
        "EXECUTE": "executor",
        "ANALYSE": "analyst",
    }

    result: Dict = {
        "supervisor_iteration": new_iteration,
    }

    if decision.action == "ANSWER":
        result["supervisor_next_specialist"] = "FINISH"
        result["messages"] = [
            AIMessage(content=f"[Supervisor] {decision.task}"),
            await _synthesise_final_answer(state, runtime, model),
        ]
    else:
        specialist = action_map[decision.action]
        result["supervisor_next_specialist"] = specialist
        result["messages"] = [
            AIMessage(content=f"[Supervisor → {specialist}] {decision.task}")
        ]

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_subtask(state: MainState) -> str:
    """Extract the latest subtask from the supervisor's last decision message.

    With structured output, messages follow the format:
    ``[Supervisor → researcher] Search for Japan's 2024 GDP``
    """
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            content = str(msg.content)
            if "[Supervisor →" in content:
                # Extract everything after "] "
                if "] " in content:
                    return content.split("] ", 1)[1].strip()
                return content
    return state.user_query


def _gather_context(state: MainState) -> str:
    """Summarise the conversation so far for the supervisor to review.

    All messages are kept in full — the structural limits (max 5 iterations,
    max 3 mini ReAct rounds per specialist) naturally bound the total context
    size.  Truncating any message type risks dropping the specific data point
    that the supervisor or final-synthesis step needs.
    """
    parts: list[str] = []
    for i, msg in enumerate(state.messages):
        role = type(msg).__name__.replace("Message", "")
        parts.append(f"[{i}] {role}: {str(msg.content)}")
    return "\n\n".join(parts) if parts else "(no work completed yet)"


async def _synthesise_final_answer(
    state: MainState,
    runtime: Runtime[Context],
    model,
) -> AIMessage:
    """Generate a polished final answer from all work completed by specialists."""
    context = _gather_context(state)

    prompt = f"""You are a **task supervisor** writing the final answer to the user.

Original query:
{state.user_query}

All work completed by your specialist team:
{context}

Write a polished, well-structured final answer that directly addresses the user's query.
Use the data and insights gathered by the specialists.  Include specific numbers and citations.
Do NOT mention the supervisor, specialists, or the process — just deliver the answer."""

    response = await model.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Write the final answer."),
    ])

    return response


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_supervisor(state: MainState) -> Literal[
    "researcher", "analyst", "executor", "__end__"
]:
    specialist = state.supervisor_next_specialist
    if specialist == "FINISH":
        _trace.info("📍 [4/5] Supervisor 完成 → __end__")
        return "__end__"
    _trace.info("   → 路由到专家: %s", specialist)
    return specialist  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Subgraph factory
# ---------------------------------------------------------------------------


def build_supervisor_subgraph() -> StateGraph:
    """Create a compiled Supervisor subgraph.

    Returns a ``CompiledStateGraph`` that can be added as a node in the parent
    graph alongside the existing ReAct / Reflection / Plan-Solve subgraphs.
    """
    builder = StateGraph(MainState)

    builder.add_node("supervisor_decide", _supervisor_decide)
    builder.add_node("supervisor_researcher", _supervisor_researcher)
    builder.add_node("supervisor_analyst", _supervisor_analyst)
    builder.add_node("supervisor_executor", _supervisor_executor)
    builder.add_node("supervisor_review", _supervisor_review)

    builder.add_edge("__start__", "supervisor_decide")

    builder.add_conditional_edges(
        "supervisor_decide", _route_supervisor, {
            "researcher": "supervisor_researcher",
            "analyst": "supervisor_analyst",
            "executor": "supervisor_executor",
            "__end__": "__end__",
        },
    )

    # All specialists return to the supervisor for review
    builder.add_edge("supervisor_researcher", "supervisor_review")
    builder.add_edge("supervisor_analyst", "supervisor_review")
    builder.add_edge("supervisor_executor", "supervisor_review")

    # Review can delegate again or finish
    builder.add_conditional_edges(
        "supervisor_review", _route_supervisor, {
            "researcher": "supervisor_researcher",
            "analyst": "supervisor_analyst",
            "executor": "supervisor_executor",
            "__end__": "__end__",
        },
    )

    return builder.compile(name="Supervisor")
