"""State structures for the multi-mode agent framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Input state — the external interface for incoming messages."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates:
    1. HumanMessage — user input
    2. AIMessage with .tool_calls — agent picks tool(s)
    3. ToolMessage(s) — tool responses
    4. AIMessage without .tool_calls — final answer

    The ``add_messages`` reducer ensures append-only semantics with
    ID-based deduplication.
    """


@dataclass
class State(InputState):
    """Internal agent state (managed by LangGraph)."""

    is_last_step: IsLastStep = field(default=False)
    """
    Managed variable set to True when step count reaches recursion_limit - 1.
    Prevents infinite loops.
    """


# ---------------------------------------------------------------------------
# Multi-mode extensions
# ---------------------------------------------------------------------------


@dataclass
class MainState(State):
    """Extended state shared across the Mode Router and all three agent subgraphs.

    Each subgraph reads/writes ``messages`` (inherited from State).  Mode-specific
    fields are used by only one subgraph and ignored by the others.
    """

    # -- routing --
    mode: str = ""
    """Which mode the router selected: ``react`` | ``reflection`` | ``plan_solve``."""

    route_reason: str = ""
    """Raw router output for debugging / traceability."""

    mode_router_model: str = ""
    """Optional override for the router LLM (falls back to Context.model)."""

    user_query: str = ""
    """Original user query, preserved so subgraphs can access it even after
    clearing or modifying ``messages``."""

    # -- reflection mode --
    reflection_iteration: int = 0
    """Current iteration count inside the Reflection subgraph."""

    # -- plan-solve mode --
    plan_steps: list[str] = field(default_factory=list)
    """Decomposed plan steps from the planner."""

    current_step: int = 0
    """Index of the step currently being executed."""

    step_results: list[str] = field(default_factory=list)
    """Results collected from each executed step."""

    # -- supervisor mode --
    supervisor_next_specialist: str = ""
    """Which specialist the supervisor wants to invoke next.
    Values: ``""`` (not started), ``"researcher"``, ``"analyst"``,
    ``"executor"``, ``"FINISH"``."""

    supervisor_iteration: int = 0
    """Number of specialist delegations so far (capped)."""

    # -- memory --
    recalled_facts: str = ""
    """Facts retrieved from long-term memory relevant to the current query.
    Injected as context before the router runs so every mode benefits."""

    memory_status: str = ""
    """Human-readable summary of the last memory operation (inject or extract).
    Always updated so LangSmith traces show memory-node activity even when
    no facts were found or stored.  Not used by any business logic."""

    conversation_summary: str = ""
    """Compressed summary of older messages when the conversation exceeds the
    context window.  Populated by the context-compression path in memory.py."""

    # -- benchmark / eval isolation --
    benchmark_mode: bool = False
    """When True, the framework skips memory extraction and the remember tool
    rejects storage to prevent synthetic eval queries from contaminating
    long-term memory.  Set by eval runners (``run_evals.py``) and never by
    user-facing invocations."""
