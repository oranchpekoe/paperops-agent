"""Agent mode subgraphs.

Each module exports a ``build_*_subgraph()`` factory that returns a compiled
LangGraph subgraph.  The main orchestrator (``graph.py``) wires them together
behind a Mode Router.
"""

from react_agent.modes.plan_solve import build_plan_solve_subgraph
from react_agent.modes.react import build_react_subgraph
from react_agent.modes.reflection import build_reflection_subgraph
from react_agent.modes.supervisor import build_supervisor_subgraph

__all__ = [
    "build_plan_solve_subgraph",
    "build_react_subgraph",
    "build_reflection_subgraph",
    "build_supervisor_subgraph",
]
