"""Build the executable evidence-bounded research query graph."""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from paperops.clients.protocols import RetrievalBackend
from paperops.research.models import ResearchStatus
from paperops.research.nodes import ResearchNodes
from paperops.research.protocols import ResearchModel
from paperops.research.state import ResearchQueryState
from paperops.settings import Settings


def build_research_graph(
    *,
    retrieval: RetrievalBackend,
    model: ResearchModel,
    settings: Settings,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> CompiledStateGraph[
    ResearchQueryState,
    None,
    ResearchQueryState,
    ResearchQueryState,
]:
    """Compile the query graph with explicit retrieval and model boundaries."""
    nodes = ResearchNodes(retrieval=retrieval, model=model, settings=settings)

    def route_after_initialize(
        state: ResearchQueryState,
    ) -> Literal["retrieve", "end"]:
        return "end" if state.get("status") == ResearchStatus.FAILED else "retrieve"

    def route_after_retrieve(
        state: ResearchQueryState,
    ) -> Literal["assess", "end"]:
        return "end" if state.get("status") == ResearchStatus.FAILED else "assess"

    def route_after_assess(
        state: ResearchQueryState,
    ) -> Literal["answer", "rewrite", "refuse", "end"]:
        if state.get("status") == ResearchStatus.FAILED:
            return "end"
        assessment = state.get("assessment")
        if assessment is not None and assessment.sufficient:
            return "answer"
        if state.get("rewrite_count", 0) < settings.research_max_rewrites:
            return "rewrite"
        return "refuse"

    def route_after_rewrite(
        state: ResearchQueryState,
    ) -> Literal["retrieve", "end"]:
        return "retrieve" if state.get("status") == ResearchStatus.RETRIEVING else "end"

    builder = StateGraph(
        ResearchQueryState,
        input_schema=ResearchQueryState,
        output_schema=ResearchQueryState,
    )
    builder.add_node("initialize_query", nodes.initialize)
    builder.add_node("retrieve_evidence", nodes.retrieve)
    builder.add_node("assess_evidence", nodes.assess)
    builder.add_node("rewrite_query", nodes.rewrite)
    builder.add_node("synthesize_answer", nodes.synthesize)
    builder.add_node("refuse_answer", nodes.refuse)

    builder.add_edge(START, "initialize_query")
    builder.add_conditional_edges(
        "initialize_query",
        route_after_initialize,
        {"retrieve": "retrieve_evidence", "end": END},
    )
    builder.add_conditional_edges(
        "retrieve_evidence",
        route_after_retrieve,
        {"assess": "assess_evidence", "end": END},
    )
    builder.add_conditional_edges(
        "assess_evidence",
        route_after_assess,
        {
            "answer": "synthesize_answer",
            "rewrite": "rewrite_query",
            "refuse": "refuse_answer",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {"retrieve": "retrieve_evidence", "end": END},
    )
    builder.add_edge("synthesize_answer", END)
    builder.add_edge("refuse_answer", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
        name="paperops-research",
    )
