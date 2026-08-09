"""Build the executable multi-paper comparison graph."""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from paperops.clients.protocols import RetrievalBackend
from paperops.comparison.models import ComparisonStatus
from paperops.comparison.nodes import ComparisonNodes
from paperops.comparison.protocols import ComparisonModel
from paperops.comparison.state import ComparisonState
from paperops.settings import Settings


def build_comparison_graph(
    *,
    retrieval: RetrievalBackend,
    model: ComparisonModel,
    settings: Settings,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> CompiledStateGraph[
    ComparisonState,
    None,
    ComparisonState,
    ComparisonState,
]:
    """Compile a bounded matrix extraction and gap-retrieval workflow."""
    nodes = ComparisonNodes(retrieval=retrieval, model=model, settings=settings)

    def after_initialize(
        state: ComparisonState,
    ) -> Literal["retrieve", "end"]:
        return (
            "retrieve"
            if state.get("status") is ComparisonStatus.RETRIEVING_INITIAL
            else "end"
        )

    def after_retrieve(state: ComparisonState) -> Literal["extract", "end"]:
        return (
            "extract" if state.get("status") is ComparisonStatus.EXTRACTING else "end"
        )

    def after_extract(state: ComparisonState) -> Literal["gaps", "end"]:
        return (
            "gaps" if state.get("status") is ComparisonStatus.RETRIEVING_GAPS else "end"
        )

    builder = StateGraph(
        ComparisonState,
        input_schema=ComparisonState,
        output_schema=ComparisonState,
    )
    builder.add_node("initialize_comparison", nodes.initialize)
    builder.add_node("retrieve_initial_matrix", nodes.retrieve_initial)
    builder.add_node("extract_matrix", nodes.extract)
    builder.add_node("retrieve_missing_cells", nodes.retrieve_gaps)
    builder.add_edge(START, "initialize_comparison")
    builder.add_conditional_edges(
        "initialize_comparison",
        after_initialize,
        {"retrieve": "retrieve_initial_matrix", "end": END},
    )
    builder.add_conditional_edges(
        "retrieve_initial_matrix",
        after_retrieve,
        {"extract": "extract_matrix", "end": END},
    )
    builder.add_conditional_edges(
        "extract_matrix",
        after_extract,
        {"gaps": "retrieve_missing_cells", "end": END},
    )
    builder.add_conditional_edges(
        "retrieve_missing_cells",
        after_retrieve,
        {"extract": "extract_matrix", "end": END},
    )
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
        name="paperops-comparison",
    )
