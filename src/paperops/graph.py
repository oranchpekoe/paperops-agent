"""Build the executable PaperOps single-document state machine."""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from paperops.clients.fakes import FakeKnowledgeBaseClient, FakeParserClient
from paperops.clients.protocols import ParserClient, RetrievalBackend
from paperops.models import JobStatus, QualityVerdict
from paperops.nodes.workflow import WorkflowNodes
from paperops.quality.rules import QualityPolicy
from paperops.settings import Settings
from paperops.state import DocumentJobState


def _route_after_initialize(
    state: DocumentJobState,
) -> Literal["parse", "end"]:
    """Stop invalid jobs before any parser side effect."""
    return "end" if state.get("status") == JobStatus.FAILED else "parse"


def _route_after_approval(
    state: DocumentJobState,
) -> Literal["ingest", "end"]:
    """Continue only after a validated approval."""
    return "end" if state.get("status") == JobStatus.FAILED else "ingest"


def _route_after_ingest(
    state: DocumentJobState,
) -> Literal["evaluate", "end"]:
    """Run retrieval verification only for a successful ingestion."""
    return "end" if state.get("status") == JobStatus.FAILED else "evaluate"


def build_graph(
    *,
    parser: ParserClient,
    knowledge_base: RetrievalBackend,
    settings: Settings,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> CompiledStateGraph[
    DocumentJobState,
    None,
    DocumentJobState,
    DocumentJobState,
]:
    """Compile PaperOps with injected clients and optional test breakpoints."""
    nodes = WorkflowNodes(
        parser=parser,
        knowledge_base=knowledge_base,
        settings=settings,
        quality_policy=QualityPolicy.from_settings(settings),
    )

    def route_after_quality(
        state: DocumentJobState,
    ) -> Literal["ingest", "retry", "review", "fail"]:
        decision = state.get("quality_decision")
        if decision is None:
            return "fail"
        if decision.verdict == QualityVerdict.PASS:
            return "ingest"
        if decision.verdict == QualityVerdict.REVIEW:
            return "review"
        if state.get("parse_attempts", 0) < settings.max_parse_attempts:
            return "retry"
        return "fail"

    builder = StateGraph(
        DocumentJobState,
        input_schema=DocumentJobState,
        output_schema=DocumentJobState,
    )
    builder.add_node("initialize", nodes.initialize)
    builder.add_node("parse_document", nodes.parse_document)
    builder.add_node("quality_check", nodes.quality_check)
    builder.add_node("mark_waiting_approval", nodes.mark_waiting_approval)
    builder.add_node("request_approval", nodes.request_approval)
    builder.add_node("fail_quality", nodes.fail_quality)
    builder.add_node("ingest_document", nodes.ingest_document)
    builder.add_node("evaluate_retrieval", nodes.evaluate_retrieval)

    builder.add_edge(START, "initialize")
    builder.add_conditional_edges(
        "initialize",
        _route_after_initialize,
        {"parse": "parse_document", "end": END},
    )
    builder.add_edge("parse_document", "quality_check")
    builder.add_conditional_edges(
        "quality_check",
        route_after_quality,
        {
            "ingest": "ingest_document",
            "retry": "parse_document",
            "review": "mark_waiting_approval",
            "fail": "fail_quality",
        },
    )
    builder.add_edge("mark_waiting_approval", "request_approval")
    builder.add_conditional_edges(
        "request_approval",
        _route_after_approval,
        {"ingest": "ingest_document", "end": END},
    )
    builder.add_edge("fail_quality", END)
    builder.add_conditional_edges(
        "ingest_document",
        _route_after_ingest,
        {"evaluate": "evaluate_retrieval", "end": END},
    )
    builder.add_edge("evaluate_retrieval", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
        name="paperops",
    )


_default_settings = Settings()
graph = build_graph(
    parser=FakeParserClient(_default_settings.artifacts_dir),
    knowledge_base=FakeKnowledgeBaseClient(),
    settings=_default_settings,
)
