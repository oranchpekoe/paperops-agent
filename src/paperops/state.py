"""LangGraph state schema for one research-paper ingestion job."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from paperops.models import (
    ApprovalDecision,
    JobStatus,
    QualityDecision,
    RetrievalReport,
    WorkflowEvent,
    WorkflowFailure,
)


class DocumentJobState(TypedDict, total=False):
    """References and decisions persisted while a document moves through the graph.

    Parsed document bodies stay in the artifact directory instead of graph state.
    This keeps checkpoints small and avoids repeatedly serialising large Markdown
    payloads.
    """

    job_id: str
    source_pdf: str
    target_knowledge_base: str
    file_hash: str
    status: JobStatus
    parse_attempts: int
    parsed_markdown_path: str
    quality_decision: QualityDecision
    approval_decision: ApprovalDecision
    indexed_document_id: str
    indexed_chunk_count: int
    evaluation_report_path: str
    retrieval_report: RetrievalReport
    failure: WorkflowFailure
    errors: Annotated[list[WorkflowFailure], operator.add]
    events: Annotated[list[WorkflowEvent], operator.add]
