"""LangGraph state schema for one research-paper ingestion job."""

from __future__ import annotations

from typing import TypedDict

from paperops.models import JobStatus, QualityDecision


class DocumentJobState(TypedDict, total=False):
    """References and decisions persisted while a document moves through the graph.

    Parsed document bodies stay in the artifact directory instead of graph state.
    This keeps checkpoints small and avoids repeatedly serialising large Markdown
    payloads.
    """

    job_id: str
    source_pdf: str
    file_hash: str
    status: JobStatus
    parse_attempts: int
    parsed_markdown_path: str
    rule_quality_score: float
    quality_decision: QualityDecision
    approval_required: bool
    ragflow_document_id: str
    evaluation_report_path: str
    errors: list[str]
