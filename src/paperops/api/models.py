"""Validated HTTP request and response models for PaperOps jobs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from paperops.models import (
    ApprovalDecision,
    JobStatus,
    QualityDecision,
    RetrievalReport,
    WorkflowEvent,
    WorkflowFailure,
)


class JobAccepted(BaseModel):
    """Acknowledge that a job was scheduled for asynchronous execution."""

    thread_id: str
    status: JobStatus
    status_url: str


class JobView(BaseModel):
    """Expose the compact, checkpointed state of one workflow thread."""

    thread_id: str
    job_id: str | None = None
    status: JobStatus
    running: bool
    approval_required: bool
    next_nodes: list[str] = Field(default_factory=list)
    parse_attempts: int = 0
    parsed_markdown_path: str | None = None
    quality_decision: QualityDecision | None = None
    indexed_document_id: str | None = None
    indexed_chunk_count: int = 0
    retrieval_report: RetrievalReport | None = None
    failure: WorkflowFailure | None = None
    events: list[WorkflowEvent] = Field(default_factory=list)
    runtime_error: str | None = None


class ApprovalAccepted(BaseModel):
    """Acknowledge a validated approval or rejection command."""

    thread_id: str
    decision: ApprovalDecision
    status_url: str


class ResumeAccepted(BaseModel):
    """Acknowledge explicit recovery of an unfinished checkpoint."""

    thread_id: str
    status_url: str


class HealthView(BaseModel):
    """Return local service liveness without invoking external dependencies."""

    status: str = "ok"
    client_mode: str
    retrieval_backend: str
    active_jobs: int
