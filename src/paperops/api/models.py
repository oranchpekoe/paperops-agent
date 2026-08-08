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
from paperops.research.models import (
    EvidenceAssessment,
    EvidenceCitation,
    QueryRewrite,
    ResearchAnswer,
    ResearchEvent,
    ResearchFailure,
    ResearchStatus,
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
    research_model: str
    active_jobs: int


class ResearchQueryRequest(BaseModel):
    """Submit one question against an existing indexed collection."""

    knowledge_base: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)


class ResearchQueryAccepted(BaseModel):
    """Acknowledge asynchronous execution of a research query."""

    thread_id: str
    status: ResearchStatus
    status_url: str


class ResearchQueryView(BaseModel):
    """Expose evidence, decisions, budgets, and the validated final answer."""

    thread_id: str
    query_id: str | None = None
    status: ResearchStatus
    running: bool
    next_nodes: list[str] = Field(default_factory=list)
    knowledge_base: str
    question: str
    current_query: str
    retrieval_round: int = 0
    rewrite_count: int = 0
    retrieval_calls: int = 0
    model_calls: int = 0
    attempted_queries: list[str] = Field(default_factory=list)
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    assessment: EvidenceAssessment | None = None
    last_rewrite: QueryRewrite | None = None
    answer: ResearchAnswer | None = None
    failure: ResearchFailure | None = None
    events: list[ResearchEvent] = Field(default_factory=list)
    runtime_error: str | None = None
