"""Validated HTTP request and response models for PaperOps jobs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from paperops.comparison.models import (
    ComparisonCell,
    ComparisonDimension,
    ComparisonDocument,
    ComparisonEvent,
    ComparisonFailure,
    ComparisonSearchAttempt,
    ComparisonStatus,
    ComparisonStopReason,
)
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
    ResearchStopReason,
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
    new_evidence_count: int = 0
    model_calls: int = 0
    attempted_queries: list[str] = Field(default_factory=list)
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    assessment: EvidenceAssessment | None = None
    last_rewrite: QueryRewrite | None = None
    answer: ResearchAnswer | None = None
    failure: ResearchFailure | None = None
    stop_reason: ResearchStopReason | None = None
    events: list[ResearchEvent] = Field(default_factory=list)
    runtime_error: str | None = None


class ComparisonRequest(BaseModel):
    """Compare explicit dimensions across already-indexed papers."""

    knowledge_base: str = Field(min_length=1, max_length=128)
    documents: list[ComparisonDocument] = Field(min_length=2, max_length=8)
    dimensions: list[ComparisonDimension] = Field(min_length=1, max_length=6)


class ComparisonAccepted(BaseModel):
    """Acknowledge asynchronous construction of an evidence matrix."""

    thread_id: str
    status: ComparisonStatus
    status_url: str


class ComparisonView(BaseModel):
    """Expose the initial baseline matrix and gap-retrieval result."""

    thread_id: str
    comparison_id: str | None = None
    status: ComparisonStatus
    running: bool
    next_nodes: list[str] = Field(default_factory=list)
    knowledge_base: str
    documents: list[ComparisonDocument] = Field(default_factory=list)
    dimensions: list[ComparisonDimension] = Field(default_factory=list)
    retrieval_round: int = 0
    gap_round: int = 0
    retrieval_calls: int = 0
    model_calls: int = 0
    new_evidence_count: int = 0
    attempted_searches: list[ComparisonSearchAttempt] = Field(default_factory=list)
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    initial_cells: list[ComparisonCell] = Field(default_factory=list)
    cells: list[ComparisonCell] = Field(default_factory=list)
    total_cells: int = 0
    initial_supported_cells: int = 0
    supported_cells: int = 0
    missing_cells: int = 0
    recovered_cell_count: int = 0
    stop_reason: ComparisonStopReason | None = None
    failure: ComparisonFailure | None = None
    events: list[ComparisonEvent] = Field(default_factory=list)
    runtime_error: str | None = None
