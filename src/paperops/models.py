"""Validated boundary models for the PaperOps workflow."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    """Lifecycle states for a single-document indexing job."""

    PENDING = "pending"
    PARSING = "parsing"
    QUALITY_CHECK = "quality_check"
    WAITING_APPROVAL = "waiting_approval"
    INDEXING = "indexing"
    RETRIEVAL_EVAL = "retrieval_eval"
    COMPLETED = "completed"
    FAILED = "failed"


class QualityVerdict(StrEnum):
    """Permitted outcomes of a document quality decision."""

    PASS = "pass"
    RETRY = "retry"
    REVIEW = "review"


class ApprovalAction(StrEnum):
    """Actions a reviewer can take for an uncertain parse."""

    APPROVE = "approve"
    REJECT = "reject"


class FailureCode(StrEnum):
    """Machine-readable failure categories emitted by the workflow."""

    INVALID_SOURCE = "invalid_source"
    PARSER_ERROR = "parser_error"
    QUALITY_CHECK_ERROR = "quality_check_error"
    QUALITY_RETRIES_EXHAUSTED = "quality_retries_exhausted"
    INVALID_APPROVAL = "invalid_approval"
    APPROVAL_REJECTED = "approval_rejected"
    INDEX_ERROR = "index_error"
    RETRIEVAL_ERROR = "retrieval_error"
    RETRIEVAL_FAILED = "retrieval_failed"


class QualityMetrics(BaseModel):
    """Small, deterministic measurements derived from a Markdown artifact."""

    character_count: int = Field(ge=0)
    heading_count: int = Field(ge=0)
    section_count: int = Field(ge=0)
    broken_image_references: int = Field(ge=0)
    replacement_character_ratio: float = Field(ge=0.0, le=1.0)


class QualityDecision(BaseModel):
    """Structured result produced by rule-based or semantic validation."""

    verdict: QualityVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    retry_reason: str | None = None
    metrics: QualityMetrics | None = None


class ParseRequest(BaseModel):
    """Idempotent request sent to a document parser."""

    job_id: str = Field(min_length=1)
    source_pdf: str = Field(min_length=1)
    file_hash: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1)


class ParseResult(BaseModel):
    """Reference to a parser artifact stored outside graph state."""

    markdown_path: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    created: bool


class IngestRequest(BaseModel):
    """Idempotent request sent to a knowledge-base service."""

    job_id: str = Field(min_length=1)
    knowledge_base: str = Field(min_length=1)
    file_hash: str = Field(min_length=1)
    markdown_path: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)


class IngestResult(BaseModel):
    """Indexed document identity returned by a retrieval backend."""

    document_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    created: bool
    chunk_count: int = Field(default=0, ge=0)


class SearchRequest(BaseModel):
    """Retrieval request used for search or document-scoped verification."""

    knowledge_base: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_document_id: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)


class SearchHit(BaseModel):
    """Evidence chunk returned by a retrieval backend."""

    document_id: str = Field(min_length=1)
    chunk_id: str | None = None
    content: str
    score: float = Field(ge=0.0, le=1.0)
    heading_path: list[str] = Field(default_factory=list)


class ApprovalDecision(BaseModel):
    """Validated response supplied when the graph resumes from an interrupt."""

    action: ApprovalAction
    note: str = ""


class RetrievalReport(BaseModel):
    """Structured result of a post-indexing retrieval probe."""

    passed: bool
    query: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    hit_count: int = Field(ge=0)
    backend: str = "unknown"
    strategy: str = "index_probe"
    evidence: list[str] = Field(default_factory=list)


class WorkflowFailure(BaseModel):
    """Structured error recorded without serialising an exception object."""

    stage: JobStatus
    code: FailureCode
    message: str = Field(min_length=1)
    retryable: bool = False


class WorkflowEvent(BaseModel):
    """Compact audit event persisted in a job checkpoint."""

    status: JobStatus
    message: str = Field(min_length=1)
    attempt: int | None = Field(default=None, ge=1)
