"""Shared domain models for the PaperOps workflow."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    """Lifecycle states for a single-document ingestion job."""

    PENDING = "pending"
    PARSING = "parsing"
    QUALITY_CHECK = "quality_check"
    WAITING_APPROVAL = "waiting_approval"
    UPLOADING = "uploading"
    RETRIEVAL_EVAL = "retrieval_eval"
    COMPLETED = "completed"
    FAILED = "failed"


class QualityVerdict(StrEnum):
    """Permitted outcomes of a document quality decision."""

    PASS = "pass"
    RETRY = "retry"
    REVIEW = "review"


class QualityDecision(BaseModel):
    """Structured result produced by rule-based or semantic validation."""

    verdict: QualityVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    retry_reason: str | None = None
