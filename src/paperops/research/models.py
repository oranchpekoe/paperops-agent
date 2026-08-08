"""Validated decisions and evidence for research-query execution."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class ResearchStatus(StrEnum):
    """Lifecycle states for one evidence-bounded research query."""

    PENDING = "pending"
    RETRIEVING = "retrieving"
    ASSESSING = "assessing"
    REWRITING = "rewriting"
    ANSWERING = "answering"
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class ResearchFailureCode(StrEnum):
    """Machine-readable failures emitted by the research graph."""

    INVALID_QUERY = "invalid_query"
    RETRIEVAL_ERROR = "retrieval_error"
    MODEL_ERROR = "model_error"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    CITATION_VALIDATION_ERROR = "citation_validation_error"


class EvidenceCitation(BaseModel):
    """One checkpointed retrieval chunk with a stable local citation id."""

    citation_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    heading_path: list[str] = Field(default_factory=list)
    retrieval_query: str = Field(min_length=1)
    retrieval_round: int = Field(ge=1)


class EvidenceAssessmentRequest(BaseModel):
    """Input supplied to the semantic evidence judge."""

    question: str = Field(min_length=1)
    attempted_queries: list[str] = Field(min_length=1)
    evidence: list[EvidenceCitation] = Field(min_length=1)


class EvidenceAssessment(BaseModel):
    """Structured judgment used only for graph routing."""

    sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    missing_aspects: list[str] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def require_missing_aspect_for_insufficient_evidence(self) -> EvidenceAssessment:
        """Require an actionable gap before spending another retrieval round."""
        if not self.sufficient and not self.missing_aspects:
            raise ValueError(
                "insufficient evidence requires at least one missing aspect"
            )
        return self


class QueryRewriteRequest(BaseModel):
    """Input supplied when the graph has retrieval budget remaining."""

    question: str = Field(min_length=1)
    attempted_queries: list[str] = Field(min_length=1)
    missing_aspects: list[str] = Field(min_length=1)


class QueryRewrite(BaseModel):
    """One auditable replacement query for a bounded retrieval retry."""

    query: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1)


class AnswerSynthesisRequest(BaseModel):
    """Evidence-only input supplied to answer generation."""

    question: str = Field(min_length=1)
    evidence: list[EvidenceCitation] = Field(min_length=1)


class ResearchAnswer(BaseModel):
    """Answer whose citations must resolve to evidence in graph state."""

    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class ResearchFailure(BaseModel):
    """Structured failure safe to persist in a checkpoint."""

    stage: ResearchStatus
    code: ResearchFailureCode
    message: str = Field(min_length=1)
    retryable: bool = False


class ResearchEvent(BaseModel):
    """Compact audit event for one research-query transition."""

    status: ResearchStatus
    message: str = Field(min_length=1)
    retrieval_round: int | None = Field(default=None, ge=1)


class ModelCallUsage(BaseModel):
    """Provider telemetry for one attempted semantic-model call."""

    operation: str = Field(min_length=1)
    success: bool
    latency_ms: float = Field(ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
