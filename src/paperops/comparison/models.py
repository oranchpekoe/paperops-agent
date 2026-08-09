"""Typed contracts for multi-paper structured comparison."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from paperops.research.models import CitationId, EvidenceCitation


class ComparisonStatus(StrEnum):
    """Lifecycle states for one comparison matrix."""

    PENDING = "pending"
    RETRIEVING_INITIAL = "retrieving_initial"
    EXTRACTING = "extracting"
    RETRIEVING_GAPS = "retrieving_gaps"
    COMPLETED = "completed"
    FAILED = "failed"


class ComparisonCellStatus(StrEnum):
    """Whether one requested document-dimension cell is grounded."""

    SUPPORTED = "supported"
    MISSING = "missing"


class ComparisonFailureCode(StrEnum):
    """Machine-readable failures emitted by the comparison graph."""

    INVALID_REQUEST = "invalid_request"
    RETRIEVAL_ERROR = "retrieval_error"
    MODEL_ERROR = "model_error"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    CITATION_VALIDATION_ERROR = "citation_validation_error"


class ComparisonStopReason(StrEnum):
    """Explain why the matrix stopped gathering evidence."""

    ALL_CELLS_SUPPORTED = "all_cells_supported"
    STAGNANT_RETRIEVAL = "stagnant_retrieval"
    GAP_BUDGET_EXHAUSTED = "gap_budget_exhausted"


class ComparisonDocument(BaseModel):
    """One already-indexed paper included in a comparison."""

    document_id: str = Field(min_length=1, max_length=256)
    label: str = Field(min_length=1, max_length=200)


class ComparisonDimension(BaseModel):
    """One explicit field that must be extracted for every paper."""

    dimension_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    description: str = Field(min_length=1, max_length=500)


class ComparisonCell(BaseModel):
    """One grounded claim or explicit gap in the evidence matrix."""

    document_id: str = Field(min_length=1)
    dimension_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    status: ComparisonCellStatus
    claim: str | None = Field(default=None, min_length=1, max_length=4000)
    citation_ids: list[CitationId] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)
    missing_reason: str | None = Field(default=None, min_length=1, max_length=1000)
    suggested_query: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_status_payload(self) -> ComparisonCell:
        """Keep supported claims and explicit gaps mutually exclusive."""
        if len(self.citation_ids) != len(set(self.citation_ids)):
            raise ValueError("comparison cell citation ids must be unique")
        if self.status is ComparisonCellStatus.SUPPORTED:
            if not self.claim or not self.citation_ids:
                raise ValueError("supported cells require a claim and citations")
            if self.missing_reason is not None or self.suggested_query is not None:
                raise ValueError("supported cells cannot contain gap fields")
        else:
            if self.claim is not None or self.citation_ids:
                raise ValueError("missing cells cannot contain a claim or citations")
            if not self.missing_reason or not self.suggested_query:
                raise ValueError("missing cells require a reason and suggested query")
        return self


class ComparisonExtractionRequest(BaseModel):
    """Evidence-only input for extracting selected dimensions from one paper."""

    document: ComparisonDocument
    dimensions: list[ComparisonDimension] = Field(min_length=1)
    evidence: list[EvidenceCitation] = Field(min_length=1)


class ComparisonExtraction(BaseModel):
    """Typed per-document response returned by the semantic model."""

    document_id: str = Field(min_length=1)
    cells: list[ComparisonCell] = Field(min_length=1)


class ComparisonSearchAttempt(BaseModel):
    """Auditable document-scoped retrieval action."""

    document_id: str = Field(min_length=1)
    dimension_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    query: str = Field(min_length=1, max_length=500)
    retrieval_round: int = Field(ge=1)


class ComparisonFailure(BaseModel):
    """Structured failure safe to persist in a checkpoint."""

    stage: ComparisonStatus
    code: ComparisonFailureCode
    message: str = Field(min_length=1)
    retryable: bool = False


class ComparisonEvent(BaseModel):
    """Compact audit event for one comparison transition."""

    status: ComparisonStatus
    message: str = Field(min_length=1)
    retrieval_round: int | None = Field(default=None, ge=1)
