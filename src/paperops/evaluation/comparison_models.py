"""Validated datasets and reports for multi-paper comparison evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from paperops.comparison.models import (
    ComparisonCell,
    ComparisonCellStatus,
    ComparisonDimension,
    ComparisonSearchAttempt,
)
from paperops.evaluation.models import (
    DatasetKind,
    EvaluationDocument,
    EvaluationQuery,
    EvidenceReference,
    RetrievalDataset,
)


class ComparisonExpectedCell(BaseModel):
    """Ground-truth status and optional paragraph evidence for one matrix cell."""

    document_id: str = Field(min_length=1)
    dimension_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    status: ComparisonCellStatus
    evidence: list[EvidenceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_evidence(self) -> ComparisonExpectedCell:
        """Require evidence only for cells labelled as supported."""
        if self.status is ComparisonCellStatus.SUPPORTED and not self.evidence:
            raise ValueError("supported comparison cells require evidence")
        if self.status is ComparisonCellStatus.MISSING and self.evidence:
            raise ValueError("missing comparison cells cannot contain evidence")
        if any(item.document_id != self.document_id for item in self.evidence):
            raise ValueError("cell evidence must belong to the cell document")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence ids must be unique within a comparison cell")
        return self


class ComparisonEvaluationTask(BaseModel):
    """One document set and its complete labelled comparison matrix."""

    task_id: str = Field(pattern=r"^[0-9A-Za-z_.-]+$")
    document_ids: list[str] = Field(min_length=2, max_length=8)
    dimensions: list[ComparisonDimension] = Field(min_length=1, max_length=6)
    expected_cells: list[ComparisonExpectedCell] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_matrix(self) -> ComparisonEvaluationTask:
        """Require a unique label for every document-by-dimension pair."""
        if len(self.document_ids) != len(set(self.document_ids)):
            raise ValueError("task document ids must be unique")
        dimension_ids = [item.dimension_id for item in self.dimensions]
        if len(dimension_ids) != len(set(dimension_ids)):
            raise ValueError("task dimension ids must be unique")
        expected_keys = {
            (document_id, dimension_id)
            for document_id in self.document_ids
            for dimension_id in dimension_ids
        }
        actual_keys = {
            (cell.document_id, cell.dimension_id) for cell in self.expected_cells
        }
        if len(actual_keys) != len(self.expected_cells):
            raise ValueError("task comparison cell keys must be unique")
        if actual_keys != expected_keys:
            raise ValueError(
                "expected_cells must cover every document-by-dimension pair exactly"
            )
        return self


class ComparisonEvaluationDataset(BaseModel):
    """Portable corpus and complete labels for comparison-Agent evaluation."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    kind: DatasetKind
    split: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    documents: list[EvaluationDocument] = Field(min_length=2)
    tasks: list[ComparisonEvaluationTask] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> ComparisonEvaluationDataset:
        """Reject duplicate ids and tasks that point outside the corpus."""
        document_ids = [item.document_id for item in self.documents]
        task_ids = [item.task_id for item in self.tasks]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("dataset document ids must be unique")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("dataset task ids must be unique")
        known = set(document_ids)
        for task in self.tasks:
            unknown = set(task.document_ids) - known
            if unknown:
                raise ValueError(
                    f"task {task.task_id} references unknown documents: "
                    f"{sorted(unknown)}"
                )
        return self

    def as_retrieval_dataset(self) -> RetrievalDataset:
        """Project labels into the shared corpus-indexing dataset contract."""
        queries: list[EvaluationQuery] = []
        for task in self.tasks:
            descriptions = {
                item.dimension_id: item.description for item in task.dimensions
            }
            for cell in task.expected_cells:
                queries.append(
                    EvaluationQuery(
                        query_id=(
                            f"{task.task_id}:{cell.document_id}:{cell.dimension_id}"
                        ),
                        text=descriptions[cell.dimension_id],
                        document_id=cell.document_id,
                        answerable=cell.status is ComparisonCellStatus.SUPPORTED,
                        evidence=cell.evidence,
                    )
                )
        return RetrievalDataset(
            name=self.name,
            version=self.version,
            kind=self.kind,
            split=self.split,
            source_url=self.source_url,
            license=self.license,
            documents=self.documents,
            queries=queries,
        )


class ComparisonCellEvaluation(BaseModel):
    """Label, prediction, and grounding metrics for one matrix cell."""

    document_id: str
    dimension_id: str
    expected_status: ComparisonCellStatus
    actual: ComparisonCell
    status_correct: bool
    grounded_correct: bool
    matched_evidence_ids: list[str] = Field(default_factory=list)
    evidence_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_recall: float | None = Field(default=None, ge=0.0, le=1.0)


class ComparisonTaskRun(BaseModel):
    """One baseline or gap-retrieval execution for a labelled task."""

    status: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    cells: list[ComparisonCellEvaluation]
    retrieval_calls: int = Field(ge=0)
    attempted_searches: list[ComparisonSearchAttempt] = Field(default_factory=list)
    evidence_count: int = Field(ge=0)
    new_evidence_count: int = Field(ge=0)
    gap_rounds: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    recovered_cell_count: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    model_latency_ms: float = Field(ge=0.0)
    failure_code: str | None = None
    failure_message: str | None = None
    stop_reason: str | None = None


class ComparisonTaskComparison(BaseModel):
    """Shared-prefix baseline and continuation result for one task."""

    task_id: str
    baseline: ComparisonTaskRun
    agent: ComparisonTaskRun


class ComparisonVariantMetrics(BaseModel):
    """Aggregate quality, refusal, cost, and latency for one variant."""

    status_accuracy: float = Field(ge=0.0, le=1.0)
    grounded_accuracy: float = Field(ge=0.0, le=1.0)
    supported_completion_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    missing_refusal_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    failure_rate: float = Field(ge=0.0, le=1.0)
    stagnant_stop_rate: float = Field(ge=0.0, le=1.0)
    average_retrieval_calls: float = Field(ge=0.0)
    average_gap_rounds: float = Field(ge=0.0)
    average_model_calls: float = Field(ge=0.0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    model_latency_ms: float = Field(ge=0.0)


class ComparisonMetricDelta(BaseModel):
    """Gap-retrieval aggregate minus the shared initial-matrix baseline."""

    status_accuracy: float
    grounded_accuracy: float
    evidence_recall: float | None = None
    average_retrieval_calls: float
    average_gap_rounds: float
    average_model_calls: float
    latency_p50_ms: float
    total_tokens: int | None = None
    baseline_missing_supported_cells: int = Field(ge=0)
    recovered_supported_cells: int = Field(ge=0)
    supported_recovery_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    incremental_tokens_per_recovery: float | None = None


class ComparisonEvaluationReport(BaseModel):
    """Reproducible shared-initial-matrix comparison-Agent report."""

    dataset_name: str
    dataset_version: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_kind: DatasetKind
    split: str
    backend: str
    index_profile: str
    model: str
    comparison_protocol: str
    document_count: int = Field(ge=2)
    task_count: int = Field(ge=1)
    cell_count: int = Field(ge=2)
    expected_supported_cell_count: int = Field(ge=0)
    expected_missing_cell_count: int = Field(ge=0)
    search_top_k: int = Field(ge=1)
    agent_max_gap_rounds: int = Field(ge=1)
    evidence_token_coverage_threshold: float = Field(ge=0.0, le=1.0)
    indexing_latency_ms: float = Field(ge=0.0)
    baseline: ComparisonVariantMetrics
    agent: ComparisonVariantMetrics
    delta: ComparisonMetricDelta
    tasks: list[ComparisonTaskComparison]
    limitations: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_counts(self) -> ComparisonEvaluationReport:
        """Keep report provenance counts aligned with task records."""
        if self.task_count != len(self.tasks):
            raise ValueError("task_count must match task records")
        cells = [cell for task in self.tasks for cell in task.baseline.cells]
        if self.cell_count != len(cells):
            raise ValueError("cell_count must match baseline cell records")
        supported = sum(
            cell.expected_status is ComparisonCellStatus.SUPPORTED for cell in cells
        )
        if self.expected_supported_cell_count != supported:
            raise ValueError("expected supported count does not match cell records")
        if self.expected_missing_cell_count != self.cell_count - supported:
            raise ValueError("expected missing count does not match cell records")
        return self
