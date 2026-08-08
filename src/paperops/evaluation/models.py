"""Validated schemas for retrieval datasets and evaluation reports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class DatasetKind(StrEnum):
    """Distinguish a wiring fixture from a reportable benchmark."""

    SMOKE_FIXTURE = "smoke_fixture"
    BENCHMARK = "benchmark"


class EvaluationSection(BaseModel):
    """One named paper section containing ordered paragraphs."""

    title: str = Field(min_length=1)
    paragraphs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_blank_paragraphs(self) -> EvaluationSection:
        """Require every paragraph to contain indexable text."""
        if any(not paragraph.strip() for paragraph in self.paragraphs):
            raise ValueError("section paragraphs must not be blank")
        return self


class EvaluationDocument(BaseModel):
    """A logical research paper rendered to Markdown before indexing."""

    document_id: str = Field(pattern=r"^[0-9A-Za-z_.-]+$")
    title: str = Field(min_length=1)
    abstract: str = ""
    sections: list[EvaluationSection] = Field(min_length=1)

    def to_markdown(self) -> str:
        """Render the structured paper without embedding evaluation labels."""
        blocks = [f"# {self.title.strip()}"]
        if self.abstract.strip():
            blocks.extend(["## Abstract", self.abstract.strip()])
        for section in self.sections:
            blocks.extend(
                [
                    f"## {section.title.strip()}",
                    "\n\n".join(paragraph.strip() for paragraph in section.paragraphs),
                ]
            )
        return "\n\n".join(blocks) + "\n"


class EvidenceReference(BaseModel):
    """An independently annotated evidence paragraph for one query."""

    evidence_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    relevance: int = Field(default=1, ge=1, le=3)


class EvaluationQuery(BaseModel):
    """One collection-wide query and its relevant evidence paragraphs."""

    query_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    document_id: str | None = None
    answerable: bool = True
    evidence: list[EvidenceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_answerability(self) -> EvaluationQuery:
        """Keep answerability labels consistent with paragraph evidence."""
        if self.answerable and not self.evidence:
            raise ValueError("answerable queries must contain evidence")
        if not self.answerable and self.evidence:
            raise ValueError("unanswerable queries must not contain evidence")
        return self


class RetrievalDataset(BaseModel):
    """Portable corpus, questions, and paragraph-level relevance labels."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    kind: DatasetKind
    split: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    documents: list[EvaluationDocument] = Field(min_length=1)
    queries: list[EvaluationQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> RetrievalDataset:
        """Reject duplicate identifiers and evidence pointing outside the corpus."""
        document_ids = [document.document_id for document in self.documents]
        query_ids = [query.query_id for query in self.queries]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id values must be unique")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be unique")
        known_documents = set(document_ids)
        for query in self.queries:
            if (
                query.document_id is not None
                and query.document_id not in known_documents
            ):
                raise ValueError(
                    f"query {query.query_id} scopes to unknown document: "
                    f"{query.document_id}"
                )
            evidence_ids = [item.evidence_id for item in query.evidence]
            if len(evidence_ids) != len(set(evidence_ids)):
                raise ValueError(
                    f"evidence_id values must be unique within query {query.query_id}"
                )
            unknown = {
                item.document_id
                for item in query.evidence
                if item.document_id not in known_documents
            }
            if unknown:
                raise ValueError(
                    f"query {query.query_id} references unknown documents: "
                    f"{sorted(unknown)}"
                )
            if query.document_id is not None:
                outside_scope = {
                    item.document_id
                    for item in query.evidence
                    if item.document_id != query.document_id
                }
                if outside_scope:
                    raise ValueError(
                        f"query {query.query_id} contains evidence outside its "
                        f"document scope: {sorted(outside_scope)}"
                    )
        return self


class RankedEvidenceHit(BaseModel):
    """One retrieved chunk with any evidence units it covers."""

    rank: int = Field(ge=1)
    document_id: str = Field(min_length=1)
    chunk_id: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    matched_evidence_ids: list[str] = Field(default_factory=list)


class QueryEvaluation(BaseModel):
    """Per-query retrieval metrics and traceable ranking output."""

    query_id: str
    query: str
    latency_ms: float = Field(ge=0.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    recall_at_k: dict[str, float]
    ndcg_at_k: dict[str, float]
    hits: list[RankedEvidenceHit]


class AggregateRetrievalMetrics(BaseModel):
    """Macro-averaged quality and latency measurements."""

    recall_at_k: dict[str, float]
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: dict[str, float]
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)


class RetrievalEvaluationReport(BaseModel):
    """Reproducible output for one backend and dataset configuration."""

    dataset_name: str
    dataset_version: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_kind: DatasetKind
    split: str
    backend: str
    index_profile: str = Field(min_length=1)
    document_count: int = Field(ge=1)
    query_count: int = Field(ge=1)
    top_k: list[int]
    chunk_size_chars: int = Field(ge=1)
    chunk_overlap_chars: int = Field(ge=0)
    evidence_token_coverage_threshold: float = Field(ge=0.0, le=1.0)
    indexing_latency_ms: float = Field(ge=0.0)
    aggregate: AggregateRetrievalMetrics
    queries: list[QueryEvaluation]


class AgentRunEvaluation(BaseModel):
    """Metrics from one graph configuration on one labelled question."""

    status: str = Field(min_length=1)
    outcome_correct: bool
    latency_ms: float = Field(ge=0.0)
    evidence_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    matched_evidence_ids: list[str] = Field(default_factory=list)
    retrieval_calls: int = Field(ge=0)
    new_evidence_count: int = Field(ge=0)
    rewrite_count: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    attempted_queries: list[str] = Field(default_factory=list)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    model_latency_ms: float = Field(default=0.0, ge=0.0)
    assessment_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    assessment_rationale: str | None = None
    selected_citation_ids: list[str] = Field(default_factory=list)
    answer_text: str | None = None
    answer_citation_ids: list[str] = Field(default_factory=list)
    failure_code: str | None = None
    failure_message: str | None = None
    stop_reason: str | None = None


class AgentQueryComparison(BaseModel):
    """Comparable one-shot and bounded-agent results for one query."""

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    answerable: bool
    baseline: AgentRunEvaluation
    agent: AgentRunEvaluation


class AgentVariantMetrics(BaseModel):
    """Aggregate outcome, grounding, cost, and latency for one variant."""

    outcome_accuracy: float = Field(ge=0.0, le=1.0)
    answerable_completion_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    unanswerable_refusal_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    failure_rate: float = Field(ge=0.0, le=1.0)
    stagnant_stop_rate: float = Field(ge=0.0, le=1.0)
    average_retrieval_calls: float = Field(ge=0.0)
    average_rewrites: float = Field(ge=0.0)
    average_model_calls: float = Field(ge=0.0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    model_latency_ms: float = Field(default=0.0, ge=0.0)


class AgentMetricDelta(BaseModel):
    """Bounded-agent aggregate minus the one-shot baseline."""

    outcome_accuracy: float
    evidence_recall: float | None = None
    average_retrieval_calls: float
    average_rewrites: float
    average_model_calls: float
    latency_p50_ms: float
    total_tokens: int | None = None
    baseline_missed_answerable: int = Field(ge=0)
    recovered_answerable: int = Field(ge=0)
    answerable_recovery_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    incremental_tokens_per_recovery: float | None = None


class AgentEvaluationReport(BaseModel):
    """Reproducible one-shot versus bounded-agent comparison report."""

    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_kind: DatasetKind
    split: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    index_profile: str = Field(min_length=1)
    model: str = Field(min_length=1)
    comparison_protocol: str = Field(min_length=1)
    document_count: int = Field(ge=1)
    query_count: int = Field(ge=1)
    answerable_query_count: int = Field(ge=0)
    unanswerable_query_count: int = Field(ge=0)
    search_top_k: int = Field(ge=1)
    baseline_max_rewrites: int = Field(default=0, ge=0, le=0)
    agent_max_rewrites: int = Field(ge=1)
    evidence_token_coverage_threshold: float = Field(ge=0.0, le=1.0)
    indexing_latency_ms: float = Field(ge=0.0)
    baseline: AgentVariantMetrics
    agent: AgentVariantMetrics
    delta: AgentMetricDelta
    queries: list[AgentQueryComparison]
    limitations: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_query_counts(self) -> AgentEvaluationReport:
        """Keep report provenance counts aligned with per-query records."""
        if self.query_count != len(self.queries):
            raise ValueError("query_count must match the number of query records")
        answerable = sum(query.answerable for query in self.queries)
        if self.answerable_query_count != answerable:
            raise ValueError("answerable_query_count does not match query records")
        if self.unanswerable_query_count != self.query_count - answerable:
            raise ValueError("unanswerable_query_count does not match query records")
        return self
