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
    evidence: list[EvidenceReference] = Field(min_length=1)


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
