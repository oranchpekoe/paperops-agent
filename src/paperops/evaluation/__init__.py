"""Offline retrieval evaluation for PaperOps backends."""

from paperops.evaluation.models import (
    DatasetKind,
    EvaluationDocument,
    EvaluationQuery,
    EvaluationSection,
    EvidenceReference,
    RetrievalDataset,
    RetrievalEvaluationReport,
)
from paperops.evaluation.retrieval import (
    evaluate_native_retrieval,
    evaluate_retrieval_backend,
)

__all__ = [
    "DatasetKind",
    "EvidenceReference",
    "EvaluationDocument",
    "EvaluationQuery",
    "EvaluationSection",
    "RetrievalDataset",
    "RetrievalEvaluationReport",
    "evaluate_native_retrieval",
    "evaluate_retrieval_backend",
]
