"""Offline retrieval evaluation for PaperOps backends."""

from paperops.evaluation.agent import evaluate_research_agent
from paperops.evaluation.models import (
    AgentEvaluationReport,
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
    "AgentEvaluationReport",
    "EvidenceReference",
    "EvaluationDocument",
    "EvaluationQuery",
    "EvaluationSection",
    "RetrievalDataset",
    "RetrievalEvaluationReport",
    "evaluate_native_retrieval",
    "evaluate_research_agent",
    "evaluate_retrieval_backend",
]
