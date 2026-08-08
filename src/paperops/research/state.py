"""LangGraph state schema for one evidence-bounded research query."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from paperops.research.models import (
    EvidenceAssessment,
    EvidenceCitation,
    QueryRewrite,
    ResearchAnswer,
    ResearchEvent,
    ResearchFailure,
    ResearchStatus,
)


class ResearchQueryState(TypedDict, total=False):
    """Persist only bounded evidence, decisions, counters, and audit records."""

    query_id: str
    knowledge_base: str
    question: str
    current_query: str
    status: ResearchStatus
    retrieval_round: int
    rewrite_count: int
    retrieval_calls: int
    model_calls: int
    attempted_queries: list[str]
    evidence: list[EvidenceCitation]
    assessment: EvidenceAssessment
    last_rewrite: QueryRewrite
    answer: ResearchAnswer
    failure: ResearchFailure
    errors: Annotated[list[ResearchFailure], operator.add]
    events: Annotated[list[ResearchEvent], operator.add]
