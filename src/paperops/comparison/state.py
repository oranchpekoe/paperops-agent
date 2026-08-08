"""Checkpointed state for multi-paper comparison execution."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from paperops.comparison.models import (
    ComparisonCell,
    ComparisonDimension,
    ComparisonDocument,
    ComparisonEvent,
    ComparisonFailure,
    ComparisonSearchAttempt,
    ComparisonStatus,
    ComparisonStopReason,
)
from paperops.research.models import EvidenceCitation


class ComparisonState(TypedDict, total=False):
    """Compact state persisted between retrieval and extraction stages."""

    comparison_id: str
    knowledge_base: str
    documents: list[ComparisonDocument]
    dimensions: list[ComparisonDimension]
    status: ComparisonStatus
    retrieval_round: int
    gap_round: int
    retrieval_calls: int
    model_calls: int
    new_evidence_count: int
    attempted_searches: list[ComparisonSearchAttempt]
    evidence: list[EvidenceCitation]
    initial_cells: list[ComparisonCell]
    cells: list[ComparisonCell]
    recovered_cell_count: int
    stop_reason: ComparisonStopReason
    failure: ComparisonFailure | None
    errors: Annotated[list[ComparisonFailure], operator.add]
    events: Annotated[list[ComparisonEvent], operator.add]
