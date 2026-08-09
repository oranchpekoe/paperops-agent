"""Control-flow and grounding tests for the multi-paper comparison graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from paperops.comparison.graph import build_comparison_graph
from paperops.comparison.models import (
    ComparisonCell,
    ComparisonCellStatus,
    ComparisonExtraction,
    ComparisonStatus,
    ComparisonStopReason,
)
from paperops.models import SearchHit, SearchRequest
from paperops.research.fakes import FakeResearchModel
from paperops.settings import Settings


class ScriptedComparisonRetrieval:
    """Return document-and-query-specific hits while retaining every request."""

    name = "scripted-comparison"

    def __init__(
        self,
        outcomes: dict[tuple[str, str], list[SearchHit] | Exception],
    ) -> None:
        self.outcomes = outcomes
        self.search_calls: list[SearchRequest] = []
        self.calls_by_key: defaultdict[tuple[str, str], int] = defaultdict(int)

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        self.search_calls.append(request)
        key = (request.expected_document_id or "", request.query)
        self.calls_by_key[key] += 1
        outcome = self.outcomes.get(key, [])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def ingest(self, request: Any) -> Any:  # pragma: no cover - protocol only
        raise NotImplementedError


def _hit(document_id: str, chunk_id: str, content: str) -> SearchHit:
    return SearchHit(
        document_id=document_id,
        chunk_id=chunk_id,
        content=content,
        score=1.0,
    )


def _settings(**updates: Any) -> Settings:
    return Settings(
        _env_file=None,
        comparison_search_top_k=2,
        comparison_max_gap_rounds=1,
        comparison_max_evidence_chars=12000,
        **updates,
    )


def _input(*, dimensions: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "knowledge_base": "uav-papers",
        "documents": [
            {"document_id": "doc-a", "label": "Paper A"},
            {"document_id": "doc-b", "label": "Paper B"},
        ],
        "dimensions": dimensions
        or [
            {
                "dimension_id": "method",
                "description": "Which method is proposed?",
            },
            {
                "dimension_id": "dataset",
                "description": "Which dataset is evaluated?",
            },
        ],
    }


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


@pytest.mark.asyncio
async def test_initial_matrix_is_document_scoped_and_complete() -> None:
    method = "Which method is proposed?"
    dataset = "Which dataset is evaluated?"
    retrieval = ScriptedComparisonRetrieval(
        {
            ("doc-a", method): [_hit("doc-a", "a-1", "A method and dataset.")],
            ("doc-a", dataset): [_hit("doc-a", "a-1", "A method and dataset.")],
            ("doc-b", method): [_hit("doc-b", "b-1", "B method and dataset.")],
            ("doc-b", dataset): [_hit("doc-b", "b-1", "B method and dataset.")],
        }
    )
    model = FakeResearchModel()
    graph = build_comparison_graph(
        retrieval=retrieval,
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(), _config("complete-matrix"))

    assert result["status"] is ComparisonStatus.COMPLETED
    assert result["stop_reason"] is ComparisonStopReason.ALL_CELLS_SUPPORTED
    assert len(result["cells"]) == 4
    assert all(
        cell.status is ComparisonCellStatus.SUPPORTED for cell in result["cells"]
    )
    assert result["retrieval_calls"] == 4
    assert result["model_calls"] == 2
    assert result["recovered_cell_count"] == 0
    assert {call.expected_document_id for call in retrieval.search_calls} == {
        "doc-a",
        "doc-b",
    }


@pytest.mark.asyncio
async def test_gap_retrieval_recovers_only_the_missing_cell() -> None:
    dimension = {
        "dimension_id": "reward",
        "description": "How is the reward designed?",
    }
    retrieval = ScriptedComparisonRetrieval(
        {
            ("doc-a", dimension["description"]): [
                _hit("doc-a", "a-overview", "The overview omits reward details.")
            ],
            ("doc-b", dimension["description"]): [
                _hit("doc-b", "b-reward", "Paper B uses a dense reward.")
            ],
            ("doc-a", "reward shaping terminal capture collision penalty"): [
                _hit(
                    "doc-a",
                    "a-reward",
                    "The reward combines capture and collision penalties.",
                )
            ],
        }
    )
    model = FakeResearchModel(
        comparison_extractions=[
            ComparisonExtraction(
                document_id="doc-a",
                cells=[
                    ComparisonCell(
                        document_id="doc-a",
                        dimension_id="reward",
                        status=ComparisonCellStatus.MISSING,
                        confidence=0.95,
                        missing_reason="The reward formula is absent.",
                        suggested_query=(
                            "reward shaping terminal capture collision penalty"
                        ),
                    )
                ],
            ),
            ComparisonExtraction(
                document_id="doc-b",
                cells=[
                    ComparisonCell(
                        document_id="doc-b",
                        dimension_id="reward",
                        status=ComparisonCellStatus.SUPPORTED,
                        claim="Paper B uses a dense reward [E2].",
                        citation_ids=["E2"],
                        confidence=0.9,
                    )
                ],
            ),
            ComparisonExtraction(
                document_id="doc-a",
                cells=[
                    ComparisonCell(
                        document_id="doc-a",
                        dimension_id="reward",
                        status=ComparisonCellStatus.SUPPORTED,
                        claim="Paper A combines capture and collision terms [E3].",
                        citation_ids=["E3"],
                        confidence=0.92,
                    )
                ],
            ),
        ]
    )
    graph = build_comparison_graph(
        retrieval=retrieval,
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        _input(dimensions=[dimension]),
        _config("recover-cell"),
    )

    assert result["status"] is ComparisonStatus.COMPLETED
    assert result["stop_reason"] is ComparisonStopReason.ALL_CELLS_SUPPORTED
    assert result["retrieval_calls"] == 3
    assert result["model_calls"] == 3
    assert result["recovered_cell_count"] == 1
    assert len(model.comparison_calls[2].dimensions) == 1
    assert model.comparison_calls[2].document.document_id == "doc-a"
    assert retrieval.search_calls[-1].expected_document_id == "doc-a"


@pytest.mark.asyncio
async def test_duplicate_gap_evidence_stops_without_reextracting() -> None:
    description = "Which simulator is used?"
    initial_a = _hit("doc-a", "a-1", "A generic experiment description.")
    retrieval = ScriptedComparisonRetrieval(
        {
            ("doc-a", description): [initial_a],
            ("doc-b", description): [_hit("doc-b", "b-1", "Paper B uses Simulator B.")],
            ("doc-a", "simulator environment implementation details"): [initial_a],
        }
    )
    model = FakeResearchModel(
        comparison_extractions=[
            ComparisonExtraction(
                document_id="doc-a",
                cells=[
                    ComparisonCell(
                        document_id="doc-a",
                        dimension_id="simulator",
                        status=ComparisonCellStatus.MISSING,
                        confidence=0.9,
                        missing_reason="The simulator is not named.",
                        suggested_query="simulator environment implementation details",
                    )
                ],
            ),
            ComparisonExtraction(
                document_id="doc-b",
                cells=[
                    ComparisonCell(
                        document_id="doc-b",
                        dimension_id="simulator",
                        status=ComparisonCellStatus.SUPPORTED,
                        claim="Paper B uses Simulator B [E2].",
                        citation_ids=["E2"],
                        confidence=0.95,
                    )
                ],
            ),
        ]
    )
    graph = build_comparison_graph(
        retrieval=retrieval,
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        _input(dimensions=[{"dimension_id": "simulator", "description": description}]),
        _config("stagnant-cell"),
    )

    assert result["status"] is ComparisonStatus.COMPLETED
    assert result["stop_reason"] is ComparisonStopReason.STAGNANT_RETRIEVAL
    assert result["new_evidence_count"] == 0
    assert result["retrieval_calls"] == 3
    assert result["model_calls"] == 2
    assert len(model.comparison_calls) == 2


@pytest.mark.asyncio
async def test_cross_document_citation_fails_closed() -> None:
    description = "Which method is proposed?"
    retrieval = ScriptedComparisonRetrieval(
        {
            ("doc-a", description): [_hit("doc-a", "a-1", "Paper A method.")],
            ("doc-b", description): [_hit("doc-b", "b-1", "Paper B method.")],
        }
    )
    model = FakeResearchModel(
        comparison_extractions=[
            ComparisonExtraction(
                document_id="doc-a",
                cells=[
                    ComparisonCell(
                        document_id="doc-a",
                        dimension_id="method",
                        status=ComparisonCellStatus.SUPPORTED,
                        claim="This incorrectly cites Paper B [E2].",
                        citation_ids=["E2"],
                        confidence=0.99,
                    )
                ],
            )
        ]
    )
    graph = build_comparison_graph(
        retrieval=retrieval,
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        _input(dimensions=[{"dimension_id": "method", "description": description}]),
        _config("cross-document-citation"),
    )

    assert result["status"] is ComparisonStatus.FAILED
    assert result["failure"].code.value == "citation_validation_error"
    assert "cross-document" in result["failure"].message
