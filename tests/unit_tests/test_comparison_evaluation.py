"""Tests for shared-initial-matrix comparison evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from paperops.comparison.models import (
    ComparisonCell,
    ComparisonCellStatus,
    ComparisonDimension,
    ComparisonExtraction,
)
from paperops.evaluation.comparison import evaluate_comparison_agent
from paperops.evaluation.comparison_models import (
    ComparisonEvaluationDataset,
    ComparisonEvaluationTask,
    ComparisonExpectedCell,
)
from paperops.evaluation.models import (
    DatasetKind,
    EvaluationDocument,
    EvaluationSection,
    EvidenceReference,
)
from paperops.models import IngestRequest, IngestResult, SearchHit, SearchRequest
from paperops.research.fakes import FakeResearchModel
from paperops.settings import Settings


class ComparisonEvaluationBackend:
    """Return deterministic document-scoped chunks for initial and gap queries."""

    name = "comparison-evaluation-scripted"

    async def ingest(self, request: IngestRequest) -> IngestResult:
        document_id = f"backend-{Path(request.markdown_path).stem}"
        return IngestResult(
            document_id=document_id,
            idempotency_key=request.idempotency_key,
            created=True,
            chunk_count=3,
        )

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        passages = {
            ("backend-paper-a", "training architecture"): (
                "a-background",
                "Paper A studies multi-agent systems in simulation.",
            ),
            ("backend-paper-a", "reported deployment year"): (
                "a-no-year",
                "Paper A does not report a deployment year.",
            ),
            ("backend-paper-b", "training architecture"): (
                "b-method",
                "Paper B trains independent policies with decentralized rewards.",
            ),
            ("backend-paper-b", "reported deployment year"): (
                "b-no-year",
                "Paper B does not report a deployment year.",
            ),
            ("backend-paper-a", "centralized critic training"): (
                "a-method",
                "Paper A uses a centralized critic during training.",
            ),
            ("backend-paper-a", "paper a deployment year"): (
                "a-no-year",
                "Paper A does not report a deployment year.",
            ),
            ("backend-paper-b", "paper b deployment year"): (
                "b-no-year",
                "Paper B does not report a deployment year.",
            ),
        }
        chunk_id, content = passages[(request.expected_document_id, request.query)]
        return [
            SearchHit(
                document_id=request.expected_document_id or "unexpected",
                chunk_id=chunk_id,
                content=content,
                score=0.9,
            )
        ]


def _missing(
    document_id: str,
    dimension_id: str,
    query: str,
) -> ComparisonCell:
    return ComparisonCell(
        document_id=document_id,
        dimension_id=dimension_id,
        status=ComparisonCellStatus.MISSING,
        confidence=0.9,
        missing_reason="The current evidence does not contain this field.",
        suggested_query=query,
    )


def _supported(
    document_id: str,
    dimension_id: str,
    citation_id: str,
    claim: str,
) -> ComparisonCell:
    return ComparisonCell(
        document_id=document_id,
        dimension_id=dimension_id,
        status=ComparisonCellStatus.SUPPORTED,
        confidence=0.95,
        claim=f"{claim} [{citation_id}].",
        citation_ids=[citation_id],
    )


def _dataset() -> ComparisonEvaluationDataset:
    return ComparisonEvaluationDataset(
        name="comparison-agent-smoke",
        version="1.0",
        kind=DatasetKind.SMOKE_FIXTURE,
        split="test",
        source_url="local://comparison-agent-test",
        license="Original test data",
        documents=[
            EvaluationDocument(
                document_id="paper-a",
                title="Paper A",
                sections=[
                    EvaluationSection(
                        title="Method",
                        paragraphs=[
                            "Paper A uses a centralized critic during training.",
                            "Paper A does not report a deployment year.",
                        ],
                    )
                ],
            ),
            EvaluationDocument(
                document_id="paper-b",
                title="Paper B",
                sections=[
                    EvaluationSection(
                        title="Method",
                        paragraphs=[
                            "Paper B trains independent policies with decentralized rewards.",
                            "Paper B does not report a deployment year.",
                        ],
                    )
                ],
            ),
        ],
        tasks=[
            ComparisonEvaluationTask(
                task_id="method-and-year",
                document_ids=["paper-a", "paper-b"],
                dimensions=[
                    ComparisonDimension(
                        dimension_id="method",
                        description="training architecture",
                    ),
                    ComparisonDimension(
                        dimension_id="deployment_year",
                        description="reported deployment year",
                    ),
                ],
                expected_cells=[
                    ComparisonExpectedCell(
                        document_id="paper-a",
                        dimension_id="method",
                        status=ComparisonCellStatus.SUPPORTED,
                        evidence=[
                            EvidenceReference(
                                evidence_id="a-method",
                                document_id="paper-a",
                                text=(
                                    "Paper A uses a centralized critic during training."
                                ),
                            )
                        ],
                    ),
                    ComparisonExpectedCell(
                        document_id="paper-a",
                        dimension_id="deployment_year",
                        status=ComparisonCellStatus.MISSING,
                    ),
                    ComparisonExpectedCell(
                        document_id="paper-b",
                        dimension_id="method",
                        status=ComparisonCellStatus.SUPPORTED,
                        evidence=[
                            EvidenceReference(
                                evidence_id="b-method",
                                document_id="paper-b",
                                text=(
                                    "Paper B trains independent policies with "
                                    "decentralized rewards."
                                ),
                            )
                        ],
                    ),
                    ComparisonExpectedCell(
                        document_id="paper-b",
                        dimension_id="deployment_year",
                        status=ComparisonCellStatus.MISSING,
                    ),
                ],
            )
        ],
    )


@pytest.mark.asyncio
async def test_comparison_evaluation_measures_annotated_recovery_and_cost(
    tmp_path: Path,
) -> None:
    model = FakeResearchModel(
        comparison_extractions=[
            ComparisonExtraction(
                document_id="backend-paper-a",
                cells=[
                    _missing(
                        "backend-paper-a", "method", "centralized critic training"
                    ),
                    _missing(
                        "backend-paper-a",
                        "deployment_year",
                        "paper a deployment year",
                    ),
                ],
            ),
            ComparisonExtraction(
                document_id="backend-paper-b",
                cells=[
                    _supported(
                        "backend-paper-b",
                        "method",
                        "E3",
                        "Paper B uses decentralized rewards",
                    ),
                    _missing(
                        "backend-paper-b",
                        "deployment_year",
                        "paper b deployment year",
                    ),
                ],
            ),
            ComparisonExtraction(
                document_id="backend-paper-a",
                cells=[
                    _supported(
                        "backend-paper-a",
                        "method",
                        "E5",
                        "Paper A uses a centralized critic",
                    ),
                    _missing(
                        "backend-paper-a",
                        "deployment_year",
                        "paper a deployment year",
                    ),
                ],
            ),
            ComparisonExtraction(
                document_id="backend-paper-b",
                cells=[
                    _missing(
                        "backend-paper-b",
                        "deployment_year",
                        "paper b deployment year",
                    )
                ],
            ),
        ]
    )
    settings = Settings(
        _env_file=None,
        native_index_db=tmp_path / "index.db",
        comparison_search_top_k=1,
        comparison_max_gap_rounds=1,
    )

    report = await evaluate_comparison_agent(
        _dataset(),
        backend=ComparisonEvaluationBackend(),
        model=model,
        settings=settings,
        work_dir=tmp_path / "evaluation",
        index_profile="scripted-v1",
    )

    assert (
        report.comparison_protocol == "shared_initial_matrix_then_gap_continuation_v1"
    )
    assert report.baseline.annotation_grounded_accuracy == 0.75
    assert report.agent.annotation_grounded_accuracy == 1.0
    assert report.delta.annotation_grounded_accuracy == 0.25
    assert report.baseline.evidence_recall == 0.5
    assert report.agent.evidence_recall == 1.0
    assert report.agent.missing_refusal_rate == 1.0
    assert report.delta.baseline_missing_supported_cells == 1
    assert report.delta.recovered_supported_cells == 1
    assert report.delta.supported_recovery_rate == 1.0
    assert report.baseline.average_retrieval_calls == 4
    assert report.agent.average_retrieval_calls == 7
    assert report.baseline.average_model_calls == 2
    assert report.agent.average_model_calls == 4
    assert report.agent.total_tokens is None
    assert len(report.tasks[0].baseline.attempted_searches) == 4
    assert len(report.tasks[0].agent.attempted_searches) == 7
    assert {
        attempt.document_id for attempt in report.tasks[0].agent.attempted_searches
    } == {"paper-a", "paper-b"}
    assert len(model.comparison_calls) == 4


def test_comparison_dataset_requires_a_complete_matrix() -> None:
    dataset = _dataset().model_dump()
    dataset["tasks"][0]["expected_cells"].pop()

    with pytest.raises(ValueError, match="every document-by-dimension pair"):
        ComparisonEvaluationDataset.model_validate(dataset)
