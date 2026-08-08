"""Tests for fair one-shot versus bounded-Agent evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from paperops.evaluation.agent import evaluate_research_agent
from paperops.evaluation.models import (
    DatasetKind,
    EvaluationDocument,
    EvaluationQuery,
    EvaluationSection,
    EvidenceReference,
    RetrievalDataset,
)
from paperops.models import IngestRequest, IngestResult, SearchHit, SearchRequest
from paperops.research.fakes import FakeResearchModel
from paperops.research.models import EvidenceAssessment, QueryRewrite, ResearchAnswer
from paperops.settings import Settings


def _insufficient() -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficient=False,
        confidence=0.9,
        rationale="The current passage does not contain the labelled answer.",
        missing_aspects=["target evidence"],
    )


def _sufficient() -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficient=True,
        confidence=0.95,
        rationale="The target evidence is present.",
    )


class EvaluationBackend:
    """Index one document and return deterministic passages by exact query."""

    name = "evaluation-scripted"

    async def ingest(self, request: IngestRequest) -> IngestResult:
        document_id = f"backend-{Path(request.markdown_path).stem}"
        return IngestResult(
            document_id=document_id,
            idempotency_key=request.idempotency_key,
            created=True,
            chunk_count=4,
        )

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        passages = {
            "Which method does the study use?": (
                "initial",
                "The study discusses several engineering systems.",
            ),
            "centralized critic training": (
                "gold",
                "The method uses a centralized critic during training.",
            ),
            "Which year was the method launched?": (
                "unknown-0",
                "The study discusses several engineering systems.",
            ),
            "launch year source one": (
                "unknown-1",
                "No product launch date is reported.",
            ),
            "launch year source two": (
                "unknown-2",
                "The conclusion does not mention commercial history.",
            ),
        }
        chunk_id, content = passages[request.query]
        return [
            SearchHit(
                document_id="backend-paper-1",
                chunk_id=chunk_id,
                content=content,
                score=0.9,
            )
        ]


def _dataset() -> RetrievalDataset:
    evidence_text = "The method uses a centralized critic during training."
    return RetrievalDataset(
        name="agent-evaluation-smoke",
        version="1.0",
        kind=DatasetKind.SMOKE_FIXTURE,
        split="test",
        source_url="local://agent-evaluation-test",
        license="Original test data",
        documents=[
            EvaluationDocument(
                document_id="paper-1",
                title="Centralized Training",
                sections=[
                    EvaluationSection(title="Method", paragraphs=[evidence_text])
                ],
            )
        ],
        queries=[
            EvaluationQuery(
                query_id="answerable",
                text="Which method does the study use?",
                evidence=[
                    EvidenceReference(
                        evidence_id="ev-method",
                        document_id="paper-1",
                        text=evidence_text,
                    )
                ],
            ),
            EvaluationQuery(
                query_id="unanswerable",
                text="Which year was the method launched?",
                answerable=False,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_agent_evaluation_measures_recovery_refusal_and_cost(
    tmp_path: Path,
) -> None:
    baseline_model = FakeResearchModel(assessments=[_insufficient(), _insufficient()])
    agent_model = FakeResearchModel(
        assessments=[
            _insufficient(),
            _sufficient(),
            _insufficient(),
            _insufficient(),
            _insufficient(),
        ],
        rewrites=[
            QueryRewrite(
                query="centralized critic training",
                reason="Target the training architecture.",
            ),
            QueryRewrite(
                query="launch year source one",
                reason="Search for a launch date.",
            ),
            QueryRewrite(
                query="launch year source two",
                reason="Try a second date-specific query.",
            ),
        ],
        answers=[
            ResearchAnswer(
                text="The method uses a centralized critic [E2].",
                citation_ids=["E2"],
            )
        ],
    )
    settings = Settings(
        _env_file=None,
        native_index_db=tmp_path / "index.db",
        research_search_top_k=1,
        research_max_rewrites=2,
        research_min_evidence_hits=1,
    )

    report = await evaluate_research_agent(
        _dataset(),
        backend=EvaluationBackend(),
        baseline_model=baseline_model,
        agent_model=agent_model,
        settings=settings,
        work_dir=tmp_path / "evaluation",
        index_profile="scripted-v1",
    )

    assert report.answerable_query_count == 1
    assert report.unanswerable_query_count == 1
    assert report.baseline.outcome_accuracy == 0.5
    assert report.agent.outcome_accuracy == 1.0
    assert report.delta.outcome_accuracy == 0.5
    assert report.baseline.evidence_recall == 0.0
    assert report.agent.evidence_recall == 1.0
    assert report.agent.unanswerable_refusal_rate == 1.0
    assert report.agent.citation_precision == 1.0
    assert report.agent.citation_recall == 1.0
    assert report.agent.average_retrieval_calls == 2.5
    assert report.agent.average_rewrites == 1.5
    assert report.agent.total_tokens is None
    assert report.queries[0].agent.matched_evidence_ids == ["ev-method"]
    assert report.queries[1].agent.status == "insufficient_evidence"
