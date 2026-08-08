"""Tests for the bounded evidence-gathering research graph."""

from __future__ import annotations

from collections import defaultdict

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from paperops.models import IngestRequest, IngestResult, SearchHit, SearchRequest
from paperops.research.fakes import FakeResearchModel
from paperops.research.graph import build_research_graph
from paperops.research.models import (
    EvidenceAssessment,
    QueryRewrite,
    ResearchAnswer,
    ResearchFailureCode,
    ResearchStatus,
)
from paperops.settings import Settings


class ScriptedRetrieval:
    """Return deterministic hits keyed by exact query text."""

    name = "scripted"

    def __init__(self, responses: dict[str, list[SearchHit]]) -> None:
        self.responses = responses
        self.search_calls: list[SearchRequest] = []
        self.calls_by_query: defaultdict[str, int] = defaultdict(int)

    async def ingest(self, request: IngestRequest) -> IngestResult:
        raise AssertionError("research graph must not ingest documents")

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        self.search_calls.append(request)
        self.calls_by_query[request.query] += 1
        return self.responses.get(request.query, [])


def _hit(
    chunk_id: str = "chunk-1", content: str = "The policy improves recall."
) -> SearchHit:
    return SearchHit(
        document_id="paper-1",
        chunk_id=chunk_id,
        content=content,
        score=0.9,
        heading_path=["Results"],
    )


def _settings(**updates: object) -> Settings:
    return Settings(
        _env_file=None,
        research_max_rewrites=2,
        research_min_evidence_hits=1,
        research_min_assessment_confidence=0.65,
        **updates,
    )


def _input() -> dict[str, str]:
    return {
        "knowledge_base": "uav-papers",
        "question": "Which policy improves retrieval recall?",
    }


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


@pytest.mark.asyncio
async def test_one_round_answer_has_resolvable_inline_citation() -> None:
    retrieval = ScriptedRetrieval({_input()["question"]: [_hit()]})
    model = FakeResearchModel()
    graph = build_research_graph(
        retrieval=retrieval,
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(), _config("one-round"))

    assert result["status"] is ResearchStatus.COMPLETED
    assert result["retrieval_calls"] == 1
    assert result["model_calls"] == 2
    assert result["answer"].citation_ids == ["E1"]
    assert "[E1]" in result["answer"].text
    assert result["evidence"][0].chunk_id == "chunk-1"


@pytest.mark.asyncio
async def test_insufficient_evidence_rewrites_then_recovers() -> None:
    question = _input()["question"]
    rewritten = "hybrid retrieval recall results"
    retrieval = ScriptedRetrieval({question: [], rewritten: [_hit()]})
    model = FakeResearchModel(
        rewrites=[QueryRewrite(query=rewritten, reason="Target the results section.")]
    )
    graph = build_research_graph(
        retrieval=retrieval,
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(), _config("rewrite-recovery"))

    assert result["status"] is ResearchStatus.COMPLETED
    assert result["attempted_queries"] == [question, rewritten]
    assert result["rewrite_count"] == 1
    assert result["retrieval_calls"] == 2
    assert result["model_calls"] == 3


@pytest.mark.asyncio
async def test_rewrite_budget_exhaustion_refuses_without_answer() -> None:
    question = _input()["question"]
    model = FakeResearchModel(
        rewrites=[
            QueryRewrite(query="rewrite one", reason="First missing aspect."),
            QueryRewrite(query="rewrite two", reason="Second missing aspect."),
        ]
    )
    graph = build_research_graph(
        retrieval=ScriptedRetrieval({}),
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(), _config("bounded-refusal"))

    assert result["status"] is ResearchStatus.INSUFFICIENT_EVIDENCE
    assert result["rewrite_count"] == 2
    assert result["retrieval_calls"] == 3
    assert result["attempted_queries"] == [question, "rewrite one", "rewrite two"]
    assert "answer" not in result
    assert model.answer_calls == []


@pytest.mark.asyncio
async def test_low_confidence_assessment_spends_bounded_rewrite_budget() -> None:
    question = _input()["question"]
    retrieval = ScriptedRetrieval(
        {
            question: [_hit()],
            "more evidence": [_hit("chunk-2", "A second result confirms the finding.")],
        }
    )
    model = FakeResearchModel(
        assessments=[
            EvidenceAssessment(
                sufficient=True,
                confidence=0.4,
                rationale="Evidence is plausible but uncertain.",
            ),
            EvidenceAssessment(
                sufficient=True,
                confidence=0.9,
                rationale="Independent evidence is sufficient.",
            ),
        ],
        rewrites=[QueryRewrite(query="more evidence", reason="Increase confidence.")],
    )
    graph = build_research_graph(
        retrieval=retrieval,
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(), _config("low-confidence"))

    assert result["status"] is ResearchStatus.COMPLETED
    assert result["rewrite_count"] == 1
    assert len(result["evidence"]) == 2


@pytest.mark.asyncio
async def test_invalid_citation_fails_closed() -> None:
    question = _input()["question"]
    model = FakeResearchModel(
        answers=[
            ResearchAnswer(
                text="This cites evidence that was never retrieved [E99].",
                citation_ids=["E99"],
            )
        ]
    )
    graph = build_research_graph(
        retrieval=ScriptedRetrieval({question: [_hit()]}),
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(), _config("invalid-citation"))

    assert result["status"] is ResearchStatus.FAILED
    assert result["failure"].code is ResearchFailureCode.CITATION_VALIDATION_ERROR


@pytest.mark.asyncio
async def test_checkpoint_resume_does_not_repeat_completed_retrieval() -> None:
    question = _input()["question"]
    retrieval = ScriptedRetrieval({question: [_hit()]})
    graph = build_research_graph(
        retrieval=retrieval,
        model=FakeResearchModel(),
        settings=_settings(),
        checkpointer=InMemorySaver(),
        interrupt_after=["retrieve_evidence"],
    )
    config = _config("research-resume")

    await graph.ainvoke(_input(), config)
    paused = await graph.aget_state(config)
    assert paused.values["retrieval_calls"] == 1

    result = await graph.ainvoke(None, config)

    assert result["status"] is ResearchStatus.COMPLETED
    assert len(retrieval.search_calls) == 1


@pytest.mark.asyncio
async def test_evidence_is_deduplicated_and_bounded_in_checkpoint() -> None:
    question = _input()["question"]
    first = _hit("chunk-1", "A" * 800)
    duplicate = _hit("chunk-1", "A" * 800)
    second = _hit("chunk-2", "B" * 800)
    graph = build_research_graph(
        retrieval=ScriptedRetrieval({question: [first, duplicate, second]}),
        model=FakeResearchModel(),
        settings=_settings(
            research_max_chunk_chars=800,
            research_max_evidence_chars=1000,
        ),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(), _config("bounded-evidence"))

    assert result["status"] is ResearchStatus.COMPLETED
    assert [item.citation_id for item in result["evidence"]] == ["E1", "E2"]
    assert [item.chunk_id for item in result["evidence"]] == ["chunk-1", "chunk-2"]
    assert sum(len(item.content) for item in result["evidence"]) == 1000
