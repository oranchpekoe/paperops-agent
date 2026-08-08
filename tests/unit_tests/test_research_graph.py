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
    ResearchStopReason,
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
async def test_query_can_limit_retrieval_to_one_indexed_document() -> None:
    scoped_input = {**_input(), "expected_document_id": "paper-expected"}
    retrieval = ScriptedRetrieval({scoped_input["question"]: [_hit()]})
    graph = build_research_graph(
        retrieval=retrieval,
        model=FakeResearchModel(),
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(scoped_input, _config("document-scoped"))

    assert result["status"] is ResearchStatus.COMPLETED
    assert result["expected_document_id"] == "paper-expected"
    assert retrieval.search_calls[0].expected_document_id == "paper-expected"


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
    retrieval = ScriptedRetrieval(
        {
            question: [_hit("chunk-1", "Partial evidence one.")],
            "rewrite one": [_hit("chunk-2", "Partial evidence two.")],
            "rewrite two": [_hit("chunk-3", "Partial evidence three.")],
        }
    )
    model = FakeResearchModel(
        assessments=[
            EvidenceAssessment(
                sufficient=False,
                confidence=0.9,
                rationale="A required detail is missing.",
                missing_aspects=["additional detail"],
                relevant_citation_ids=[],
            )
            for _ in range(3)
        ],
        rewrites=[
            QueryRewrite(query="rewrite one", reason="First missing aspect."),
            QueryRewrite(query="rewrite two", reason="Second missing aspect."),
        ],
    )
    graph = build_research_graph(
        retrieval=retrieval,
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(), _config("bounded-refusal"))

    assert result["status"] is ResearchStatus.INSUFFICIENT_EVIDENCE
    assert result["rewrite_count"] == 2
    assert result["retrieval_calls"] == 3
    assert result["attempted_queries"] == [question, "rewrite one", "rewrite two"]
    assert result["stop_reason"] is ResearchStopReason.BUDGET_EXHAUSTED
    assert "answer" not in result
    assert model.answer_calls == []


@pytest.mark.asyncio
async def test_stagnant_rewrite_stops_before_another_model_judgment() -> None:
    question = _input()["question"]
    duplicate = _hit("chunk-1", "Only the same partial evidence is available.")
    retrieval = ScriptedRetrieval({question: [duplicate], "focused query": [duplicate]})
    model = FakeResearchModel(
        assessments=[
            EvidenceAssessment(
                sufficient=False,
                confidence=0.9,
                rationale="The result is incomplete.",
                missing_aspects=["missing result"],
                relevant_citation_ids=[],
            )
        ],
        rewrites=[
            QueryRewrite(query="focused query", reason="Target the missing result.")
        ],
    )
    graph = build_research_graph(
        retrieval=retrieval,
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(), _config("stagnant-retrieval"))

    assert result["status"] is ResearchStatus.INSUFFICIENT_EVIDENCE
    assert result["stop_reason"] is ResearchStopReason.STAGNANT_RETRIEVAL
    assert result["retrieval_calls"] == 2
    assert result["rewrite_count"] == 1
    assert result["new_evidence_count"] == 0
    assert result["model_calls"] == 2
    assert len(model.assessment_calls) == 1


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
                relevant_citation_ids=["E1"],
            ),
            EvidenceAssessment(
                sufficient=True,
                confidence=0.9,
                rationale="Independent evidence is sufficient.",
                relevant_citation_ids=["E2"],
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
    assert [item.citation_id for item in model.answer_calls[0].evidence] == ["E2"]


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
async def test_single_valid_citation_missing_inline_marker_is_repaired() -> None:
    question = _input()["question"]
    model = FakeResearchModel(
        answers=[
            ResearchAnswer(
                text="The workflow coordinates specialist agents.",
                citation_ids=["E1"],
            )
        ]
    )
    graph = build_research_graph(
        retrieval=ScriptedRetrieval({question: [_hit()]}),
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(), _config("repair-inline-citation"))

    assert result["status"] is ResearchStatus.COMPLETED
    assert result["answer"].text == ("The workflow coordinates specialist agents [E1].")
    assert result["answer"].citation_ids == ["E1"]


@pytest.mark.asyncio
async def test_unknown_relevant_evidence_selection_fails_before_synthesis() -> None:
    question = _input()["question"]
    model = FakeResearchModel(
        assessments=[
            EvidenceAssessment(
                sufficient=True,
                confidence=0.9,
                rationale="The selected evidence appears sufficient.",
                relevant_citation_ids=["E99"],
            )
        ]
    )
    graph = build_research_graph(
        retrieval=ScriptedRetrieval({question: [_hit()]}),
        model=model,
        settings=_settings(),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(), _config("unknown-selection"))

    assert result["status"] is ResearchStatus.FAILED
    assert result["failure"].code is ResearchFailureCode.INVALID_MODEL_OUTPUT
    assert "unknown relevant citations" in result["failure"].message
    assert model.answer_calls == []


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
