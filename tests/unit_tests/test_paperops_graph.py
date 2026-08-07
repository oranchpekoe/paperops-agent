from pathlib import Path
from threading import get_ident
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from paperops import ApprovalAction, FailureCode, JobStatus
from paperops.clients import FakeKnowledgeBaseClient, FakeParserClient
from paperops.graph import build_graph
from paperops.settings import Settings

VALID_MARKDOWN = (
    "# UAV reinforcement learning\n\n"
    "## Abstract\n\n"
    + "This paper contains enough deterministic research content for validation. " * 5
    + "\n\n## Method\n\nThe method coordinates multiple agents safely.\n"
)
REVIEW_MARKDOWN = (
    "## Abstract\n\n"
    + "This artifact is long enough but has no top-level document title. " * 6
)


def _source_pdf(tmp_path: Path) -> Path:
    source = tmp_path / "uav_rl.pdf"
    source.write_bytes(b"%PDF-1.7\nPR2 deterministic fixture")
    return source


def _settings(tmp_path: Path, *, max_attempts: int = 2) -> Settings:
    return Settings(
        _env_file=None,
        artifacts_dir=tmp_path / "artifacts",
        knowledge_dir=tmp_path / "knowledge",
        max_parse_attempts=max_attempts,
        min_markdown_characters=80,
        min_section_count=1,
        min_retrieval_hits=1,
    )


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _input(source: Path) -> dict[str, Any]:
    return {
        "source_pdf": str(source),
        "target_knowledge_base": "uav-rl-papers",
    }


@pytest.mark.asyncio
async def test_happy_path_completes_and_writes_report(tmp_path: Path) -> None:
    source = _source_pdf(tmp_path)
    settings = _settings(tmp_path)
    parser = FakeParserClient(settings.artifacts_dir, [VALID_MARKDOWN])
    knowledge_base = FakeKnowledgeBaseClient()
    graph = build_graph(
        parser=parser,
        knowledge_base=knowledge_base,
        settings=settings,
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(source), _config("happy"))

    assert result["status"] is JobStatus.COMPLETED
    assert result["parse_attempts"] == 1
    assert result["retrieval_report"].passed is True
    assert Path(result["evaluation_report_path"]).is_file()
    assert parser.created_artifacts == 1
    assert knowledge_base.created_documents == 1
    assert VALID_MARKDOWN not in repr(result)


@pytest.mark.asyncio
async def test_filesystem_calls_run_outside_the_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_pdf(tmp_path)
    settings = _settings(tmp_path)
    parser = FakeParserClient(settings.artifacts_dir, [VALID_MARKDOWN])
    graph = build_graph(
        parser=parser,
        knowledge_base=FakeKnowledgeBaseClient(),
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    event_loop_thread = get_ident()

    with monkeypatch.context() as context:
        for method_name in (
            "is_file",
            "mkdir",
            "open",
            "read_text",
            "replace",
            "write_text",
        ):
            original = getattr(Path, method_name)

            def guarded(
                path: Path,
                *args: Any,
                _method: Any = original,
                _name: str = method_name,
                **kwargs: Any,
            ) -> Any:
                assert get_ident() != event_loop_thread, (
                    f"Path.{_name} ran on the event-loop thread"
                )
                return _method(path, *args, **kwargs)

            context.setattr(Path, method_name, guarded)

        result = await graph.ainvoke(_input(source), _config("non-blocking-io"))

    assert result["status"] is JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_invalid_source_stops_before_parser_side_effects(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    parser = FakeParserClient(settings.artifacts_dir)
    knowledge_base = FakeKnowledgeBaseClient()
    graph = build_graph(
        parser=parser,
        knowledge_base=knowledge_base,
        settings=settings,
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        {
            "source_pdf": str(tmp_path / "missing.pdf"),
            "target_knowledge_base": "uav-rl-papers",
        },
        _config("invalid-source"),
    )

    assert result["status"] is JobStatus.FAILED
    assert result["failure"].code is FailureCode.INVALID_SOURCE
    assert parser.calls == []
    assert knowledge_base.ingest_calls == []


@pytest.mark.asyncio
async def test_low_quality_parse_retries_then_completes(tmp_path: Path) -> None:
    source = _source_pdf(tmp_path)
    settings = _settings(tmp_path)
    parser = FakeParserClient(settings.artifacts_dir, ["too short", VALID_MARKDOWN])
    graph = build_graph(
        parser=parser,
        knowledge_base=FakeKnowledgeBaseClient(),
        settings=settings,
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(source), _config("retry"))

    assert result["status"] is JobStatus.COMPLETED
    assert result["parse_attempts"] == 2
    assert parser.created_artifacts == 2


@pytest.mark.asyncio
async def test_parser_exception_is_structured_and_retried(tmp_path: Path) -> None:
    source = _source_pdf(tmp_path)
    settings = _settings(tmp_path)
    parser = FakeParserClient(
        settings.artifacts_dir,
        [RuntimeError("temporary parser outage"), VALID_MARKDOWN],
    )
    graph = build_graph(
        parser=parser,
        knowledge_base=FakeKnowledgeBaseClient(),
        settings=settings,
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(source), _config("parser-error"))

    assert result["status"] is JobStatus.COMPLETED
    assert result["parse_attempts"] == 2
    assert result["errors"][0].code is FailureCode.PARSER_ERROR
    assert result["errors"][0].retryable is True


@pytest.mark.asyncio
async def test_retry_exhaustion_fails_closed(tmp_path: Path) -> None:
    source = _source_pdf(tmp_path)
    settings = _settings(tmp_path)
    parser = FakeParserClient(settings.artifacts_dir, ["short", "still short"])
    knowledge_base = FakeKnowledgeBaseClient()
    graph = build_graph(
        parser=parser,
        knowledge_base=knowledge_base,
        settings=settings,
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(source), _config("exhausted"))

    assert result["status"] is JobStatus.FAILED
    assert result["failure"].code is FailureCode.QUALITY_RETRIES_EXHAUSTED
    assert result["parse_attempts"] == 2
    assert knowledge_base.created_documents == 0


@pytest.mark.asyncio
async def test_review_interrupt_can_resume_with_approval(tmp_path: Path) -> None:
    source = _source_pdf(tmp_path)
    settings = _settings(tmp_path)
    parser = FakeParserClient(settings.artifacts_dir, [REVIEW_MARKDOWN])
    knowledge_base = FakeKnowledgeBaseClient()
    checkpointer = InMemorySaver()
    graph = build_graph(
        parser=parser,
        knowledge_base=knowledge_base,
        settings=settings,
        checkpointer=checkpointer,
    )
    config = _config("approval")

    await graph.ainvoke(_input(source), config)
    paused = await graph.aget_state(config)

    assert paused.values["status"] is JobStatus.WAITING_APPROVAL
    assert paused.tasks[0].interrupts

    result = await graph.ainvoke(
        Command(
            resume={
                "action": ApprovalAction.APPROVE.value,
                "note": "Artifact is acceptable for the MVP.",
            }
        ),
        config,
    )

    assert result["status"] is JobStatus.COMPLETED
    assert result["approval_decision"].action is ApprovalAction.APPROVE
    assert knowledge_base.created_documents == 1


@pytest.mark.asyncio
async def test_review_rejection_stops_before_ingestion(tmp_path: Path) -> None:
    source = _source_pdf(tmp_path)
    settings = _settings(tmp_path)
    knowledge_base = FakeKnowledgeBaseClient()
    graph = build_graph(
        parser=FakeParserClient(settings.artifacts_dir, [REVIEW_MARKDOWN]),
        knowledge_base=knowledge_base,
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    config = _config("rejection")
    await graph.ainvoke(_input(source), config)

    result = await graph.ainvoke(
        Command(resume={"action": ApprovalAction.REJECT.value, "note": "Broken"}),
        config,
    )

    assert result["status"] is JobStatus.FAILED
    assert result["failure"].code is FailureCode.APPROVAL_REJECTED
    assert knowledge_base.created_documents == 0


@pytest.mark.asyncio
async def test_checkpoint_resume_does_not_parse_again(tmp_path: Path) -> None:
    source = _source_pdf(tmp_path)
    settings = _settings(tmp_path)
    parser = FakeParserClient(settings.artifacts_dir, [VALID_MARKDOWN])
    checkpointer = InMemorySaver()
    graph = build_graph(
        parser=parser,
        knowledge_base=FakeKnowledgeBaseClient(),
        settings=settings,
        checkpointer=checkpointer,
        interrupt_after=["parse_document"],
    )
    config = _config("checkpoint")

    await graph.ainvoke(_input(source), config)
    paused = await graph.aget_state(config)

    assert paused.values["status"] is JobStatus.QUALITY_CHECK
    assert parser.created_artifacts == 1

    result = await graph.ainvoke(None, config)

    assert result["status"] is JobStatus.COMPLETED
    assert parser.created_artifacts == 1
    assert len(parser.calls) == 1


@pytest.mark.asyncio
async def test_duplicate_job_reuses_parser_and_ingestion_side_effects(
    tmp_path: Path,
) -> None:
    source = _source_pdf(tmp_path)
    settings = _settings(tmp_path)
    parser = FakeParserClient(settings.artifacts_dir, [VALID_MARKDOWN])
    knowledge_base = FakeKnowledgeBaseClient()
    graph = build_graph(
        parser=parser,
        knowledge_base=knowledge_base,
        settings=settings,
        checkpointer=InMemorySaver(),
    )

    first = await graph.ainvoke(_input(source), _config("duplicate-a"))
    second = await graph.ainvoke(_input(source), _config("duplicate-b"))

    assert first["job_id"] == second["job_id"]
    assert first["ragflow_document_id"] == second["ragflow_document_id"]
    assert parser.created_artifacts == 1
    assert knowledge_base.created_documents == 1
    assert len(knowledge_base.ingest_calls) == 2


@pytest.mark.asyncio
async def test_retrieval_miss_produces_structured_failure(tmp_path: Path) -> None:
    source = _source_pdf(tmp_path)
    settings = _settings(tmp_path)
    graph = build_graph(
        parser=FakeParserClient(settings.artifacts_dir, [VALID_MARKDOWN]),
        knowledge_base=FakeKnowledgeBaseClient(return_hits=False),
        settings=settings,
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(source), _config("retrieval-miss"))

    assert result["status"] is JobStatus.FAILED
    assert result["failure"].code is FailureCode.RETRIEVAL_FAILED
    assert result["retrieval_report"].passed is False


@pytest.mark.asyncio
async def test_ingestion_error_produces_structured_failure(tmp_path: Path) -> None:
    source = _source_pdf(tmp_path)
    settings = _settings(tmp_path)
    knowledge_base = FakeKnowledgeBaseClient(fail_ingest_times=1)
    graph = build_graph(
        parser=FakeParserClient(settings.artifacts_dir, [VALID_MARKDOWN]),
        knowledge_base=knowledge_base,
        settings=settings,
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(_input(source), _config("ingestion-error"))

    assert result["status"] is JobStatus.FAILED
    assert result["failure"].code is FailureCode.INGEST_ERROR
    assert knowledge_base.created_documents == 0
    assert knowledge_base.search_calls == []
