"""API and SQLite recovery tests for PR3 PaperOps jobs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from fastapi.testclient import TestClient

from paperops.api.app import create_app
from paperops.clients.fakes import FakeKnowledgeBaseClient, FakeParserClient
from paperops.research.fakes import FakeResearchModel
from paperops.retrieval.native import NativeRetrievalBackend
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


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        artifacts_dir=tmp_path / "artifacts",
        knowledge_dir=tmp_path / "knowledge",
        checkpoint_db=tmp_path / "paperops.db",
        min_markdown_characters=80,
        min_section_count=1,
    )


def _wait_for_job(
    client: TestClient,
    status_url: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(status_url)
        assert response.status_code == 200, response.text
        latest = response.json()
        if predicate(latest):
            return latest
        time.sleep(0.01)
    raise AssertionError(f"PaperOps job did not reach the expected state: {latest}")


def _submit(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/jobs",
        data={"target_knowledge_base": "dataset-1"},
        files={
            "file": (
                "uav_rl.pdf",
                b"%PDF-1.7\nPR3 deterministic fixture",
                "application/pdf",
            )
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def test_api_job_completes_and_survives_application_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(
        settings=settings,
        parser=FakeParserClient(settings.artifacts_dir, [VALID_MARKDOWN]),
        knowledge_base=FakeKnowledgeBaseClient(),
    )

    with TestClient(app) as client:
        accepted = _submit(client)
        completed = _wait_for_job(
            client,
            accepted["status_url"],
            lambda job: job["status"] == "completed" and not job["running"],
        )
        assert completed["indexed_document_id"].startswith("doc-")
        assert completed["indexed_chunk_count"] == 1
        assert completed["retrieval_report"]["passed"] is True
        health = client.get("/health").json()
        assert health["client_mode"] == "fake"
        assert health["retrieval_backend"] == "fake"

    assert settings.checkpoint_db.is_file()

    restarted_app = create_app(
        settings=settings,
        parser=FakeParserClient(settings.artifacts_dir),
        knowledge_base=FakeKnowledgeBaseClient(),
    )
    with TestClient(restarted_app) as restarted_client:
        recovered = restarted_client.get(accepted["status_url"])
        assert recovered.status_code == 200
        assert recovered.json()["status"] == "completed"
        assert recovered.json()["thread_id"] == accepted["thread_id"]


def test_api_runs_workflow_through_native_retrieval_backend(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(
        update={
            "native_index_db": tmp_path / "native-index.db",
            "native_chunk_size_chars": 240,
            "native_chunk_overlap_chars": 40,
        }
    )
    app = create_app(
        settings=settings,
        parser=FakeParserClient(settings.artifacts_dir, [VALID_MARKDOWN]),
        knowledge_base=NativeRetrievalBackend(settings),
    )

    with TestClient(app) as client:
        accepted = _submit(client)
        completed = _wait_for_job(
            client,
            accepted["status_url"],
            lambda job: job["status"] == "completed" and not job["running"],
        )
        health = client.get("/health").json()

    assert completed["indexed_document_id"].startswith("native-")
    assert completed["indexed_chunk_count"] >= 2
    assert completed["retrieval_report"]["backend"] == "native_fts5_bm25"
    assert completed["retrieval_report"]["strategy"] == ("document_scoped_index_probe")
    assert health["retrieval_backend"] == "native_fts5_bm25"


def test_api_exposes_and_resumes_human_approval(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(
        settings=settings,
        parser=FakeParserClient(settings.artifacts_dir, [REVIEW_MARKDOWN]),
        knowledge_base=FakeKnowledgeBaseClient(),
    )

    with TestClient(app) as client:
        accepted = _submit(client)
        waiting = _wait_for_job(
            client,
            accepted["status_url"],
            lambda job: job["approval_required"] and not job["running"],
        )
        assert waiting["status"] == "waiting_approval"
        assert "missing_title" in waiting["quality_decision"]["issues"]

    knowledge_base = FakeKnowledgeBaseClient()
    restarted_app = create_app(
        settings=settings,
        parser=FakeParserClient(settings.artifacts_dir),
        knowledge_base=knowledge_base,
    )
    with TestClient(restarted_app) as restarted_client:
        recovered = restarted_client.get(accepted["status_url"])
        assert recovered.status_code == 200
        assert recovered.json()["approval_required"] is True

        approval = restarted_client.post(
            f"/jobs/{accepted['thread_id']}/approval",
            json={"action": "approve", "note": "Reviewed against the PDF."},
        )
        assert approval.status_code == 202, approval.text
        completed = _wait_for_job(
            restarted_client,
            accepted["status_url"],
            lambda job: job["status"] == "completed" and not job["running"],
        )

    assert completed["approval_required"] is False
    assert knowledge_base.created_documents == 1


def test_api_rejects_non_pdf_payload(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(
        settings=settings,
        parser=FakeParserClient(settings.artifacts_dir),
        knowledge_base=FakeKnowledgeBaseClient(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            data={"target_knowledge_base": "dataset-1"},
            files={"file": ("notes.pdf", b"not-a-pdf", "application/pdf")},
        )

    assert response.status_code == 422
    assert "PDF signature" in response.json()["detail"]


def test_api_research_query_returns_checkpointed_citations(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    knowledge_base = FakeKnowledgeBaseClient()
    app = create_app(
        settings=settings,
        parser=FakeParserClient(settings.artifacts_dir, [VALID_MARKDOWN]),
        knowledge_base=knowledge_base,
    )

    with TestClient(app) as client:
        ingestion = _submit(client)
        _wait_for_job(
            client,
            ingestion["status_url"],
            lambda job: job["status"] == "completed" and not job["running"],
        )

        accepted_response = client.post(
            "/queries",
            json={
                "knowledge_base": "dataset-1",
                "question": "How are multiple agents coordinated?",
            },
        )
        assert accepted_response.status_code == 202, accepted_response.text
        accepted = accepted_response.json()
        completed = _wait_for_job(
            client,
            accepted["status_url"],
            lambda query: query["status"] == "completed" and not query["running"],
        )
        health = client.get("/health").json()

    assert completed["answer"]["citation_ids"] == ["E1"]
    assert "[E1]" in completed["answer"]["text"]
    assert completed["evidence"][0]["citation_id"] == "E1"
    assert completed["retrieval_calls"] == 1
    assert completed["model_calls"] == 2
    assert health["research_model"] == "fake-research-model"

    restarted_app = create_app(
        settings=settings,
        parser=FakeParserClient(settings.artifacts_dir),
        knowledge_base=FakeKnowledgeBaseClient(),
    )
    with TestClient(restarted_app) as restarted_client:
        recovered = restarted_client.get(accepted["status_url"])

    assert recovered.status_code == 200
    assert recovered.json()["status"] == "completed"
    assert recovered.json()["answer"]["citation_ids"] == ["E1"]


def test_api_retries_only_the_failed_research_stage(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    knowledge_base = FakeKnowledgeBaseClient()
    research_model = FakeResearchModel(
        assessments=[RuntimeError("temporary model outage")]
    )
    app = create_app(
        settings=settings,
        parser=FakeParserClient(settings.artifacts_dir, [VALID_MARKDOWN]),
        knowledge_base=knowledge_base,
        research_model=research_model,
    )

    with TestClient(app) as client:
        ingestion = _submit(client)
        _wait_for_job(
            client,
            ingestion["status_url"],
            lambda job: job["status"] == "completed" and not job["running"],
        )
        accepted = client.post(
            "/queries",
            json={
                "knowledge_base": "dataset-1",
                "question": "What does the indexed evidence report?",
            },
        ).json()
        failed = _wait_for_job(
            client,
            accepted["status_url"],
            lambda query: query["status"] == "failed" and not query["running"],
        )
        assert failed["failure"]["retryable"] is True
        assert failed["failure"]["stage"] == "assessing"

        resumed = client.post(f"{accepted['status_url']}/resume")
        assert resumed.status_code == 202, resumed.text
        completed = _wait_for_job(
            client,
            accepted["status_url"],
            lambda query: query["status"] == "completed" and not query["running"],
        )

    assert completed["failure"] is None
    assert completed["retrieval_calls"] == 1
    assert completed["model_calls"] == 3
    assert len(knowledge_base.search_calls) == 2
    assert len(research_model.assessment_calls) == 2
