"""Contract tests for the MinerU and RAGFlow HTTP adapters."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
import pytest

from paperops.clients import MinerUError, MinerUTimeout
from paperops.clients.mineru import MinerUClient
from paperops.clients.ragflow import RAGFlowClient
from paperops.models import IngestRequest, ParseRequest, SearchRequest
from paperops.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        artifacts_dir=tmp_path / "artifacts",
        knowledge_dir=tmp_path / "knowledge",
        mineru_base_url="http://mineru.test",
        mineru_poll_interval_seconds=0.001,
        mineru_task_timeout_seconds=1,
        ragflow_base_url="http://ragflow.test",
        ragflow_api_key="test-key",
        ragflow_poll_interval_seconds=0.001,
        ragflow_index_timeout_seconds=1,
    )


def _zip_result(filename: str, content: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr(filename, content)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_mineru_client_submits_polls_downloads_and_reuses(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7\nfixture")
    markdown = "# Paper\n\n## Abstract\n\nParsed content."
    zip_content = _zip_result("paper/auto/paper.md", markdown)
    calls: list[tuple[str, str]] = []
    status_polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal status_polls
        assert request.url.host == "mineru.test"
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/tasks":
            body = request.read()
            assert b'name="files"' in body
            assert b"%PDF-1.7" in body
            assert b'name="parse_method"\r\n\r\nocr' in body
            return httpx.Response(
                202,
                json={
                    "task_id": "task-1",
                    "status_url": "http://untrusted.invalid/tasks/task-1",
                    "result_url": "http://untrusted.invalid/tasks/task-1/result",
                },
            )
        if request.url.path == "/tasks/task-1":
            status_polls += 1
            task_status = "pending" if status_polls == 1 else "completed"
            return httpx.Response(200, json={"status": task_status})
        if request.url.path == "/tasks/task-1/result":
            return httpx.Response(
                200,
                content=zip_content,
                headers={"content-type": "application/zip"},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as async_client:
        with httpx.Client(transport=transport) as sync_client:
            client = MinerUClient(
                settings,
                async_client=async_client,
                sync_client=sync_client,
            )
            request = ParseRequest(
                job_id="job-1",
                source_pdf=str(source),
                file_hash="a" * 64,
                attempt=2,
                idempotency_key="parse:job-1:2",
            )

            first = await client.parse(request)
            calls_after_first = list(calls)
            second = await client.parse(request)

    assert first.created is True
    assert second.created is False
    assert Path(first.markdown_path).read_text(encoding="utf-8") == markdown
    assert calls == calls_after_first


@pytest.mark.asyncio
async def test_mineru_client_rejects_zip_traversal(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7\nfixture")
    unsafe_zip = _zip_result("../escape.md", "unsafe")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                202,
                json={
                    "task_id": "unsafe",
                    "status_url": "/tasks/unsafe",
                    "result_url": "/tasks/unsafe/result",
                },
            )
        if request.url.path == "/tasks/unsafe":
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(200, content=unsafe_zip)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as async_client:
        with httpx.Client(transport=transport) as sync_client:
            client = MinerUClient(
                settings,
                async_client=async_client,
                sync_client=sync_client,
            )
            with pytest.raises(MinerUError, match="Unsafe path"):
                await client.parse(
                    ParseRequest(
                        job_id="unsafe-job",
                        source_pdf=str(source),
                        file_hash="b" * 64,
                        attempt=1,
                        idempotency_key="parse:unsafe-job:1",
                    )
                )

    assert not (settings.artifacts_dir / "unsafe-job" / "escape.md").exists()


@pytest.mark.asyncio
async def test_mineru_client_recovers_persisted_task_after_timeout(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"mineru_task_timeout_seconds": 0.003}
    )
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7\nfixture")
    zip_content = _zip_result("paper.md", "# Recovered\n\n## Result\n\nContent")
    can_complete = False
    submission_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submission_count
        assert request.url.host == "mineru.test"
        if request.method == "POST":
            submission_count += 1
            return httpx.Response(
                202,
                json={
                    "task_id": "recoverable",
                    "status_url": "/tasks/recoverable",
                    "result_url": "/tasks/recoverable/result",
                },
            )
        if request.url.path == "/tasks/recoverable":
            return httpx.Response(
                200,
                json={"status": "completed" if can_complete else "processing"},
            )
        return httpx.Response(200, content=zip_content)

    request = ParseRequest(
        job_id="recoverable-job",
        source_pdf=str(source),
        file_hash="d" * 64,
        attempt=1,
        idempotency_key="parse:recoverable-job:1",
    )
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as async_client:
        with httpx.Client(transport=transport) as sync_client:
            first_client = MinerUClient(
                settings,
                async_client=async_client,
                sync_client=sync_client,
            )
            with pytest.raises(MinerUTimeout):
                await first_client.parse(request)

    can_complete = True
    async with httpx.AsyncClient(transport=transport) as async_client:
        with httpx.Client(transport=transport) as sync_client:
            restarted_client = MinerUClient(
                settings,
                async_client=async_client,
                sync_client=sync_client,
            )
            recovered = await restarted_client.parse(request)

    assert submission_count == 1
    assert recovered.created is True
    assert Path(recovered.markdown_path).is_file()


@pytest.mark.asyncio
async def test_ragflow_client_uploads_indexes_retrieves_and_reuses(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    markdown_path = tmp_path / "parsed.md"
    markdown_path.write_text("# Indexed paper\n\nEvidence", encoding="utf-8")
    uploaded_document: dict[str, object] | None = None
    upload_calls = 0
    index_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal uploaded_document, upload_calls, index_calls
        assert request.headers["authorization"] == "Bearer test-key"
        path = request.url.path
        if request.method == "GET" and path.endswith("/documents"):
            query = request.url.params
            if query.get("name"):
                documents = [uploaded_document] if uploaded_document else []
            else:
                uploaded_document = {
                    **(uploaded_document or {}),
                    "run": "DONE",
                    "progress": 1.0,
                }
                documents = [uploaded_document]
            return httpx.Response(200, json={"code": 0, "data": {"docs": documents}})
        if request.method == "POST" and path.endswith("/documents"):
            upload_calls += 1
            body = request.read()
            assert b'name="file"' in body
            assert b"Evidence" in body
            uploaded_document = {
                "id": "doc-1",
                "name": "paper-cccccccccccccccccccc.md",
                "run": "0",
                "progress": 0.0,
            }
            return httpx.Response(200, json={"code": 0, "data": [uploaded_document]})
        if request.method == "POST" and path.endswith("/chunks"):
            index_calls += 1
            return httpx.Response(200, json={"code": 0, "data": True})
        if request.method == "POST" and path.endswith("/retrieval"):
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "chunks": [
                            {
                                "document_id": "doc-1",
                                "content": "retrieved evidence",
                                "similarity": 0.91,
                            }
                        ]
                    },
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as async_client:
        with httpx.Client(transport=transport) as sync_client:
            client = RAGFlowClient(
                settings,
                async_client=async_client,
                sync_client=sync_client,
            )
            ingest_request = IngestRequest(
                job_id="job-1",
                knowledge_base="dataset-1",
                file_hash="c" * 64,
                markdown_path=str(markdown_path),
                idempotency_key="ingest:dataset-1:hash",
            )
            first = await client.ingest(ingest_request)
            second = await client.ingest(ingest_request)
            hits = await client.search(
                SearchRequest(
                    knowledge_base="dataset-1",
                    query="What evidence?",
                    expected_document_id="doc-1",
                )
            )

    assert first.created is True
    assert second.created is False
    assert first.document_id == second.document_id == "doc-1"
    assert upload_calls == 1
    assert index_calls == 1
    assert hits[0].content == "retrieved evidence"
    assert hits[0].score == pytest.approx(0.91)
