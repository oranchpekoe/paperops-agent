"""Tests for heading-aware chunking and the native SQLite retrieval baseline."""

from pathlib import Path

import pytest

from paperops.models import IngestRequest, SearchRequest
from paperops.retrieval.chunking import build_index_probe, chunk_markdown
from paperops.retrieval.native import NativeRetrievalBackend
from paperops.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        native_index_db=tmp_path / "native-index.db",
        native_chunk_size_chars=240,
        native_chunk_overlap_chars=40,
        native_search_top_k=8,
    )


def test_heading_aware_chunker_never_mixes_sections() -> None:
    markdown = (
        "# Cooperative UAV Encirclement\n\n"
        "## Reward Design\n\n"
        + "Distance reduction and collision penalty shape the reward. " * 8
        + "\n\n## Training Setup\n\n"
        + "The policy is trained with centralized learning. " * 8
    )

    chunks = chunk_markdown(markdown, max_chars=220, overlap_chars=30)

    assert len(chunks) >= 4
    assert all(chunk.content for chunk in chunks)
    assert all(
        not (
            "collision penalty" in chunk.content
            and "centralized learning" in chunk.content
        )
        for chunk in chunks
    )
    assert any(chunk.heading_path[-1] == "Reward Design" for chunk in chunks)
    assert any(chunk.heading_path[-1] == "Training Setup" for chunk in chunks)


def test_probe_prefers_document_headings() -> None:
    markdown = "# Main Title\n\n## Reward Function\n\nBody text."

    assert build_index_probe(markdown, "fallback") == "Main Title Reward Function"


@pytest.mark.asyncio
async def test_native_backend_indexes_searches_and_reuses_document(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = NativeRetrievalBackend(settings)
    markdown_path = tmp_path / "paper.md"
    markdown_path.write_text(
        "# Cooperative UAV Encirclement\n\n"
        "## Reward Design\n\n"
        "The reward function combines distance reduction with a collision penalty.\n\n"
        "## Training Setup\n\n"
        "Centralized training is used for the multi-agent policy.",
        encoding="utf-8",
    )
    request = IngestRequest(
        job_id="job-1",
        knowledge_base="uav-papers",
        file_hash="a" * 64,
        markdown_path=str(markdown_path),
        idempotency_key="ingest:uav-papers:hash-a",
    )

    first = await backend.ingest(request)
    second = await backend.ingest(request)
    hits = await backend.search(
        SearchRequest(
            knowledge_base="uav-papers",
            query="collision penalty reward function",
            expected_document_id=first.document_id,
            top_k=5,
        )
    )

    assert first.created is True
    assert second.created is False
    assert first.document_id == second.document_id
    assert first.chunk_count == second.chunk_count == 2
    assert hits
    assert hits[0].document_id == first.document_id
    assert hits[0].chunk_id
    assert hits[0].heading_path[-1] == "Reward Design"
    assert "collision penalty" in hits[0].content


@pytest.mark.asyncio
async def test_native_backend_filters_collections_and_sanitizes_query(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = NativeRetrievalBackend(settings)
    for collection, suffix, content in [
        ("uav-papers", "a", "# 无人机协同围捕\n\n奖励函数包含距离惩罚。"),
        ("vision-papers", "b", "# 图像去雾方法\n\n网络预测大气光。"),
    ]:
        path = tmp_path / f"{suffix}.md"
        path.write_text(content, encoding="utf-8")
        await backend.ingest(
            IngestRequest(
                job_id=f"job-{suffix}",
                knowledge_base=collection,
                file_hash=suffix * 64,
                markdown_path=str(path),
                idempotency_key=f"ingest:{collection}:{suffix}",
            )
        )

    hits = await backend.search(
        SearchRequest(
            knowledge_base="uav-papers",
            query="无人机协同围捕奖励函数",
        )
    )
    operator_like_query = await backend.search(
        SearchRequest(
            knowledge_base="uav-papers",
            query='" OR * NEAR(',
        )
    )

    assert hits
    assert all("无人机" in hit.content for hit in hits)
    assert operator_like_query == []
