"""Tests for sqlite-vec retrieval, RRF, and bounded reranking."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest

from paperops.api.app import _build_retrieval_backend
from paperops.models import IngestRequest, IngestResult, SearchHit, SearchRequest
from paperops.retrieval.dense import DenseRetrievalBackend
from paperops.retrieval.hybrid import HybridRetrievalBackend, RerankedRetrievalBackend
from paperops.retrieval.native import NativeRetrievalBackend
from paperops.retrieval.providers import FastEmbedProvider
from paperops.settings import Settings


class HashEmbeddingProvider:
    """Small deterministic test double that preserves token identity."""

    name = "test-hash-embedding-v1"
    dimension = 64

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        vector = [0.0] * cls.dimension
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            slot = hashlib.sha256(token.encode()).digest()[0] % cls.dimension
            vector[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self._embed(document) for document in documents]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)


class ReverseReranker:
    name = "test-reverse-reranker"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        del query
        return [float(index) for index, _ in enumerate(documents)]


class StaticBackend:
    name = "static"

    async def ingest(self, request: IngestRequest) -> IngestResult:
        return IngestResult(
            document_id="doc",
            idempotency_key=request.idempotency_key,
            created=True,
            chunk_count=1,
        )

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        return [
            SearchHit(
                document_id="doc",
                chunk_id=f"chunk-{index}",
                content=f"candidate {index}",
                score=1.0 / index,
            )
            for index in range(1, min(request.top_k, 3) + 1)
        ]


class RankedBackend(StaticBackend):
    def __init__(self, chunk_ids: list[str]) -> None:
        self.chunk_ids = chunk_ids

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        return [
            SearchHit(
                document_id="doc",
                chunk_id=chunk_id,
                content=chunk_id,
                score=1.0 / rank,
            )
            for rank, chunk_id in enumerate(
                self.chunk_ids[: request.top_k],
                start=1,
            )
        ]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        native_index_db=tmp_path / "hybrid-index.db",
        native_chunk_size_chars=300,
        native_chunk_overlap_chars=40,
        native_search_top_k=10,
    )


def _request(path: Path, suffix: str) -> IngestRequest:
    return IngestRequest(
        job_id=f"job-{suffix}",
        knowledge_base="papers",
        file_hash=suffix * 64,
        markdown_path=str(path),
        idempotency_key=f"ingest:papers:{suffix}",
    )


def test_application_builds_explicit_hybrid_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "paperops.api.app.FastEmbedProvider",
        lambda *args, **kwargs: HashEmbeddingProvider(),
    )
    settings = _settings(tmp_path).model_copy(
        update={"retrieval_backend": "hybrid", "retrieval_candidate_k": 20}
    )

    backend = _build_retrieval_backend(settings)

    assert isinstance(backend, HybridRetrievalBackend)
    assert backend.candidate_k == 20
    assert backend.sparse.settings.native_search_top_k == 20


def test_default_native_profile_does_not_load_optional_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected model load: {args}, {kwargs}")

    monkeypatch.setattr("paperops.api.app.FastEmbedProvider", fail_if_loaded)

    backend = _build_retrieval_backend(_settings(tmp_path))

    assert isinstance(backend, NativeRetrievalBackend)


def test_optional_model_provider_has_actionable_install_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "paperops.retrieval.providers.importlib.import_module",
        missing_module,
    )

    with pytest.raises(RuntimeError, match="uv sync --extra retrieval-models"):
        FastEmbedProvider("missing-model", cache_dir=tmp_path)


@pytest.mark.asyncio
async def test_dense_backend_indexes_filters_and_reuses_chunks(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    backend = DenseRetrievalBackend(settings, HashEmbeddingProvider())
    reward_path = tmp_path / "reward.md"
    reward_path.write_text(
        "# UAV Control\n\n## Reward\n\nCollision penalty protects every aircraft.",
        encoding="utf-8",
    )
    vision_path = tmp_path / "vision.md"
    vision_path.write_text(
        "# Image Restoration\n\n## Model\n\nAtmospheric light explains haze.",
        encoding="utf-8",
    )

    first = await backend.ingest(_request(reward_path, "a"))
    reused = await backend.ingest(_request(reward_path, "a"))
    await backend.ingest(_request(vision_path, "b"))
    hits = await backend.search(
        SearchRequest(
            knowledge_base="papers",
            query="aircraft collision penalty",
            top_k=2,
        )
    )

    assert first.created is True
    assert reused.created is False
    assert first.document_id == reused.document_id
    assert hits[0].document_id == first.document_id
    assert "Collision penalty" in hits[0].content
    assert hits[0].chunk_id


@pytest.mark.asyncio
async def test_hybrid_backend_shares_ids_and_fuses_rankings(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    sparse = NativeRetrievalBackend(settings)
    dense = DenseRetrievalBackend(settings, HashEmbeddingProvider())
    backend = HybridRetrievalBackend(sparse, dense, candidate_k=5, rrf_k=60)
    source = tmp_path / "paper.md"
    source.write_text(
        "# Cooperative Flight\n\n## Safety\n\n"
        "A collision penalty preserves safe separation between aircraft.",
        encoding="utf-8",
    )

    result = await backend.ingest(_request(source, "c"))
    hits = await backend.search(
        SearchRequest(
            knowledge_base="papers",
            query="aircraft collision safety",
            top_k=3,
        )
    )

    assert result.chunk_count == 1
    assert hits
    assert hits[0].document_id == result.document_id
    assert hits[0].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_reranker_only_reorders_bounded_candidates() -> None:
    backend = RerankedRetrievalBackend(
        StaticBackend(),
        ReverseReranker(),
        candidate_k=3,
    )

    hits = await backend.search(
        SearchRequest(knowledge_base="papers", query="query", top_k=2)
    )

    assert [hit.chunk_id for hit in hits] == ["chunk-3", "chunk-2"]
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)


@pytest.mark.asyncio
async def test_rrf_breaks_equal_scores_by_stable_chunk_id() -> None:
    backend = HybridRetrievalBackend(
        RankedBackend(["chunk-b", "chunk-a"]),
        RankedBackend(["chunk-a", "chunk-b"]),
        candidate_k=2,
    )

    hits = await backend.search(
        SearchRequest(knowledge_base="papers", query="query", top_k=2)
    )

    assert [hit.chunk_id for hit in hits] == ["chunk-a", "chunk-b"]
