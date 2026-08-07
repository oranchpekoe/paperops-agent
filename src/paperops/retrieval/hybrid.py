"""Rank fusion and bounded cross-encoder reranking for local retrieval."""

from __future__ import annotations

import asyncio
import math

from paperops.clients.protocols import RetrievalBackend
from paperops.models import IngestRequest, IngestResult, SearchHit, SearchRequest
from paperops.retrieval.providers import Reranker


class HybridRetrievalBackend:
    """Fuse sparse and dense candidates with reciprocal rank fusion."""

    def __init__(
        self,
        sparse: RetrievalBackend,
        dense: RetrievalBackend,
        *,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        """Configure two first-stage retrievers and rank-only fusion."""
        if candidate_k < 1 or candidate_k > 100:
            raise ValueError("candidate_k must be between 1 and 100")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        self.sparse = sparse
        self.dense = dense
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.name = f"hybrid_rrf:{sparse.name}+{dense.name}"

    async def ingest(self, request: IngestRequest) -> IngestResult:
        """Populate both indexes with the same deterministic chunks."""
        sparse_result = await self.sparse.ingest(request)
        dense_result = await self.dense.ingest(request)
        if (
            sparse_result.document_id != dense_result.document_id
            or sparse_result.chunk_count != dense_result.chunk_count
        ):
            raise ValueError("sparse and dense indexes produced inconsistent chunks")
        return IngestResult(
            document_id=sparse_result.document_id,
            idempotency_key=request.idempotency_key,
            created=sparse_result.created or dense_result.created,
            chunk_count=sparse_result.chunk_count,
        )

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        """Retrieve candidates concurrently and combine them by rank."""
        candidate_limit = max(request.top_k, self.candidate_k)
        candidate_request = request.model_copy(update={"top_k": candidate_limit})
        sparse_hits, dense_hits = await asyncio.gather(
            self.sparse.search(candidate_request),
            self.dense.search(candidate_request),
        )
        fused_scores: dict[str, float] = {}
        hits_by_key: dict[str, SearchHit] = {}
        for ranking in (sparse_hits, dense_hits):
            for rank, hit in enumerate(ranking, start=1):
                key = hit.chunk_id or f"{hit.document_id}:{hit.content}"
                fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )
                hits_by_key.setdefault(key, hit)
        ordered = sorted(
            fused_scores,
            key=lambda key: (-fused_scores[key], key),
        )
        best_score = fused_scores[ordered[0]] if ordered else 0.0
        return [
            hits_by_key[key].model_copy(
                update={"score": fused_scores[key] / best_score if best_score else 0.0}
            )
            for key in ordered[: request.top_k]
        ]


class RerankedRetrievalBackend:
    """Apply a cross encoder only to a bounded first-stage candidate set."""

    def __init__(
        self,
        base: RetrievalBackend,
        reranker: Reranker,
        *,
        candidate_k: int = 20,
    ) -> None:
        """Configure the source ranking and maximum reranking cost."""
        if candidate_k < 1 or candidate_k > 100:
            raise ValueError("candidate_k must be between 1 and 100")
        self.base = base
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.name = f"reranked:{base.name}+{reranker.name}"

    async def ingest(self, request: IngestRequest) -> IngestResult:
        """Delegate indexing to the first-stage backend."""
        return await self.base.ingest(request)

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        """Rerank a bounded candidate set and return calibrated scores."""
        candidate_limit = max(request.top_k, self.candidate_k)
        candidates = await self.base.search(
            request.model_copy(update={"top_k": candidate_limit})
        )
        if not candidates:
            return []
        raw_scores = await asyncio.to_thread(
            self.reranker.rerank,
            request.query,
            [candidate.content for candidate in candidates],
        )
        if len(raw_scores) != len(candidates):
            raise ValueError("reranker returned an unexpected score count")
        ranking = sorted(
            zip(candidates, raw_scores, strict=True),
            key=lambda item: (
                -item[1],
                item[0].chunk_id or "",
                item[0].document_id,
            ),
        )
        return [
            hit.model_copy(update={"score": _sigmoid(score)})
            for hit, score in ranking[: request.top_k]
        ]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
