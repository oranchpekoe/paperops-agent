"""Backend-agnostic evidence matching and retrieval benchmarking."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from paperops.clients.protocols import RetrievalBackend
from paperops.evaluation.models import (
    AggregateRetrievalMetrics,
    EvaluationQuery,
    EvidenceReference,
    QueryEvaluation,
    RankedEvidenceHit,
    RetrievalDataset,
    RetrievalEvaluationReport,
)
from paperops.models import IngestRequest, SearchHit, SearchRequest
from paperops.retrieval.native import NativeRetrievalBackend
from paperops.settings import Settings

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_COLLECTION_RE = re.compile(r"[^0-9A-Za-z_-]+")


def load_retrieval_dataset(path: Path) -> RetrievalDataset:
    """Load and validate one UTF-8 JSON retrieval dataset."""
    return RetrievalDataset.model_validate_json(path.read_text(encoding="utf-8"))


def write_evaluation_report(
    report: RetrievalEvaluationReport,
    path: Path,
) -> None:
    """Persist a stable, human-readable JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _tokens(text: str) -> Counter[str]:
    return Counter(token.lower() for token in _TOKEN_RE.findall(text))


def evidence_token_coverage(evidence: str, chunk: str) -> float:
    """Measure how much of an annotated evidence paragraph a chunk retains."""
    evidence_tokens = _tokens(evidence)
    if not evidence_tokens:
        return 0.0
    chunk_tokens = _tokens(chunk)
    overlap = sum(
        min(count, chunk_tokens.get(token, 0))
        for token, count in evidence_tokens.items()
    )
    return overlap / sum(evidence_tokens.values())


def matches_evidence(
    *,
    content: str,
    logical_document_id: str | None,
    evidence: EvidenceReference,
    threshold: float,
) -> bool:
    """Match one retrieved passage to one labelled evidence unit."""
    if logical_document_id != evidence.document_id:
        return False
    normalized_evidence = " ".join(evidence.text.lower().split())
    normalized_chunk = " ".join(content.lower().split())
    if normalized_evidence in normalized_chunk:
        return True
    return evidence_token_coverage(evidence.text, content) >= threshold


def _discounted_cumulative_gain(grades: list[int], limit: int) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades[:limit], start=1)
    )


def _evaluate_query(
    query: EvaluationQuery,
    hits: list[SearchHit],
    document_ids: dict[str, str],
    top_k: tuple[int, ...],
    threshold: float,
    latency_ms: float,
) -> QueryEvaluation:
    ranked_hits: list[RankedEvidenceHit] = []
    matches_by_rank: list[set[str]] = []
    evidence_by_id = {item.evidence_id: item for item in query.evidence}

    for rank, hit in enumerate(hits, start=1):
        logical_document_id = document_ids.get(hit.document_id)
        matched = {
            evidence.evidence_id
            for evidence in query.evidence
            if matches_evidence(
                content=hit.content,
                logical_document_id=logical_document_id,
                evidence=evidence,
                threshold=threshold,
            )
        }
        matches_by_rank.append(matched)
        ranked_hits.append(
            RankedEvidenceHit(
                rank=rank,
                document_id=logical_document_id or hit.document_id,
                chunk_id=hit.chunk_id,
                score=hit.score,
                matched_evidence_ids=sorted(matched),
            )
        )

    first_relevant_rank = next(
        (rank for rank, matched in enumerate(matches_by_rank, start=1) if matched),
        None,
    )
    reciprocal_rank = (
        1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
    )

    recall_at_k: dict[str, float] = {}
    ndcg_at_k: dict[str, float] = {}
    ideal_grades = sorted(
        (evidence.relevance for evidence in query.evidence),
        reverse=True,
    )
    for limit in top_k:
        covered: set[str] = set()
        retrieved_grades: list[int] = []
        for matched in matches_by_rank[:limit]:
            newly_covered = matched - covered
            retrieved_grades.append(
                max(
                    (evidence_by_id[item].relevance for item in newly_covered),
                    default=0,
                )
            )
            covered.update(newly_covered)
        recall_at_k[str(limit)] = len(covered) / len(evidence_by_id)
        actual_dcg = _discounted_cumulative_gain(retrieved_grades, limit)
        ideal_dcg = _discounted_cumulative_gain(ideal_grades, limit)
        ndcg_at_k[str(limit)] = actual_dcg / ideal_dcg if ideal_dcg else 0.0

    return QueryEvaluation(
        query_id=query.query_id,
        query=query.text,
        latency_ms=latency_ms,
        reciprocal_rank=reciprocal_rank,
        recall_at_k=recall_at_k,
        ndcg_at_k=ndcg_at_k,
        hits=ranked_hits,
    )


def percentile(values: list[float], quantile: float) -> float:
    """Interpolate one quantile from a non-empty numeric sample."""
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _aggregate(
    queries: list[QueryEvaluation],
    top_k: tuple[int, ...],
) -> AggregateRetrievalMetrics:
    latencies = [query.latency_ms for query in queries]
    return AggregateRetrievalMetrics(
        recall_at_k={
            str(limit): statistics.fmean(
                query.recall_at_k[str(limit)] for query in queries
            )
            for limit in top_k
        },
        mrr=statistics.fmean(query.reciprocal_rank for query in queries),
        ndcg_at_k={
            str(limit): statistics.fmean(
                query.ndcg_at_k[str(limit)] for query in queries
            )
            for limit in top_k
        },
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
    )


def dataset_sha256(dataset: RetrievalDataset) -> str:
    """Hash the canonical validated dataset representation."""
    canonical = json.dumps(
        dataset.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationCorpusIndex:
    """Identity and document-id mapping shared by offline evaluators."""

    collection_id: str
    dataset_sha256: str
    backend_to_logical_document_id: dict[str, str]
    logical_to_backend_document_id: dict[str, str]
    indexing_latency_ms: float


def _collection_id(
    dataset: RetrievalDataset,
    dataset_sha256: str,
    settings: Settings,
    index_profile: str,
) -> str:
    safe_name = _COLLECTION_RE.sub("-", dataset.name).strip("-")[:40]
    index_fingerprint = hashlib.sha256(index_profile.encode()).hexdigest()[:12]
    raw = (
        f"eval-{safe_name or 'dataset'}-{dataset_sha256[:16]}-"
        f"{settings.native_chunk_size_chars}-"
        f"{settings.native_chunk_overlap_chars}-{index_fingerprint}"
    )
    return raw[:128]


async def evaluate_retrieval_backend(
    dataset: RetrievalDataset,
    *,
    backend: RetrievalBackend,
    settings: Settings,
    work_dir: Path,
    index_profile: str,
    top_k: tuple[int, ...] = (1, 3, 5, 10),
    evidence_token_coverage_threshold: float = 0.6,
) -> RetrievalEvaluationReport:
    """Index a dataset and evaluate one backend against identical labels."""
    limits = tuple(sorted(set(top_k)))
    if not limits or limits[0] < 1:
        raise ValueError("top_k must contain positive integers")
    if limits[-1] > settings.native_search_top_k:
        raise ValueError(
            "largest top_k exceeds PAPEROPS_NATIVE_SEARCH_TOP_K; increase the "
            "backend limit before evaluating"
        )
    if not 0.0 <= evidence_token_coverage_threshold <= 1.0:
        raise ValueError("evidence_token_coverage_threshold must be between 0 and 1")

    corpus = await index_evaluation_corpus(
        dataset,
        backend=backend,
        settings=settings,
        work_dir=work_dir,
        index_profile=index_profile,
    )

    query_reports: list[QueryEvaluation] = []
    for query in dataset.queries:
        if not query.answerable:
            continue
        query_started = perf_counter()
        hits = await backend.search(
            SearchRequest(
                knowledge_base=corpus.collection_id,
                query=query.text,
                expected_document_id=(
                    corpus.logical_to_backend_document_id[query.document_id]
                    if query.document_id is not None
                    else None
                ),
                top_k=limits[-1],
            )
        )
        latency_ms = (perf_counter() - query_started) * 1000
        query_reports.append(
            _evaluate_query(
                query,
                hits,
                corpus.backend_to_logical_document_id,
                limits,
                evidence_token_coverage_threshold,
                latency_ms,
            )
        )

    if not query_reports:
        raise ValueError("retrieval evaluation requires answerable queries")

    return RetrievalEvaluationReport(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_sha256=corpus.dataset_sha256,
        dataset_kind=dataset.kind,
        split=dataset.split,
        backend=backend.name,
        index_profile=index_profile,
        document_count=len(dataset.documents),
        query_count=len(query_reports),
        top_k=list(limits),
        chunk_size_chars=settings.native_chunk_size_chars,
        chunk_overlap_chars=settings.native_chunk_overlap_chars,
        evidence_token_coverage_threshold=evidence_token_coverage_threshold,
        indexing_latency_ms=corpus.indexing_latency_ms,
        aggregate=_aggregate(query_reports, limits),
        queries=query_reports,
    )


async def index_evaluation_corpus(
    dataset: RetrievalDataset,
    *,
    backend: RetrievalBackend,
    settings: Settings,
    work_dir: Path,
    index_profile: str,
) -> EvaluationCorpusIndex:
    """Index a labelled corpus once for retrieval and Agent evaluations."""
    fingerprint = dataset_sha256(dataset)
    collection = _collection_id(dataset, fingerprint, settings, index_profile)
    corpus_dir = work_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    backend_to_logical_document_id: dict[str, str] = {}
    logical_to_backend_document_id: dict[str, str] = {}

    indexing_started = perf_counter()
    for document in dataset.documents:
        markdown = document.to_markdown()
        markdown_path = corpus_dir / f"{document.document_id}.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        file_hash = hashlib.sha256(markdown.encode()).hexdigest()
        result = await backend.ingest(
            IngestRequest(
                job_id=f"eval-{document.document_id}",
                knowledge_base=collection,
                file_hash=file_hash,
                markdown_path=str(markdown_path),
                idempotency_key=(
                    f"evaluation:{dataset.name}:{dataset.version}:"
                    f"{dataset.split}:{fingerprint}:"
                    f"{settings.native_chunk_size_chars}:"
                    f"{settings.native_chunk_overlap_chars}:"
                    f"{index_profile}:"
                    f"{document.document_id}:{file_hash}"
                ),
            )
        )
        backend_to_logical_document_id[result.document_id] = document.document_id
        logical_to_backend_document_id[document.document_id] = result.document_id
    indexing_latency_ms = (perf_counter() - indexing_started) * 1000
    return EvaluationCorpusIndex(
        collection_id=collection,
        dataset_sha256=fingerprint,
        backend_to_logical_document_id=backend_to_logical_document_id,
        logical_to_backend_document_id=logical_to_backend_document_id,
        indexing_latency_ms=indexing_latency_ms,
    )


async def evaluate_native_retrieval(
    dataset: RetrievalDataset,
    *,
    settings: Settings,
    work_dir: Path,
    top_k: tuple[int, ...] = (1, 3, 5, 10),
    evidence_token_coverage_threshold: float = 0.6,
) -> RetrievalEvaluationReport:
    """Evaluate the dependency-free SQLite FTS5/BM25 baseline."""
    return await evaluate_retrieval_backend(
        dataset,
        backend=NativeRetrievalBackend(settings),
        settings=settings,
        work_dir=work_dir,
        index_profile="native-fts5-bm25",
        top_k=top_k,
        evidence_token_coverage_threshold=evidence_token_coverage_threshold,
    )


def report_summary(report: RetrievalEvaluationReport) -> str:
    """Format compact metrics for CLI output without hiding dataset provenance."""
    return json.dumps(
        {
            "dataset": report.dataset_name,
            "version": report.dataset_version,
            "dataset_sha256": report.dataset_sha256,
            "kind": report.dataset_kind,
            "backend": report.backend,
            "index_profile": report.index_profile,
            "documents": report.document_count,
            "queries": report.query_count,
            "chunk_size_chars": report.chunk_size_chars,
            "chunk_overlap_chars": report.chunk_overlap_chars,
            "recall_at_k": report.aggregate.recall_at_k,
            "mrr": report.aggregate.mrr,
            "ndcg_at_k": report.aggregate.ndcg_at_k,
            "latency_p50_ms": report.aggregate.latency_p50_ms,
            "latency_p95_ms": report.aggregate.latency_p95_ms,
        },
        indent=2,
    )
