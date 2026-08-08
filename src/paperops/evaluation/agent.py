"""Evaluate one-shot and bounded research graphs on identical labelled queries."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from time import perf_counter
from typing import cast

from langgraph.graph.state import CompiledStateGraph

from paperops.clients.protocols import RetrievalBackend
from paperops.evaluation.models import (
    AgentEvaluationReport,
    AgentMetricDelta,
    AgentQueryComparison,
    AgentRunEvaluation,
    AgentVariantMetrics,
    EvaluationQuery,
    RetrievalDataset,
)
from paperops.evaluation.retrieval import (
    EvaluationCorpusIndex,
    index_evaluation_corpus,
    matches_evidence,
    percentile,
)
from paperops.research.graph import build_research_graph
from paperops.research.models import (
    EvidenceCitation,
    ModelCallUsage,
    ResearchStatus,
)
from paperops.research.protocols import ResearchModel
from paperops.research.state import ResearchQueryState
from paperops.settings import Settings


def _matched_evidence_ids(
    query: EvaluationQuery,
    citations: list[EvidenceCitation],
    corpus: EvaluationCorpusIndex,
    threshold: float,
) -> set[str]:
    matched: set[str] = set()
    for citation in citations:
        logical_document_id = corpus.backend_to_logical_document_id.get(
            citation.document_id
        )
        matched.update(
            evidence.evidence_id
            for evidence in query.evidence
            if matches_evidence(
                content=citation.content,
                logical_document_id=logical_document_id,
                evidence=evidence,
                threshold=threshold,
            )
        )
    return matched


def _token_totals(
    usage: list[ModelCallUsage],
    model_calls: int,
) -> tuple[int | None, int | None, int | None, float]:
    latency_ms = sum(item.latency_ms for item in usage)
    if model_calls == 0:
        return 0, 0, 0, latency_ms
    if len(usage) != model_calls:
        return None, None, None, latency_ms
    fields = (
        [item.prompt_tokens for item in usage],
        [item.completion_tokens for item in usage],
        [item.total_tokens for item in usage],
    )
    if any(any(value is None for value in values) for values in fields):
        return None, None, None, latency_ms
    return (
        sum(value for value in fields[0] if value is not None),
        sum(value for value in fields[1] if value is not None),
        sum(value for value in fields[2] if value is not None),
        latency_ms,
    )


async def _run_query(
    *,
    graph: CompiledStateGraph[
        ResearchQueryState,
        None,
        ResearchQueryState,
        ResearchQueryState,
    ],
    model: ResearchModel,
    query: EvaluationQuery,
    corpus: EvaluationCorpusIndex,
    threshold: float,
) -> AgentRunEvaluation:
    model.drain_usage()
    started = perf_counter()
    result = cast(
        ResearchQueryState,
        await graph.ainvoke(
            {
                "knowledge_base": corpus.collection_id,
                "question": query.text,
            }
        ),
    )
    latency_ms = (perf_counter() - started) * 1000
    usage = model.drain_usage()

    status = result.get("status", ResearchStatus.FAILED)
    evidence = result.get("evidence", [])
    matched = _matched_evidence_ids(query, evidence, corpus, threshold)
    evidence_recall = len(matched) / len(query.evidence) if query.answerable else None

    answer = result.get("answer")
    citation_precision: float | None = None
    citation_recall: float | None = 0.0 if query.answerable else None
    if query.answerable and answer is not None:
        evidence_by_citation = {item.citation_id: item for item in evidence}
        cited_chunks = [
            evidence_by_citation[citation_id]
            for citation_id in answer.citation_ids
            if citation_id in evidence_by_citation
        ]
        cited_matches = _matched_evidence_ids(query, cited_chunks, corpus, threshold)
        citation_precision = (
            sum(
                bool(_matched_evidence_ids(query, [citation], corpus, threshold))
                for citation in cited_chunks
            )
            / len(cited_chunks)
            if cited_chunks
            else 0.0
        )
        citation_recall = len(cited_matches) / len(query.evidence)

    outcome_correct = (
        status == ResearchStatus.COMPLETED
        if query.answerable
        else status == ResearchStatus.INSUFFICIENT_EVIDENCE
    )
    model_calls = result.get("model_calls", 0)
    prompt_tokens, completion_tokens, total_tokens, model_latency_ms = _token_totals(
        usage,
        model_calls,
    )
    failure = result.get("failure")
    return AgentRunEvaluation(
        status=status.value if isinstance(status, ResearchStatus) else str(status),
        outcome_correct=outcome_correct,
        latency_ms=latency_ms,
        evidence_recall=evidence_recall,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        matched_evidence_ids=sorted(matched),
        retrieval_calls=result.get("retrieval_calls", 0),
        rewrite_count=result.get("rewrite_count", 0),
        model_calls=model_calls,
        attempted_queries=result.get("attempted_queries", []),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model_latency_ms=model_latency_ms,
        failure_code=failure.code.value if failure is not None else None,
    )


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _sum_tokens(runs: list[AgentRunEvaluation], field: str) -> int | None:
    values = [getattr(run, field) for run in runs]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _aggregate(
    comparisons: list[AgentQueryComparison],
    variant: str,
) -> AgentVariantMetrics:
    runs = [getattr(comparison, variant) for comparison in comparisons]
    answerable = [
        getattr(comparison, variant)
        for comparison in comparisons
        if comparison.answerable
    ]
    unanswerable = [
        getattr(comparison, variant)
        for comparison in comparisons
        if not comparison.answerable
    ]
    latencies = [run.latency_ms for run in runs]
    return AgentVariantMetrics(
        outcome_accuracy=statistics.fmean(run.outcome_correct for run in runs),
        answerable_completion_rate=(
            statistics.fmean(
                run.status == ResearchStatus.COMPLETED.value for run in answerable
            )
            if answerable
            else None
        ),
        unanswerable_refusal_rate=(
            statistics.fmean(
                run.status == ResearchStatus.INSUFFICIENT_EVIDENCE.value
                for run in unanswerable
            )
            if unanswerable
            else None
        ),
        evidence_recall=_mean_optional([run.evidence_recall for run in answerable]),
        citation_precision=_mean_optional(
            [run.citation_precision for run in answerable]
        ),
        citation_recall=_mean_optional([run.citation_recall for run in answerable]),
        failure_rate=statistics.fmean(
            run.status == ResearchStatus.FAILED.value for run in runs
        ),
        average_retrieval_calls=statistics.fmean(run.retrieval_calls for run in runs),
        average_rewrites=statistics.fmean(run.rewrite_count for run in runs),
        average_model_calls=statistics.fmean(run.model_calls for run in runs),
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
        prompt_tokens=_sum_tokens(runs, "prompt_tokens"),
        completion_tokens=_sum_tokens(runs, "completion_tokens"),
        total_tokens=_sum_tokens(runs, "total_tokens"),
        model_latency_ms=sum(run.model_latency_ms for run in runs),
    )


async def evaluate_research_agent(
    dataset: RetrievalDataset,
    *,
    backend: RetrievalBackend,
    baseline_model: ResearchModel,
    agent_model: ResearchModel,
    settings: Settings,
    work_dir: Path,
    index_profile: str,
    evidence_token_coverage_threshold: float = 0.6,
) -> AgentEvaluationReport:
    """Compare zero-rewrite and bounded-rewrite versions of the same graph."""
    if settings.research_max_rewrites < 1:
        raise ValueError("Agent evaluation requires research_max_rewrites >= 1")
    if baseline_model.name != agent_model.name:
        raise ValueError("Baseline and Agent must use the same model")
    if not 0.0 <= evidence_token_coverage_threshold <= 1.0:
        raise ValueError("evidence_token_coverage_threshold must be between 0 and 1")

    corpus = await index_evaluation_corpus(
        dataset,
        backend=backend,
        settings=settings,
        work_dir=work_dir,
        index_profile=index_profile,
    )
    baseline_settings = settings.model_copy(update={"research_max_rewrites": 0})
    baseline_graph = build_research_graph(
        retrieval=backend,
        model=baseline_model,
        settings=baseline_settings,
    )
    agent_graph = build_research_graph(
        retrieval=backend,
        model=agent_model,
        settings=settings,
    )

    comparisons: list[AgentQueryComparison] = []
    for query in dataset.queries:
        baseline = await _run_query(
            graph=baseline_graph,
            model=baseline_model,
            query=query,
            corpus=corpus,
            threshold=evidence_token_coverage_threshold,
        )
        agent = await _run_query(
            graph=agent_graph,
            model=agent_model,
            query=query,
            corpus=corpus,
            threshold=evidence_token_coverage_threshold,
        )
        comparisons.append(
            AgentQueryComparison(
                query_id=query.query_id,
                query=query.text,
                answerable=query.answerable,
                baseline=baseline,
                agent=agent,
            )
        )

    baseline_metrics = _aggregate(comparisons, "baseline")
    agent_metrics = _aggregate(comparisons, "agent")
    evidence_delta = (
        agent_metrics.evidence_recall - baseline_metrics.evidence_recall
        if agent_metrics.evidence_recall is not None
        and baseline_metrics.evidence_recall is not None
        else None
    )
    token_delta = (
        agent_metrics.total_tokens - baseline_metrics.total_tokens
        if agent_metrics.total_tokens is not None
        and baseline_metrics.total_tokens is not None
        else None
    )
    return AgentEvaluationReport(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_sha256=corpus.dataset_sha256,
        dataset_kind=dataset.kind,
        split=dataset.split,
        backend=backend.name,
        index_profile=index_profile,
        model=agent_model.name,
        document_count=len(dataset.documents),
        query_count=len(dataset.queries),
        answerable_query_count=sum(query.answerable for query in dataset.queries),
        unanswerable_query_count=sum(not query.answerable for query in dataset.queries),
        search_top_k=settings.research_search_top_k,
        agent_max_rewrites=settings.research_max_rewrites,
        evidence_token_coverage_threshold=evidence_token_coverage_threshold,
        indexing_latency_ms=corpus.indexing_latency_ms,
        baseline=baseline_metrics,
        agent=agent_metrics,
        delta=AgentMetricDelta(
            outcome_accuracy=(
                agent_metrics.outcome_accuracy - baseline_metrics.outcome_accuracy
            ),
            evidence_recall=evidence_delta,
            average_retrieval_calls=(
                agent_metrics.average_retrieval_calls
                - baseline_metrics.average_retrieval_calls
            ),
            average_rewrites=(
                agent_metrics.average_rewrites - baseline_metrics.average_rewrites
            ),
            average_model_calls=(
                agent_metrics.average_model_calls - baseline_metrics.average_model_calls
            ),
            latency_p50_ms=(
                agent_metrics.latency_p50_ms - baseline_metrics.latency_p50_ms
            ),
            total_tokens=token_delta,
        ),
        queries=comparisons,
        limitations=[
            "Evidence recall is cumulative across retrieval rounds, not Recall@K.",
            "Citation metrics measure overlap with labelled evidence, not semantic answer correctness.",
            "Token totals are null when the model provider omits usage telemetry.",
        ],
    )


def write_agent_evaluation_report(report: AgentEvaluationReport, path: Path) -> None:
    """Write a stable UTF-8 JSON Agent comparison report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def agent_report_summary(report: AgentEvaluationReport) -> str:
    """Print the central quality-versus-cost comparison without overclaiming."""
    return json.dumps(
        {
            "dataset": report.dataset_name,
            "dataset_sha256": report.dataset_sha256,
            "kind": report.dataset_kind,
            "backend": report.backend,
            "model": report.model,
            "queries": report.query_count,
            "answerable_queries": report.answerable_query_count,
            "unanswerable_queries": report.unanswerable_query_count,
            "baseline": report.baseline.model_dump(mode="json"),
            "agent": report.agent.model_dump(mode="json"),
            "delta": report.delta.model_dump(mode="json"),
        },
        indent=2,
    )
