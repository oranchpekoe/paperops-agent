"""Evaluate bounded comparison gap retrieval from a shared initial matrix."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from time import perf_counter
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.state import CompiledStateGraph

from paperops.clients.protocols import RetrievalBackend
from paperops.comparison.graph import build_comparison_graph
from paperops.comparison.models import (
    ComparisonCell,
    ComparisonCellStatus,
    ComparisonDimension,
    ComparisonDocument,
    ComparisonEvent,
    ComparisonFailure,
    ComparisonFailureCode,
    ComparisonSearchAttempt,
    ComparisonStatus,
    ComparisonStopReason,
)
from paperops.comparison.protocols import ComparisonModel
from paperops.comparison.state import ComparisonState
from paperops.evaluation.comparison_models import (
    ComparisonCellEvaluation,
    ComparisonEvaluationDataset,
    ComparisonEvaluationReport,
    ComparisonEvaluationTask,
    ComparisonMetricDelta,
    ComparisonTaskComparison,
    ComparisonTaskRun,
    ComparisonVariantMetrics,
)
from paperops.evaluation.models import EvidenceReference
from paperops.evaluation.retrieval import (
    EvaluationCorpusIndex,
    index_evaluation_corpus,
    matches_evidence,
    percentile,
)
from paperops.research.models import EvidenceCitation, ModelCallUsage
from paperops.settings import Settings

ComparisonGraph = CompiledStateGraph[
    ComparisonState,
    None,
    ComparisonState,
    ComparisonState,
]


def load_comparison_dataset(path: Path) -> ComparisonEvaluationDataset:
    """Load and validate one UTF-8 comparison dataset."""
    return ComparisonEvaluationDataset.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _evaluation_checkpointer() -> InMemorySaver:
    """Allow only explicit comparison state types in paired checkpoints."""
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=(
            ComparisonStatus,
            ComparisonCellStatus,
            ComparisonStopReason,
            ComparisonFailureCode,
            ComparisonDocument,
            ComparisonDimension,
            ComparisonCell,
            ComparisonSearchAttempt,
            ComparisonFailure,
            ComparisonEvent,
            EvidenceCitation,
        )
    )
    return InMemorySaver(serde=serializer)


def _token_totals(
    usage: list[ModelCallUsage],
    model_calls: int,
) -> tuple[int | None, int | None, int | None, float]:
    """Aggregate provider usage without inventing omitted token telemetry."""
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


def _matched_evidence_ids(
    citations: list[EvidenceCitation],
    *,
    expected_evidence: list[EvidenceReference],
    corpus: EvaluationCorpusIndex,
    threshold: float,
) -> set[str]:
    """Return paragraph labels covered by the supplied backend citations."""
    matched: set[str] = set()
    for citation in citations:
        logical_document_id = corpus.backend_to_logical_document_id.get(
            citation.document_id
        )
        matched.update(
            item.evidence_id
            for item in expected_evidence
            if matches_evidence(
                content=citation.content,
                logical_document_id=logical_document_id,
                evidence=item,
                threshold=threshold,
            )
        )
    return matched


def _evaluate_cells(
    *,
    result: ComparisonState,
    task: ComparisonEvaluationTask,
    corpus: EvaluationCorpusIndex,
    threshold: float,
) -> list[ComparisonCellEvaluation]:
    """Score predicted statuses and citations against the complete task matrix."""
    backend_cells = {
        (
            corpus.backend_to_logical_document_id.get(
                cell.document_id,
                cell.document_id,
            ),
            cell.dimension_id,
        ): cell
        for cell in result.get("cells", [])
    }
    evidence_by_id = {item.citation_id: item for item in result.get("evidence", [])}
    failed = result.get("status") is ComparisonStatus.FAILED
    evaluations: list[ComparisonCellEvaluation] = []
    for expected in task.expected_cells:
        key = (expected.document_id, expected.dimension_id)
        actual_backend = backend_cells.get(key)
        if actual_backend is None:
            actual = ComparisonCell(
                document_id=expected.document_id,
                dimension_id=expected.dimension_id,
                status=ComparisonCellStatus.MISSING,
                confidence=0.0,
                missing_reason="The workflow did not return this matrix cell.",
                suggested_query="No query was produced because execution failed.",
            )
        else:
            actual = actual_backend.model_copy(
                update={"document_id": expected.document_id}
            )

        matched = _matched_evidence_ids(
            list(evidence_by_id.values()),
            expected_evidence=expected.evidence,
            corpus=corpus,
            threshold=threshold,
        )
        evidence_recall = (
            len(matched) / len(expected.evidence)
            if expected.status is ComparisonCellStatus.SUPPORTED
            else None
        )
        citation_precision: float | None = None
        citation_recall: float | None = None
        if expected.status is ComparisonCellStatus.SUPPORTED:
            cited = [
                evidence_by_id[item]
                for item in actual.citation_ids
                if item in evidence_by_id
            ]
            cited_matches = _matched_evidence_ids(
                cited,
                expected_evidence=expected.evidence,
                corpus=corpus,
                threshold=threshold,
            )
            citation_precision = (
                sum(
                    bool(
                        _matched_evidence_ids(
                            [citation],
                            expected_evidence=expected.evidence,
                            corpus=corpus,
                            threshold=threshold,
                        )
                    )
                    for citation in cited
                )
                / len(cited)
                if cited
                else 0.0
            )
            citation_recall = len(cited_matches) / len(expected.evidence)

        status_correct = not failed and actual.status is expected.status
        grounded_correct = status_correct and (
            expected.status is ComparisonCellStatus.MISSING or bool(citation_recall)
        )
        evaluations.append(
            ComparisonCellEvaluation(
                document_id=expected.document_id,
                dimension_id=expected.dimension_id,
                expected_status=expected.status,
                actual=actual,
                status_correct=status_correct,
                grounded_correct=grounded_correct,
                matched_evidence_ids=sorted(matched),
                evidence_recall=evidence_recall,
                citation_precision=citation_precision,
                citation_recall=citation_recall,
            )
        )
    return evaluations


def _evaluate_run(
    *,
    result: ComparisonState,
    usage: list[ModelCallUsage],
    latency_ms: float,
    task: ComparisonEvaluationTask,
    corpus: EvaluationCorpusIndex,
    threshold: float,
) -> ComparisonTaskRun:
    """Convert graph state into a reproducible task-level evaluation."""
    status = result.get("status", ComparisonStatus.FAILED)
    model_calls = result.get("model_calls", 0)
    prompt, completion, total, model_latency = _token_totals(usage, model_calls)
    failure = result.get("failure")
    stop_reason = result.get("stop_reason")
    return ComparisonTaskRun(
        status=status.value if isinstance(status, ComparisonStatus) else str(status),
        latency_ms=latency_ms,
        cells=_evaluate_cells(
            result=result,
            task=task,
            corpus=corpus,
            threshold=threshold,
        ),
        retrieval_calls=result.get("retrieval_calls", 0),
        attempted_searches=[
            item.model_copy(
                update={
                    "document_id": corpus.backend_to_logical_document_id.get(
                        item.document_id,
                        item.document_id,
                    )
                }
            )
            for item in result.get("attempted_searches", [])
        ],
        evidence_count=len(result.get("evidence", [])),
        new_evidence_count=result.get("new_evidence_count", 0),
        gap_rounds=result.get("gap_round", 0),
        model_calls=model_calls,
        recovered_cell_count=result.get("recovered_cell_count", 0),
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        model_latency_ms=model_latency,
        failure_code=failure.code.value if failure is not None else None,
        failure_message=failure.message if failure is not None else None,
        stop_reason=(
            stop_reason.value
            if isinstance(stop_reason, ComparisonStopReason)
            else stop_reason
        ),
    )


async def _resume_to_terminal(
    *,
    graph: ComparisonGraph,
    model: ComparisonModel,
    config: RunnableConfig,
    initial: ComparisonState,
    max_resumes: int,
) -> tuple[ComparisonState, list[ModelCallUsage], float]:
    """Resume a bounded interrupted graph until no next node remains."""
    result = initial
    usage: list[ModelCallUsage] = []
    latency_ms = 0.0
    for _ in range(max_resumes):
        snapshot = await graph.aget_state(config)
        if not snapshot.next:
            return result, usage, latency_ms
        started = perf_counter()
        result = cast(ComparisonState, await graph.ainvoke(None, config))
        latency_ms += (perf_counter() - started) * 1000
        usage.extend(model.drain_usage())
    raise RuntimeError("Comparison evaluation exceeded its bounded resume count")


async def _run_paired_task(
    *,
    graph: ComparisonGraph,
    model: ComparisonModel,
    settings: Settings,
    task: ComparisonEvaluationTask,
    document_titles: dict[str, str],
    corpus: EvaluationCorpusIndex,
    threshold: float,
) -> tuple[ComparisonTaskRun, ComparisonTaskRun]:
    """Share initial retrieval and extraction, then continue only the Agent arm."""
    config: RunnableConfig = {
        "configurable": {
            "thread_id": f"comparison-{corpus.dataset_sha256[:12]}-{task.task_id}"
        }
    }
    documents = [
        ComparisonDocument(
            document_id=corpus.logical_to_backend_document_id[document_id],
            label=document_titles[document_id],
        )
        for document_id in task.document_ids
    ]
    model.drain_usage()
    started = perf_counter()
    prefix = cast(
        ComparisonState,
        await graph.ainvoke(
            {
                "knowledge_base": corpus.collection_id,
                "documents": documents,
                "dimensions": task.dimensions,
            },
            config,
        ),
    )
    prefix_latency_ms = (perf_counter() - started) * 1000
    prefix_usage = model.drain_usage()
    snapshot = await graph.aget_state(config)

    if prefix.get("status") is ComparisonStatus.FAILED or not snapshot.next:
        run = _evaluate_run(
            result=prefix,
            usage=prefix_usage,
            latency_ms=prefix_latency_ms,
            task=task,
            corpus=corpus,
            threshold=threshold,
        )
        return run.model_copy(deep=True), run

    baseline_state = cast(ComparisonState, dict(prefix))
    baseline_state["status"] = ComparisonStatus.COMPLETED
    baseline_state["cells"] = [
        item.model_copy(deep=True) for item in prefix.get("initial_cells", [])
    ]
    baseline_state["gap_round"] = 0
    baseline_state["recovered_cell_count"] = 0
    baseline_state["stop_reason"] = ComparisonStopReason.GAP_BUDGET_EXHAUSTED
    baseline = _evaluate_run(
        result=baseline_state,
        usage=prefix_usage,
        latency_ms=prefix_latency_ms,
        task=task,
        corpus=corpus,
        threshold=threshold,
    )
    final, continuation_usage, continuation_latency_ms = await _resume_to_terminal(
        graph=graph,
        model=model,
        config=config,
        initial=prefix,
        max_resumes=settings.comparison_max_gap_rounds + 2,
    )
    agent = _evaluate_run(
        result=final,
        usage=[*prefix_usage, *continuation_usage],
        latency_ms=prefix_latency_ms + continuation_latency_ms,
        task=task,
        corpus=corpus,
        threshold=threshold,
    )
    return baseline, agent


def _mean_optional(values: list[float | None]) -> float | None:
    """Average present values while preserving an undefined population."""
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _sum_tokens(runs: list[ComparisonTaskRun], field: str) -> int | None:
    """Return null if any provider call omitted token telemetry."""
    values = [getattr(run, field) for run in runs]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _aggregate(
    comparisons: list[ComparisonTaskComparison],
    variant: str,
) -> ComparisonVariantMetrics:
    """Aggregate task costs and cell-level quality for one variant."""
    runs = [getattr(item, variant) for item in comparisons]
    cells = [cell for run in runs for cell in run.cells]
    supported = [
        cell for cell in cells if cell.expected_status is ComparisonCellStatus.SUPPORTED
    ]
    missing = [
        cell for cell in cells if cell.expected_status is ComparisonCellStatus.MISSING
    ]
    latencies = [run.latency_ms for run in runs]
    return ComparisonVariantMetrics(
        status_accuracy=statistics.fmean(cell.status_correct for cell in cells),
        grounded_accuracy=statistics.fmean(cell.grounded_correct for cell in cells),
        supported_completion_rate=(
            statistics.fmean(cell.status_correct for cell in supported)
            if supported
            else None
        ),
        missing_refusal_rate=(
            statistics.fmean(cell.status_correct for cell in missing)
            if missing
            else None
        ),
        evidence_recall=_mean_optional([cell.evidence_recall for cell in supported]),
        citation_precision=_mean_optional(
            [cell.citation_precision for cell in supported]
        ),
        citation_recall=_mean_optional([cell.citation_recall for cell in supported]),
        failure_rate=statistics.fmean(
            run.status == ComparisonStatus.FAILED.value for run in runs
        ),
        stagnant_stop_rate=statistics.fmean(
            run.stop_reason == ComparisonStopReason.STAGNANT_RETRIEVAL.value
            for run in runs
        ),
        average_retrieval_calls=statistics.fmean(run.retrieval_calls for run in runs),
        average_gap_rounds=statistics.fmean(run.gap_rounds for run in runs),
        average_model_calls=statistics.fmean(run.model_calls for run in runs),
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
        prompt_tokens=_sum_tokens(runs, "prompt_tokens"),
        completion_tokens=_sum_tokens(runs, "completion_tokens"),
        total_tokens=_sum_tokens(runs, "total_tokens"),
        model_latency_ms=sum(run.model_latency_ms for run in runs),
    )


async def evaluate_comparison_agent(
    dataset: ComparisonEvaluationDataset,
    *,
    backend: RetrievalBackend,
    model: ComparisonModel,
    settings: Settings,
    work_dir: Path,
    index_profile: str,
    evidence_token_coverage_threshold: float = 0.6,
) -> ComparisonEvaluationReport:
    """Compare zero and bounded gap retrieval from an identical initial matrix."""
    if settings.comparison_max_gap_rounds < 1:
        raise ValueError("Comparison evaluation requires at least one gap round")
    if not 0.0 <= evidence_token_coverage_threshold <= 1.0:
        raise ValueError("evidence_token_coverage_threshold must be between 0 and 1")

    retrieval_dataset = dataset.as_retrieval_dataset()
    corpus = await index_evaluation_corpus(
        retrieval_dataset,
        backend=backend,
        settings=settings,
        work_dir=work_dir,
        index_profile=index_profile,
    )
    graph = build_comparison_graph(
        retrieval=backend,
        model=model,
        settings=settings,
        checkpointer=_evaluation_checkpointer(),
        interrupt_after=["extract_matrix"],
    )
    document_titles = {item.document_id: item.title for item in dataset.documents}
    comparisons: list[ComparisonTaskComparison] = []
    for task in dataset.tasks:
        baseline, agent = await _run_paired_task(
            graph=graph,
            model=model,
            settings=settings,
            task=task,
            document_titles=document_titles,
            corpus=corpus,
            threshold=evidence_token_coverage_threshold,
        )
        comparisons.append(
            ComparisonTaskComparison(
                task_id=task.task_id,
                baseline=baseline,
                agent=agent,
            )
        )

    baseline_metrics = _aggregate(comparisons, "baseline")
    agent_metrics = _aggregate(comparisons, "agent")
    baseline_missing = [
        baseline_cell
        for item in comparisons
        for baseline_cell in item.baseline.cells
        if baseline_cell.expected_status is ComparisonCellStatus.SUPPORTED
        and baseline_cell.actual.status is ComparisonCellStatus.MISSING
    ]
    agent_by_key = {
        (item.task_id, cell.document_id, cell.dimension_id): cell
        for item in comparisons
        for cell in item.agent.cells
    }
    recovered = sum(
        agent_by_key[
            (item.task_id, cell.document_id, cell.dimension_id)
        ].grounded_correct
        for item in comparisons
        for cell in item.baseline.cells
        if cell.expected_status is ComparisonCellStatus.SUPPORTED
        and cell.actual.status is ComparisonCellStatus.MISSING
    )
    token_delta = (
        agent_metrics.total_tokens - baseline_metrics.total_tokens
        if agent_metrics.total_tokens is not None
        and baseline_metrics.total_tokens is not None
        else None
    )
    evidence_delta = (
        agent_metrics.evidence_recall - baseline_metrics.evidence_recall
        if agent_metrics.evidence_recall is not None
        and baseline_metrics.evidence_recall is not None
        else None
    )
    cells = [cell for item in dataset.tasks for cell in item.expected_cells]
    return ComparisonEvaluationReport(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_sha256=corpus.dataset_sha256,
        dataset_kind=dataset.kind,
        split=dataset.split,
        backend=backend.name,
        index_profile=index_profile,
        model=model.name,
        comparison_protocol="shared_initial_matrix_then_gap_continuation_v1",
        document_count=len(dataset.documents),
        task_count=len(dataset.tasks),
        cell_count=len(cells),
        expected_supported_cell_count=sum(
            cell.status is ComparisonCellStatus.SUPPORTED for cell in cells
        ),
        expected_missing_cell_count=sum(
            cell.status is ComparisonCellStatus.MISSING for cell in cells
        ),
        search_top_k=settings.comparison_search_top_k,
        agent_max_gap_rounds=settings.comparison_max_gap_rounds,
        evidence_token_coverage_threshold=evidence_token_coverage_threshold,
        indexing_latency_ms=corpus.indexing_latency_ms,
        baseline=baseline_metrics,
        agent=agent_metrics,
        delta=ComparisonMetricDelta(
            status_accuracy=(
                agent_metrics.status_accuracy - baseline_metrics.status_accuracy
            ),
            grounded_accuracy=(
                agent_metrics.grounded_accuracy - baseline_metrics.grounded_accuracy
            ),
            evidence_recall=evidence_delta,
            average_retrieval_calls=(
                agent_metrics.average_retrieval_calls
                - baseline_metrics.average_retrieval_calls
            ),
            average_gap_rounds=(
                agent_metrics.average_gap_rounds - baseline_metrics.average_gap_rounds
            ),
            average_model_calls=(
                agent_metrics.average_model_calls - baseline_metrics.average_model_calls
            ),
            latency_p50_ms=(
                agent_metrics.latency_p50_ms - baseline_metrics.latency_p50_ms
            ),
            total_tokens=token_delta,
            baseline_missing_supported_cells=len(baseline_missing),
            recovered_supported_cells=recovered,
            supported_recovery_rate=(
                recovered / len(baseline_missing) if baseline_missing else None
            ),
            incremental_tokens_per_recovery=(
                token_delta / recovered
                if token_delta is not None and recovered > 0
                else None
            ),
        ),
        tasks=comparisons,
        limitations=[
            "A smoke_fixture validates wiring only and cannot establish product lift.",
            "Grounded correctness requires citation overlap with labelled evidence, not full semantic claim grading.",
            "Both variants share initial retrieval and extraction; only the Agent arm performs gap retrieval.",
            "Token totals are null when the model provider omits usage telemetry.",
        ],
    )


def write_comparison_evaluation_report(
    report: ComparisonEvaluationReport,
    path: Path,
) -> None:
    """Write a stable UTF-8 JSON comparison report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def comparison_report_summary(report: ComparisonEvaluationReport) -> str:
    """Print the central recovery-versus-cost result without overclaiming."""
    return json.dumps(
        {
            "dataset": report.dataset_name,
            "dataset_sha256": report.dataset_sha256,
            "kind": report.dataset_kind,
            "backend": report.backend,
            "model": report.model,
            "tasks": report.task_count,
            "cells": report.cell_count,
            "baseline": report.baseline.model_dump(mode="json"),
            "agent": report.agent.model_dump(mode="json"),
            "delta": report.delta.model_dump(mode="json"),
        },
        indent=2,
    )
