"""Command-line entry points for dataset preparation and retrieval evaluation."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from paperops.clients.protocols import RetrievalBackend
from paperops.evaluation.agent import (
    agent_report_summary,
    evaluate_research_agent,
    write_agent_evaluation_report,
)
from paperops.evaluation.models import DatasetKind
from paperops.evaluation.qasper import convert_qasper, write_retrieval_dataset
from paperops.evaluation.retrieval import (
    evaluate_retrieval_backend,
    load_retrieval_dataset,
    report_summary,
    write_evaluation_report,
)
from paperops.research.fakes import FakeResearchModel
from paperops.research.openai_compatible import OpenAICompatibleResearchModel
from paperops.research.protocols import ResearchModel
from paperops.retrieval import (
    DenseRetrievalBackend,
    FastEmbedProvider,
    FastEmbedReranker,
    HybridRetrievalBackend,
    NativeRetrievalBackend,
    RerankedRetrievalBackend,
)
from paperops.settings import Settings

_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _top_k(value: str) -> tuple[int, ...]:
    try:
        limits = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "top-k must be comma-separated integers"
        ) from exc
    if not limits or limits[0] < 1:
        raise argparse.ArgumentTypeError("top-k values must be positive")
    return limits


def _add_backend_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--strategy",
        choices=("native", "dense", "hybrid", "hybrid-reranked"),
        default="native",
    )
    command.add_argument("--embedding-model", default=_DEFAULT_EMBEDDING_MODEL)
    command.add_argument("--reranker-model", default=_DEFAULT_RERANKER_MODEL)
    command.add_argument("--candidate-k", type=_positive_integer, default=20)
    command.add_argument("--rrf-k", type=_positive_integer, default=60)
    command.add_argument(
        "--model-cache",
        type=Path,
        default=Path(".paperops-eval/model-cache"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paperops-eval")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare-qasper",
        help="convert an official downloaded QASPER JSON split",
    )
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--split", required=True)
    prepare.add_argument("--max-documents", type=_positive_integer)
    prepare.add_argument("--max-queries", type=_positive_integer)
    prepare.add_argument("--max-answerable-queries", type=_positive_integer)
    prepare.add_argument("--max-unanswerable-queries", type=_positive_integer)
    prepare.add_argument(
        "--include-unanswerable",
        action="store_true",
        help="include only unanimously unanswerable annotations for refusal metrics",
    )

    evaluate = commands.add_parser(
        "evaluate",
        aliases=["evaluate-native"],
        help="compare sparse, dense, hybrid, or reranked local retrieval",
    )
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--work-dir", type=Path, default=Path(".paperops-eval"))
    evaluate.add_argument("--top-k", type=_top_k, default=(1, 3, 5, 10))
    evaluate.add_argument("--chunk-size", type=_positive_integer, default=1200)
    evaluate.add_argument("--chunk-overlap", type=int, default=160)
    evaluate.add_argument("--evidence-coverage", type=float, default=0.6)
    _add_backend_arguments(evaluate)

    agent = commands.add_parser(
        "evaluate-agent",
        help="compare the same research graph with zero versus bounded rewrites",
    )
    agent.add_argument("--dataset", type=Path, required=True)
    agent.add_argument("--output", type=Path, required=True)
    agent.add_argument("--work-dir", type=Path, default=Path(".paperops-agent-eval"))
    agent.add_argument("--chunk-size", type=_positive_integer, default=1200)
    agent.add_argument("--chunk-overlap", type=int, default=160)
    agent.add_argument("--evidence-coverage", type=float, default=0.6)
    agent.add_argument("--search-top-k", type=_positive_integer, default=10)
    agent.add_argument("--max-rewrites", type=_positive_integer, default=2)
    agent.add_argument("--min-evidence-hits", type=_positive_integer, default=1)
    agent.add_argument("--min-confidence", type=float, default=0.65)
    _add_backend_arguments(agent)
    return parser


def _build_backend(
    args: argparse.Namespace,
    settings: Settings,
) -> tuple[RetrievalBackend, str]:
    sparse = NativeRetrievalBackend(settings)
    if args.strategy == "native":
        return sparse, "native-fts5-bm25"

    embedding = FastEmbedProvider(
        args.embedding_model,
        cache_dir=args.model_cache,
    )
    index_profile = f"fastembed:{args.embedding_model}"
    dense = DenseRetrievalBackend(settings, embedding)
    if args.strategy == "dense":
        return dense, index_profile

    hybrid = HybridRetrievalBackend(
        sparse,
        dense,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )
    if args.strategy == "hybrid":
        return hybrid, index_profile

    reranker = FastEmbedReranker(
        args.reranker_model,
        cache_dir=args.model_cache,
    )
    return (
        RerankedRetrievalBackend(
            hybrid,
            reranker,
            candidate_k=args.candidate_k,
        ),
        index_profile,
    )


async def _evaluate(args: argparse.Namespace) -> int:
    dataset = load_retrieval_dataset(args.dataset)
    if dataset.kind == DatasetKind.SMOKE_FIXTURE:
        print(
            "warning: smoke_fixture validates wiring only; do not report its scores "
            "as benchmark results",
            file=sys.stderr,
        )
    work_dir: Path = args.work_dir
    settings = Settings(
        native_index_db=work_dir / "native-index.db",
        native_chunk_size_chars=args.chunk_size,
        native_chunk_overlap_chars=args.chunk_overlap,
        native_search_top_k=max(max(args.top_k), args.candidate_k),
    )
    backend, index_profile = _build_backend(args, settings)
    report = await evaluate_retrieval_backend(
        dataset,
        backend=backend,
        settings=settings,
        work_dir=work_dir,
        index_profile=index_profile,
        top_k=args.top_k,
        evidence_token_coverage_threshold=args.evidence_coverage,
    )
    write_evaluation_report(report, args.output)
    print(report_summary(report))
    return 0


def _build_research_model(settings: Settings) -> ResearchModel:
    if settings.research_model_mode == "fake":
        return FakeResearchModel()
    return OpenAICompatibleResearchModel(settings)


async def _evaluate_agent(args: argparse.Namespace) -> int:
    dataset = load_retrieval_dataset(args.dataset)
    if dataset.kind == DatasetKind.SMOKE_FIXTURE:
        print(
            "warning: smoke_fixture and fake model runs validate wiring only; "
            "do not report their scores as benchmark results",
            file=sys.stderr,
        )
    work_dir: Path = args.work_dir
    settings = Settings(
        native_index_db=work_dir / "native-index.db",
        native_chunk_size_chars=args.chunk_size,
        native_chunk_overlap_chars=args.chunk_overlap,
        native_search_top_k=max(args.search_top_k, args.candidate_k),
        research_search_top_k=args.search_top_k,
        research_max_rewrites=args.max_rewrites,
        research_min_evidence_hits=args.min_evidence_hits,
        research_min_assessment_confidence=args.min_confidence,
    )
    backend, index_profile = _build_backend(args, settings)
    baseline_model = _build_research_model(settings)
    agent_model = _build_research_model(settings)
    try:
        report = await evaluate_research_agent(
            dataset,
            backend=backend,
            baseline_model=baseline_model,
            agent_model=agent_model,
            settings=settings,
            work_dir=work_dir,
            index_profile=index_profile,
            evidence_token_coverage_threshold=args.evidence_coverage,
        )
    finally:
        for model in (baseline_model, agent_model):
            if isinstance(model, OpenAICompatibleResearchModel):
                await model.aclose()
    write_agent_evaluation_report(report, args.output)
    print(agent_report_summary(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch one explicit offline evaluation command."""
    args = _parser().parse_args(argv)
    if args.command == "prepare-qasper":
        dataset = convert_qasper(
            args.input,
            split=args.split,
            max_documents=args.max_documents,
            max_queries=args.max_queries,
            max_answerable_queries=args.max_answerable_queries,
            max_unanswerable_queries=args.max_unanswerable_queries,
            include_unanswerable=args.include_unanswerable,
        )
        write_retrieval_dataset(dataset, args.output)
        print(
            f"wrote {len(dataset.documents)} documents and "
            f"{len(dataset.queries)} queries to {args.output}"
        )
        return 0
    if args.command == "evaluate-agent":
        return asyncio.run(_evaluate_agent(args))
    return asyncio.run(_evaluate(args))


if __name__ == "__main__":
    raise SystemExit(main())
