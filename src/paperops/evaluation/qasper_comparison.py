"""Build a fixed multi-paper comparison diagnostic from official QASPER."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paperops.comparison.models import ComparisonCellStatus, ComparisonDimension
from paperops.evaluation.comparison_models import (
    ComparisonEvaluationDataset,
    ComparisonEvaluationTask,
    ComparisonExpectedCell,
)
from paperops.evaluation.qasper import convert_qasper


@dataclass(frozen=True, slots=True)
class QasperComparisonTaskSpec:
    """Curated QASPER query mapping for one shared comparison matrix."""

    task_id: str
    document_ids: tuple[str, ...]
    dimensions: tuple[tuple[str, str], ...]
    cells: tuple[tuple[str, str, str], ...]


# The fixed profiles intentionally use supported labels only. A QASPER question's
# unanswerable label does not establish that a broader comparison dimension is absent
# from the whole paper, so projecting it to a `missing` cell would create false ground
# truth. Missing-cell behavior remains covered by controlled tests.
DEFAULT_QASPER_COMPARISON_SPECS = (
    QasperComparisonTaskSpec(
        task_id="experimental-setup",
        document_ids=(
            "1912.01214",
            "1708.01464",
            "1810.04428",
            "1905.12801",
            "1611.08661",
            "1806.04511",
        ),
        dimensions=(
            ("datasets", "datasets or corpora used in the experiments"),
            (
                "baselines",
                "baseline systems or comparison methods used in the experiments",
            ),
        ),
        cells=(
            (
                "1912.01214",
                "datasets",
                "9a05a5f4351db75da371f7ac12eb0b03607c4b87",
            ),
            (
                "1912.01214",
                "baselines",
                "b6f15fb6279b82e34a5bf4828b7b5ddabfdf1d54",
            ),
            (
                "1708.01464",
                "datasets",
                "4eaf9787f51cd7cdc45eb85cf223d752328c6ee4",
            ),
            (
                "1708.01464",
                "baselines",
                "0752d71a0a1f73b3482a888313622ce9e9870d6e",
            ),
            (
                "1810.04428",
                "datasets",
                "ebf0d9f9260ed61cbfd79b962df3899d05f9ebfb",
            ),
            (
                "1810.04428",
                "baselines",
                "326588b1de9ba0fd049ab37c907e6e5413e14acd",
            ),
            (
                "1905.12801",
                "datasets",
                "73bddaaf601a4f944a3182ca0f4de85a19cdc1d2",
            ),
            (
                "1905.12801",
                "baselines",
                "c59078efa7249acfb9043717237c96ae762c0a8c",
            ),
            (
                "1611.08661",
                "datasets",
                "b36f867fcda5ad62c46d23513369337352aa01d2",
            ),
            (
                "1611.08661",
                "baselines",
                "932b39fd6c47c6a880621a62e6a978491d881d60",
            ),
            (
                "1806.04511",
                "datasets",
                "f1f1dcc67b3e4d554bfeb508226cdadb3c32d2e9",
            ),
            (
                "1806.04511",
                "baselines",
                "a103636c8d1dbfa53341133aeb751ffec269415c",
            ),
        ),
    ),
    QasperComparisonTaskSpec(
        task_id="evaluation-metrics",
        document_ids=("1708.01464", "1810.04428", "1905.12801"),
        dimensions=(
            ("evaluation_metrics", "evaluation metrics used in the experiments"),
        ),
        cells=(
            (
                "1708.01464",
                "evaluation_metrics",
                "55c8f7acbfd4f5cde634aaecd775b3bb32e9ffa3",
            ),
            (
                "1810.04428",
                "evaluation_metrics",
                "f651cd144b7749e82aa1374779700812f64c8799",
            ),
            (
                "1905.12801",
                "evaluation_metrics",
                "90d946ccc3abf494890e147dd85bd489b8f3f0e8",
            ),
        ),
    ),
)

HELDOUT_QASPER_COMPARISON_SPECS = (
    QasperComparisonTaskSpec(
        task_id="heldout-experimental-setup",
        document_ids=(
            "1903.09722",
            "1910.12574",
            "2002.01984",
            "1604.00727",
            "1908.09246",
        ),
        dimensions=(
            ("datasets", "datasets or corpora used in the experiments"),
            (
                "baselines",
                "baseline systems or comparison methods used in the experiments",
            ),
        ),
        cells=(
            (
                "1903.09722",
                "datasets",
                "6ca938324dc7e1742a840d0a54dc13cc207394a1",
            ),
            (
                "1903.09722",
                "baselines",
                "4fa6fbb9df1a4c32583d4ef70d2b29ece4b3d802",
            ),
            (
                "1910.12574",
                "datasets",
                "c81f215d457bdb913a5bade2b4283f19c4ee826c",
            ),
            (
                "1910.12574",
                "baselines",
                "81a35b9572c9d574a30cc2164f47750716157fc8",
            ),
            (
                "2002.01984",
                "datasets",
                "e807d347742b2799bc347c0eff19b4c270449fee",
            ),
            (
                "2002.01984",
                "baselines",
                "ff338921e34c15baf1eae0074938bf79ee65fdd2",
            ),
            (
                "1604.00727",
                "datasets",
                "784ce5a983c5f2cc95a2c60ce66f2a8a50f3636f",
            ),
            (
                "1604.00727",
                "baselines",
                "7705dd04acedaefee30d8b2c9978537afb2040dc",
            ),
            (
                "1908.09246",
                "datasets",
                "56b034c303983b2e276ed6518d6b080f7b8abe6a",
            ),
            (
                "1908.09246",
                "baselines",
                "0602a974a879e6eae223cdf048410b5a0111665e",
            ),
        ),
    ),
)


def convert_qasper_comparison(
    source: Path,
    *,
    split: str,
    specs: tuple[QasperComparisonTaskSpec, ...] = DEFAULT_QASPER_COMPARISON_SPECS,
    profile_name: str = "development",
) -> ComparisonEvaluationDataset:
    """Select fixed QASPER labels without changing their evidence or status."""
    source_dataset = convert_qasper(
        source,
        split=split,
        include_unanswerable=True,
    )
    documents = {item.document_id: item for item in source_dataset.documents}
    queries = {item.query_id: item for item in source_dataset.queries}
    selected_document_ids: list[str] = []
    tasks: list[ComparisonEvaluationTask] = []

    for spec in specs:
        dimensions = [
            ComparisonDimension(dimension_id=dimension_id, description=description)
            for dimension_id, description in spec.dimensions
        ]
        expected_cells: list[ComparisonExpectedCell] = []
        for document_id, dimension_id, query_id in spec.cells:
            if document_id not in documents:
                raise ValueError(
                    f"QASPER comparison spec references missing document {document_id}"
                )
            query = queries.get(query_id)
            if query is None:
                raise ValueError(
                    f"QASPER comparison spec references missing query {query_id}"
                )
            if query.document_id != document_id:
                raise ValueError(
                    f"QASPER query {query_id} belongs to {query.document_id}, "
                    f"not {document_id}"
                )
            expected_cells.append(
                ComparisonExpectedCell(
                    document_id=document_id,
                    dimension_id=dimension_id,
                    status=(
                        ComparisonCellStatus.SUPPORTED
                        if query.answerable
                        else ComparisonCellStatus.MISSING
                    ),
                    evidence=query.evidence,
                    source_query_id=query.query_id,
                    source_question=query.text,
                )
            )
        tasks.append(
            ComparisonEvaluationTask(
                task_id=spec.task_id,
                document_ids=list(spec.document_ids),
                dimensions=dimensions,
                expected_cells=expected_cells,
            )
        )
        for document_id in spec.document_ids:
            if document_id not in selected_document_ids:
                selected_document_ids.append(document_id)

    return ComparisonEvaluationDataset(
        name=f"qasper-multi-paper-comparison-{profile_name}",
        version=f"0.3-paperops-comparison-{profile_name}-v1",
        kind=source_dataset.kind,
        split=source_dataset.split,
        source_url=source_dataset.source_url,
        license=source_dataset.license,
        documents=[documents[document_id] for document_id in selected_document_ids],
        tasks=tasks,
    )


def write_comparison_dataset(
    dataset: ComparisonEvaluationDataset,
    destination: Path,
) -> None:
    """Persist a generated comparison dataset outside the repository."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
