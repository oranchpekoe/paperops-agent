"""Tests for the fixed QASPER multi-paper diagnostic converter."""

from __future__ import annotations

import json
from pathlib import Path

from paperops.comparison.models import ComparisonCellStatus
from paperops.evaluation.cli import main
from paperops.evaluation.models import DatasetKind
from paperops.evaluation.qasper_comparison import (
    DEFAULT_QASPER_COMPARISON_SPECS,
    HELDOUT_QASPER_COMPARISON_SPECS,
    QasperComparisonTaskSpec,
    convert_qasper_comparison,
    write_comparison_dataset,
)


def _paper(title: str, evidence: str, *, answerable: bool) -> dict:
    return {
        "title": title,
        "abstract": "A test abstract.",
        "full_text": [
            {
                "section_name": "Experiment",
                "paragraphs": [evidence],
            }
        ],
        "qas": [
            {
                "question_id": f"query-{title[-1].lower()}",
                "question": "Which evaluation metric is reported?",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": not answerable,
                            "evidence": [evidence] if answerable else [],
                        }
                    }
                ],
            }
        ],
    }


def test_qasper_comparison_converter_preserves_supported_and_missing_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "qasper.json"
    source.write_text(
        json.dumps(
            {
                "paper-a": _paper(
                    "Paper A",
                    "Paper A reports exact-match accuracy.",
                    answerable=True,
                ),
                "paper-b": _paper(
                    "Paper B",
                    "Paper B describes training but reports no metric.",
                    answerable=False,
                ),
            }
        ),
        encoding="utf-8",
    )
    specs = (
        QasperComparisonTaskSpec(
            task_id="metrics",
            document_ids=("paper-a", "paper-b"),
            dimensions=(("metric", "evaluation metric"),),
            cells=(
                ("paper-a", "metric", "query-a"),
                ("paper-b", "metric", "query-b"),
            ),
        ),
    )

    dataset = convert_qasper_comparison(
        source,
        split="validation",
        specs=specs,
    )

    assert dataset.kind is DatasetKind.BENCHMARK
    assert len(dataset.documents) == 2
    assert len(dataset.tasks) == 1
    supported, missing = dataset.tasks[0].expected_cells
    assert supported.status is ComparisonCellStatus.SUPPORTED
    assert supported.evidence[0].text == ("Paper A reports exact-match accuracy.")
    assert supported.source_query_id == "query-a"
    assert supported.source_question == "Which evaluation metric is reported?"
    assert missing.status is ComparisonCellStatus.MISSING
    assert missing.evidence == []
    assert missing.source_query_id == "query-b"


def test_frozen_heldout_profile_uses_distinct_papers() -> None:
    development_documents = {
        document_id
        for spec in DEFAULT_QASPER_COMPARISON_SPECS
        for document_id in spec.document_ids
    }
    heldout_documents = {
        document_id
        for spec in HELDOUT_QASPER_COMPARISON_SPECS
        for document_id in spec.document_ids
    }

    assert development_documents.isdisjoint(heldout_documents)


def test_comparison_retrieval_cli_projects_labels_without_a_model(
    tmp_path: Path,
) -> None:
    source = tmp_path / "qasper.json"
    source.write_text(
        json.dumps(
            {
                "paper-a": _paper(
                    "Paper A",
                    "Paper A reports the evaluation metric exact-match accuracy.",
                    answerable=True,
                ),
                "paper-b": _paper(
                    "Paper B",
                    "Paper B describes training but reports no metric.",
                    answerable=False,
                ),
            }
        ),
        encoding="utf-8",
    )
    specs = (
        QasperComparisonTaskSpec(
            task_id="metrics",
            document_ids=("paper-a", "paper-b"),
            dimensions=(("metric", "evaluation metric"),),
            cells=(
                ("paper-a", "metric", "query-a"),
                ("paper-b", "metric", "query-b"),
            ),
        ),
    )
    dataset_path = tmp_path / "comparison.json"
    report_path = tmp_path / "report.json"
    write_comparison_dataset(
        convert_qasper_comparison(source, split="validation", specs=specs),
        dataset_path,
    )

    exit_code = main(
        [
            "evaluate-comparison-retrieval",
            "--dataset",
            str(dataset_path),
            "--output",
            str(report_path),
            "--work-dir",
            str(tmp_path / "work"),
            "--strategy",
            "native",
            "--top-k",
            "1",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["backend"] == "native_fts5_bm25"
    assert report["document_count"] == 2
    assert report["query_count"] == 1
