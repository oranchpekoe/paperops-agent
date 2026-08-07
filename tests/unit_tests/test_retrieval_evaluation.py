"""Tests for dataset conversion, evidence matching, and retrieval metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from paperops.evaluation.models import DatasetKind, RetrievalDataset
from paperops.evaluation.qasper import convert_qasper
from paperops.evaluation.retrieval import (
    evaluate_native_retrieval,
    evidence_token_coverage,
    load_retrieval_dataset,
)
from paperops.settings import Settings

FIXTURE = Path(__file__).parents[1] / "fixtures" / "retrieval" / "research_smoke.json"


def test_dataset_rejects_evidence_for_unknown_document() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["queries"][0]["evidence"][0]["document_id"] = "missing-paper"

    with pytest.raises(ValidationError, match="unknown documents"):
        RetrievalDataset.model_validate(payload)


def test_evidence_coverage_handles_a_chunk_split() -> None:
    evidence = "A centralized critic consumes joint observations from all vehicles"
    partial_chunk = "critic consumes joint observations from all vehicles"

    assert evidence_token_coverage(evidence, partial_chunk) == pytest.approx(7 / 9)


def test_qasper_converter_keeps_answerable_text_evidence(tmp_path: Path) -> None:
    source = tmp_path / "qasper-dev.json"
    source.write_text(
        json.dumps(
            {
                "paper-1": {
                    "title": "A Research Paper",
                    "abstract": "An abstract.",
                    "full_text": [
                        {
                            "section_name": "Method",
                            "paragraphs": [
                                "The model uses a centralized critic during training."
                            ],
                        }
                    ],
                    "qas": [
                        {
                            "question_id": "question-1",
                            "question": "What is used during training?",
                            "answers": [
                                {
                                    "answer": {
                                        "unanswerable": False,
                                        "evidence": [
                                            "The model uses a centralized critic during training."
                                        ],
                                    }
                                },
                                {
                                    "answer": {
                                        "unanswerable": False,
                                        "evidence": [
                                            "The model uses a centralized critic during training."
                                        ],
                                    }
                                },
                                {
                                    "answer": {
                                        "unanswerable": False,
                                        "evidence": ["FLOAT SELECTED: Figure 1"],
                                    }
                                },
                            ],
                        },
                        {
                            "question_id": "question-2",
                            "question": "What is unavailable?",
                            "answers": [
                                {"answer": {"unanswerable": True, "evidence": []}}
                            ],
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = convert_qasper(source, split="validation")

    assert dataset.kind == DatasetKind.BENCHMARK
    assert len(dataset.documents) == 1
    assert len(dataset.queries) == 1
    assert dataset.queries[0].evidence[0].relevance == 2


@pytest.mark.asyncio
async def test_native_evaluation_produces_traceable_metrics(tmp_path: Path) -> None:
    dataset = load_retrieval_dataset(FIXTURE)
    settings = Settings(
        _env_file=None,
        native_index_db=tmp_path / "native-index.db",
        native_chunk_size_chars=500,
        native_chunk_overlap_chars=50,
        native_search_top_k=5,
    )

    report = await evaluate_native_retrieval(
        dataset,
        settings=settings,
        work_dir=tmp_path / "evaluation",
        top_k=(1, 3, 5),
    )

    assert report.dataset_kind == DatasetKind.SMOKE_FIXTURE
    assert report.document_count == 3
    assert report.query_count == 5
    assert len(report.dataset_sha256) == 64
    assert report.index_profile == "native-fts5-bm25"
    assert report.chunk_size_chars == 500
    assert report.chunk_overlap_chars == 50
    assert 0.0 <= report.aggregate.recall_at_k["1"] <= 1.0
    assert report.aggregate.recall_at_k["5"] == 1.0
    assert 0.0 <= report.aggregate.mrr <= 1.0
    assert all(query.hits for query in report.queries)
    assert any(
        hit.matched_evidence_ids for query in report.queries for hit in query.hits
    )
