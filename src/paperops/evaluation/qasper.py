"""Convert official QASPER JSON into the PaperOps retrieval schema."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from paperops.evaluation.models import (
    DatasetKind,
    EvaluationDocument,
    EvaluationQuery,
    EvaluationSection,
    EvidenceReference,
    RetrievalDataset,
)

QASPER_SOURCE_URL = "https://huggingface.co/datasets/allenai/qasper"
QASPER_LICENSE = "CC-BY-4.0"
_SAFE_ID_RE = re.compile(r"[^0-9A-Za-z_.-]+")


def _safe_id(value: object, fallback: str) -> str:
    normalized = _SAFE_ID_RE.sub("-", str(value)).strip("-.")
    return normalized or fallback


def _text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n\n".join(str(item).strip() for item in value if str(item).strip())
    return ""


def _sections(raw_full_text: object) -> list[EvaluationSection]:
    sections: list[EvaluationSection] = []
    if isinstance(raw_full_text, list):
        entries = [entry for entry in raw_full_text if isinstance(entry, dict)]
    elif isinstance(raw_full_text, dict):
        names = raw_full_text.get("section_name", [])
        paragraphs = raw_full_text.get("paragraphs", [])
        if not isinstance(names, list) or not isinstance(paragraphs, list):
            return []
        entries = [
            {"section_name": name, "paragraphs": section_paragraphs}
            for name, section_paragraphs in zip(names, paragraphs, strict=False)
        ]
    else:
        return []

    for index, entry in enumerate(entries, start=1):
        raw_paragraphs = entry.get("paragraphs", [])
        if not isinstance(raw_paragraphs, list):
            continue
        paragraphs = [_text(paragraph) for paragraph in raw_paragraphs]
        paragraphs = [paragraph for paragraph in paragraphs if paragraph]
        if not paragraphs:
            continue
        sections.append(
            EvaluationSection(
                title=_text(entry.get("section_name")) or f"Section {index}",
                paragraphs=paragraphs,
            )
        )
    return sections


def _qas(raw_qas: object) -> list[dict[str, Any]]:
    if isinstance(raw_qas, list):
        return [item for item in raw_qas if isinstance(item, dict)]
    if not isinstance(raw_qas, dict):
        return []
    questions = raw_qas.get("question", [])
    question_ids = raw_qas.get("question_id", [])
    answers = raw_qas.get("answers", [])
    if not all(isinstance(items, list) for items in (questions, question_ids, answers)):
        return []
    return [
        {"question": question, "question_id": question_id, "answers": answer}
        for question, question_id, answer in zip(
            questions,
            question_ids,
            answers,
            strict=False,
        )
    ]


def _answer_payload(annotation: object) -> dict[str, Any] | None:
    if not isinstance(annotation, dict):
        return None
    answer = annotation.get("answer", annotation)
    return answer if isinstance(answer, dict) else None


def _evidence_counts(raw_answers: object, document_text: str) -> Counter[str]:
    if not isinstance(raw_answers, list):
        return Counter()
    normalized_document = " ".join(document_text.split())
    evidence_counts: Counter[str] = Counter()
    for annotation in raw_answers:
        answer = _answer_payload(annotation)
        if answer is None or answer.get("unanswerable") is True:
            continue
        evidence_values = answer.get("evidence", [])
        if not isinstance(evidence_values, list):
            continue
        seen_for_annotation: set[str] = set()
        for value in evidence_values:
            evidence = _text(value)
            if not evidence or evidence.startswith("FLOAT SELECTED"):
                continue
            normalized = " ".join(evidence.split())
            if (
                normalized not in normalized_document
                or normalized in seen_for_annotation
            ):
                continue
            seen_for_annotation.add(normalized)
            evidence_counts[normalized] += 1
    return evidence_counts


def _unanimously_unanswerable(raw_answers: object) -> bool:
    """Accept a refusal label only when every valid annotation agrees."""
    if not isinstance(raw_answers, list):
        return False
    answers = [
        answer
        for annotation in raw_answers
        if (answer := _answer_payload(annotation)) is not None
    ]
    return bool(answers) and all(
        answer.get("unanswerable") is True for answer in answers
    )


def _selection_complete(
    *,
    total: int,
    answerable: int,
    unanswerable: int,
    max_queries: int | None,
    max_answerable_queries: int | None,
    max_unanswerable_queries: int | None,
) -> bool:
    if max_queries is not None and total >= max_queries:
        return True
    quota_checks = [
        count >= limit
        for count, limit in (
            (answerable, max_answerable_queries),
            (unanswerable, max_unanswerable_queries),
        )
        if limit is not None
    ]
    return bool(quota_checks) and all(quota_checks)


def convert_qasper(
    source: Path,
    *,
    split: str,
    max_documents: int | None = None,
    max_queries: int | None = None,
    max_answerable_queries: int | None = None,
    max_unanswerable_queries: int | None = None,
    include_unanswerable: bool = False,
) -> RetrievalDataset:
    """Convert one downloaded QASPER split without fetching data implicitly."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        papers = list(payload.items())
    elif isinstance(payload, list):
        papers = [
            (str(paper.get("id", index)), paper)
            for index, paper in enumerate(payload)
            if isinstance(paper, dict)
        ]
    else:
        raise ValueError("QASPER source must contain an object or list of papers")

    documents: list[EvaluationDocument] = []
    queries: list[EvaluationQuery] = []
    answerable_count = 0
    unanswerable_count = 0
    for paper_index, (raw_paper_id, raw_paper) in enumerate(papers, start=1):
        if max_documents is not None and len(documents) >= max_documents:
            break
        if _selection_complete(
            total=len(queries),
            answerable=answerable_count,
            unanswerable=unanswerable_count,
            max_queries=max_queries,
            max_answerable_queries=max_answerable_queries,
            max_unanswerable_queries=max_unanswerable_queries,
        ):
            break
        if not isinstance(raw_paper, dict):
            continue
        sections = _sections(raw_paper.get("full_text"))
        if not sections:
            continue
        document_id = _safe_id(raw_paper_id, f"paper-{paper_index}")
        document = EvaluationDocument(
            document_id=document_id,
            title=_text(raw_paper.get("title")) or document_id,
            abstract=_text(raw_paper.get("abstract")),
            sections=sections,
        )
        document_text = document.to_markdown()
        paper_queries: list[EvaluationQuery] = []
        for question_index, raw_query in enumerate(
            _qas(raw_paper.get("qas")),
            start=1,
        ):
            if _selection_complete(
                total=len(queries) + len(paper_queries),
                answerable=answerable_count,
                unanswerable=unanswerable_count,
                max_queries=max_queries,
                max_answerable_queries=max_answerable_queries,
                max_unanswerable_queries=max_unanswerable_queries,
            ):
                break
            question = _text(raw_query.get("question"))
            if not question:
                continue
            query_id = _safe_id(
                raw_query.get("question_id"),
                f"{document_id}-question-{question_index}",
            )
            counts = _evidence_counts(raw_query.get("answers"), document_text)
            evidence = [
                EvidenceReference(
                    evidence_id=(
                        f"{query_id}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"
                    ),
                    document_id=document_id,
                    text=text,
                    relevance=min(count, 3),
                )
                for text, count in sorted(counts.items())
            ]
            if evidence and (
                max_answerable_queries is None
                or answerable_count < max_answerable_queries
            ):
                paper_queries.append(
                    EvaluationQuery(
                        query_id=query_id,
                        text=question,
                        evidence=evidence,
                    )
                )
                answerable_count += 1
            elif (
                include_unanswerable
                and _unanimously_unanswerable(raw_query.get("answers"))
                and (
                    max_unanswerable_queries is None
                    or unanswerable_count < max_unanswerable_queries
                )
            ):
                paper_queries.append(
                    EvaluationQuery(
                        query_id=query_id,
                        text=question,
                        answerable=False,
                    )
                )
                unanswerable_count += 1
        if paper_queries:
            documents.append(document)
            queries.extend(paper_queries)

    if not documents or not queries:
        raise ValueError("No eligible labelled queries were found in the input")
    return RetrievalDataset(
        name="qasper",
        version=(
            "0.3-paperops-v2-with-unanswerable"
            if include_unanswerable
            else "0.3-paperops-v1"
        ),
        kind=DatasetKind.BENCHMARK,
        split=split,
        source_url=QASPER_SOURCE_URL,
        license=QASPER_LICENSE,
        documents=documents,
        queries=queries,
    )


def write_retrieval_dataset(dataset: RetrievalDataset, destination: Path) -> None:
    """Persist converted data separately from the downloaded source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
