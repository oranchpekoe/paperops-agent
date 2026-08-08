"""Dependency-injected nodes for evidence-matrix construction and repair."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from paperops.clients.protocols import RetrievalBackend
from paperops.comparison.models import (
    ComparisonCell,
    ComparisonCellStatus,
    ComparisonDimension,
    ComparisonDocument,
    ComparisonEvent,
    ComparisonExtraction,
    ComparisonExtractionRequest,
    ComparisonFailure,
    ComparisonFailureCode,
    ComparisonSearchAttempt,
    ComparisonStatus,
    ComparisonStopReason,
)
from paperops.comparison.protocols import ComparisonModel
from paperops.comparison.state import ComparisonState
from paperops.models import SearchHit, SearchRequest
from paperops.research.evidence import append_single_inline_citation, merge_evidence
from paperops.research.models import EvidenceCitation
from paperops.settings import Settings


def _cell_key(document_id: str, dimension_id: str) -> str:
    return f"{document_id}:{dimension_id}"


def _normalized_query(value: str) -> str:
    return " ".join(value.casefold().split())


def _comparison_id(
    knowledge_base: str,
    documents: list[ComparisonDocument],
    dimensions: list[ComparisonDimension],
) -> str:
    identity = ":".join(
        [
            knowledge_base,
            *(document.document_id for document in documents),
            *(dimension.dimension_id for dimension in dimensions),
        ]
    )
    return f"comparison-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"


def _failure_update(
    stage: ComparisonStatus,
    code: ComparisonFailureCode,
    message: str,
    *,
    retryable: bool = False,
) -> dict[str, Any]:
    failure = ComparisonFailure(
        stage=stage,
        code=code,
        message=message,
        retryable=retryable,
    )
    return {
        "status": ComparisonStatus.FAILED,
        "failure": failure,
        "errors": [failure],
        "events": [
            ComparisonEvent(
                status=ComparisonStatus.FAILED,
                message=f"{stage.value} failed: {code.value}.",
            )
        ],
    }


def _default_gap_cell(
    document: ComparisonDocument,
    dimension: ComparisonDimension,
    reason: str,
) -> ComparisonCell:
    query = f"{document.label} {dimension.description}"[:500].strip()
    return ComparisonCell(
        document_id=document.document_id,
        dimension_id=dimension.dimension_id,
        status=ComparisonCellStatus.MISSING,
        confidence=1.0,
        missing_reason=reason,
        suggested_query=query,
    )


@dataclass(slots=True)
class ComparisonNodes:
    """Hold retrieval and semantic dependencies outside checkpoint state."""

    retrieval: RetrievalBackend
    model: ComparisonModel
    settings: Settings

    async def initialize(self, state: ComparisonState) -> dict[str, Any]:
        """Validate and normalize documents, dimensions, and budgets."""
        knowledge_base = state.get("knowledge_base", "").strip()
        try:
            documents = [
                ComparisonDocument.model_validate(document)
                for document in state.get("documents", [])
            ]
            dimensions = [
                ComparisonDimension.model_validate(dimension)
                for dimension in state.get("dimensions", [])
            ]
        except ValidationError as exc:
            return _failure_update(
                ComparisonStatus.PENDING,
                ComparisonFailureCode.INVALID_REQUEST,
                f"Comparison request failed validation: {exc}",
            )
        documents = [
            document.model_copy(
                update={
                    "document_id": document.document_id.strip(),
                    "label": document.label.strip(),
                }
            )
            for document in documents
        ]
        dimensions = [
            dimension.model_copy(update={"description": dimension.description.strip()})
            for dimension in dimensions
        ]
        document_ids = [document.document_id for document in documents]
        dimension_ids = [dimension.dimension_id for dimension in dimensions]
        invalid = (
            not knowledge_base
            or not 2 <= len(documents) <= self.settings.comparison_max_documents
            or not 1 <= len(dimensions) <= self.settings.comparison_max_dimensions
            or len(document_ids) != len(set(document_ids))
            or len(dimension_ids) != len(set(dimension_ids))
            or any(
                not document.document_id or not document.label for document in documents
            )
            or any(not dimension.description for dimension in dimensions)
        )
        if invalid:
            return _failure_update(
                ComparisonStatus.PENDING,
                ComparisonFailureCode.INVALID_REQUEST,
                "Comparison requires a knowledge base, "
                f"2-{self.settings.comparison_max_documents} unique documents, "
                f"and 1-{self.settings.comparison_max_dimensions} unique "
                "non-empty dimensions.",
            )
        return {
            "comparison_id": _comparison_id(
                knowledge_base,
                documents,
                dimensions,
            ),
            "knowledge_base": knowledge_base,
            "documents": documents,
            "dimensions": dimensions,
            "status": ComparisonStatus.RETRIEVING_INITIAL,
            "retrieval_round": 0,
            "gap_round": 0,
            "retrieval_calls": 0,
            "model_calls": 0,
            "new_evidence_count": 0,
            "attempted_searches": [],
            "evidence": [],
            "initial_cells": [],
            "cells": [],
            "recovered_cell_count": 0,
            "failure": None,
            "events": [
                ComparisonEvent(
                    status=ComparisonStatus.PENDING,
                    message=(
                        f"Validated {len(documents)} documents and "
                        f"{len(dimensions)} comparison dimensions."
                    ),
                )
            ],
        }

    async def retrieve_initial(self, state: ComparisonState) -> dict[str, Any]:
        """Retrieve every requested dimension inside its owning document."""
        attempts = [
            ComparisonSearchAttempt(
                document_id=document.document_id,
                dimension_id=dimension.dimension_id,
                query=dimension.description,
                retrieval_round=1,
            )
            for document in state["documents"]
            for dimension in state["dimensions"]
        ]
        return await self._retrieve_attempts(
            state,
            attempts,
            stage=ComparisonStatus.RETRIEVING_INITIAL,
            gap_round=0,
        )

    async def retrieve_gaps(self, state: ComparisonState) -> dict[str, Any]:
        """Run one focused search only for cells still missing evidence."""
        attempted = {
            (
                item.document_id,
                item.dimension_id,
                _normalized_query(item.query),
            )
            for item in state.get("attempted_searches", [])
        }
        retrieval_round = state.get("retrieval_round", 1) + 1
        attempts = []
        for cell in state.get("cells", []):
            if cell.status is not ComparisonCellStatus.MISSING:
                continue
            query = (cell.suggested_query or "").strip()
            key = (cell.document_id, cell.dimension_id, _normalized_query(query))
            if not query or key in attempted:
                continue
            attempts.append(
                ComparisonSearchAttempt(
                    document_id=cell.document_id,
                    dimension_id=cell.dimension_id,
                    query=query,
                    retrieval_round=retrieval_round,
                )
            )
        if not attempts:
            return {
                "status": ComparisonStatus.COMPLETED,
                "new_evidence_count": 0,
                "stop_reason": ComparisonStopReason.STAGNANT_RETRIEVAL,
                "events": [
                    ComparisonEvent(
                        status=ComparisonStatus.COMPLETED,
                        message="No novel gap query remained; stopped safely.",
                        retrieval_round=retrieval_round,
                    )
                ],
            }
        return await self._retrieve_attempts(
            state,
            attempts,
            stage=ComparisonStatus.RETRIEVING_GAPS,
            gap_round=state.get("gap_round", 0) + 1,
        )

    async def _retrieve_attempts(
        self,
        state: ComparisonState,
        attempts: list[ComparisonSearchAttempt],
        *,
        stage: ComparisonStatus,
        gap_round: int,
    ) -> dict[str, Any]:
        requests = [
            SearchRequest(
                knowledge_base=state["knowledge_base"],
                query=attempt.query,
                expected_document_id=attempt.document_id,
                top_k=self.settings.comparison_search_top_k,
            )
            for attempt in attempts
        ]
        try:
            results = await asyncio.gather(
                *(self.retrieval.search(request) for request in requests)
            )
        except Exception as exc:
            update = _failure_update(
                stage,
                ComparisonFailureCode.RETRIEVAL_ERROR,
                f"{type(exc).__name__}: {exc}",
                retryable=True,
            )
            update["retrieval_calls"] = state.get("retrieval_calls", 0) + len(attempts)
            return update

        evidence = state.get("evidence", [])
        previous_count = len(evidence)
        for attempt, hits in zip(attempts, results, strict=True):
            scoped_hits: list[SearchHit] = [
                hit for hit in hits if hit.document_id == attempt.document_id
            ]
            evidence = merge_evidence(
                evidence,
                scoped_hits,
                query=attempt.query,
                retrieval_round=attempt.retrieval_round,
                max_chunk_chars=self.settings.research_max_chunk_chars,
                max_evidence_chars=self.settings.comparison_max_evidence_chars,
            )
        new_evidence_count = len(evidence) - previous_count
        retrieval_calls = state.get("retrieval_calls", 0) + len(attempts)
        common: dict[str, Any] = {
            "retrieval_round": attempts[0].retrieval_round,
            "gap_round": gap_round,
            "retrieval_calls": retrieval_calls,
            "new_evidence_count": new_evidence_count,
            "attempted_searches": [*state.get("attempted_searches", []), *attempts],
            "evidence": evidence,
        }
        if stage is ComparisonStatus.RETRIEVING_GAPS and new_evidence_count == 0:
            return {
                **common,
                "status": ComparisonStatus.COMPLETED,
                "stop_reason": ComparisonStopReason.STAGNANT_RETRIEVAL,
                "events": [
                    ComparisonEvent(
                        status=ComparisonStatus.COMPLETED,
                        message=(
                            f"Gap retrieval ran {len(attempts)} scoped searches "
                            "but added no new evidence."
                        ),
                        retrieval_round=attempts[0].retrieval_round,
                    )
                ],
            }
        return {
            **common,
            "status": ComparisonStatus.EXTRACTING,
            "events": [
                ComparisonEvent(
                    status=stage,
                    message=(
                        f"Ran {len(attempts)} document-scoped searches and added "
                        f"{new_evidence_count} unique chunks."
                    ),
                    retrieval_round=attempts[0].retrieval_round,
                )
            ],
        }

    async def extract(self, state: ComparisonState) -> dict[str, Any]:
        """Fill all initial cells or re-extract only cells that were missing."""
        documents = {item.document_id: item for item in state["documents"]}
        dimensions = {item.dimension_id: item for item in state["dimensions"]}
        existing_cells = {
            _cell_key(cell.document_id, cell.dimension_id): cell
            for cell in state.get("cells", [])
        }
        if existing_cells:
            targets = [
                cell
                for cell in existing_cells.values()
                if cell.status is ComparisonCellStatus.MISSING
            ]
            target_ids_by_document: dict[str, list[str]] = {}
            for cell in targets:
                target_ids_by_document.setdefault(cell.document_id, []).append(
                    cell.dimension_id
                )
        else:
            target_ids_by_document = {
                document_id: list(dimensions) for document_id in documents
            }

        model_calls = state.get("model_calls", 0)
        extracted_cells: list[ComparisonCell] = []
        for document_id, dimension_ids in target_ids_by_document.items():
            document = documents[document_id]
            target_dimensions = [dimensions[item] for item in dimension_ids]
            document_evidence = [
                item
                for item in state.get("evidence", [])
                if item.document_id == document_id
            ]
            if not document_evidence:
                extracted_cells.extend(
                    _default_gap_cell(
                        document,
                        dimension,
                        "No evidence was retrieved from the selected document.",
                    )
                    for dimension in target_dimensions
                )
                continue
            try:
                extraction = ComparisonExtraction.model_validate(
                    await self.model.extract_comparison(
                        ComparisonExtractionRequest(
                            document=document,
                            dimensions=target_dimensions,
                            evidence=document_evidence,
                        )
                    )
                )
                model_calls += 1
            except ValidationError as exc:
                update = _failure_update(
                    ComparisonStatus.EXTRACTING,
                    ComparisonFailureCode.INVALID_MODEL_OUTPUT,
                    f"Comparison extraction failed validation: {exc}",
                )
                update["model_calls"] = model_calls + 1
                return update
            except Exception as exc:
                update = _failure_update(
                    ComparisonStatus.EXTRACTING,
                    ComparisonFailureCode.MODEL_ERROR,
                    f"{type(exc).__name__}: {exc}",
                    retryable=True,
                )
                update["model_calls"] = model_calls + 1
                return update

            validated = self._validate_extraction(
                extraction,
                document,
                target_dimensions,
                document_evidence,
            )
            if isinstance(validated, str):
                update = _failure_update(
                    ComparisonStatus.EXTRACTING,
                    ComparisonFailureCode.CITATION_VALIDATION_ERROR,
                    validated,
                )
                update["model_calls"] = model_calls
                return update
            extracted_cells.extend(validated)

        for cell in extracted_cells:
            existing_cells[_cell_key(cell.document_id, cell.dimension_id)] = cell
        cells = [
            existing_cells[_cell_key(document.document_id, dimension.dimension_id)]
            for document in state["documents"]
            for dimension in state["dimensions"]
        ]
        initial_cells = state.get("initial_cells", [])
        if not initial_cells:
            initial_cells = [cell.model_copy(deep=True) for cell in cells]
        initially_missing = {
            _cell_key(cell.document_id, cell.dimension_id)
            for cell in initial_cells
            if cell.status is ComparisonCellStatus.MISSING
        }
        recovered = sum(
            _cell_key(cell.document_id, cell.dimension_id) in initially_missing
            and cell.status is ComparisonCellStatus.SUPPORTED
            for cell in cells
        )
        missing_count = sum(
            cell.status is ComparisonCellStatus.MISSING for cell in cells
        )
        if missing_count == 0:
            status = ComparisonStatus.COMPLETED
            stop_reason = ComparisonStopReason.ALL_CELLS_SUPPORTED
        elif state.get("gap_round", 0) >= self.settings.comparison_max_gap_rounds:
            status = ComparisonStatus.COMPLETED
            stop_reason = ComparisonStopReason.GAP_BUDGET_EXHAUSTED
        else:
            status = ComparisonStatus.RETRIEVING_GAPS
            stop_reason = None
        return {
            "status": status,
            "model_calls": model_calls,
            "initial_cells": initial_cells,
            "cells": cells,
            "recovered_cell_count": recovered,
            **({"stop_reason": stop_reason} if stop_reason is not None else {}),
            "events": [
                ComparisonEvent(
                    status=(
                        ComparisonStatus.COMPLETED
                        if status is ComparisonStatus.COMPLETED
                        else ComparisonStatus.EXTRACTING
                    ),
                    message=(
                        f"Matrix contains {len(cells) - missing_count} supported "
                        f"and {missing_count} missing cells; recovered {recovered}."
                    ),
                    retrieval_round=state.get("retrieval_round") or None,
                )
            ],
        }

    def _validate_extraction(
        self,
        extraction: ComparisonExtraction,
        document: ComparisonDocument,
        dimensions: list[ComparisonDimension],
        evidence: list[EvidenceCitation],
    ) -> list[ComparisonCell] | str:
        expected_ids = [item.dimension_id for item in dimensions]
        actual_ids = [item.dimension_id for item in extraction.cells]
        if extraction.document_id != document.document_id:
            return "extraction document_id does not match the requested document"
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(
            expected_ids
        ):
            return "extraction must return exactly one cell per requested dimension"
        available = {item.citation_id for item in evidence}
        validated: list[ComparisonCell] = []
        by_dimension = {item.dimension_id: item for item in extraction.cells}
        dimension_map = {item.dimension_id: item for item in dimensions}
        for dimension_id in expected_ids:
            cell = by_dimension[dimension_id]
            if cell.document_id != document.document_id:
                return "comparison cell document_id does not match its extraction"
            if (
                cell.status is ComparisonCellStatus.SUPPORTED
                and cell.confidence < self.settings.comparison_min_cell_confidence
            ):
                validated.append(
                    _default_gap_cell(
                        document,
                        dimension_map[dimension_id],
                        "The extracted claim was below the configured confidence threshold.",
                    )
                )
                continue
            invalid = set(cell.citation_ids) - available
            missing_inline = [
                item
                for item in cell.citation_ids
                if cell.claim is not None and f"[{item}]" not in cell.claim
            ]
            if invalid:
                return f"unknown or cross-document citations: {sorted(invalid)}"
            if (
                cell.status is ComparisonCellStatus.SUPPORTED
                and len(cell.citation_ids) == 1
                and missing_inline == cell.citation_ids
            ):
                cell = cell.model_copy(
                    update={
                        "claim": append_single_inline_citation(
                            cell.claim or "",
                            cell.citation_ids[0],
                        )
                    }
                )
                missing_inline = []
            if missing_inline:
                return f"missing inline citation markers: {missing_inline}"
            validated.append(cell)
        return validated
