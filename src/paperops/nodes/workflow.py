"""Dependency-injected nodes for the single-document PaperOps workflow."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.types import interrupt
from pydantic import ValidationError

from paperops.clients.protocols import ParserClient, RetrievalBackend
from paperops.models import (
    ApprovalAction,
    ApprovalDecision,
    FailureCode,
    IngestRequest,
    JobStatus,
    ParseRequest,
    QualityDecision,
    QualityVerdict,
    RetrievalReport,
    SearchRequest,
    WorkflowEvent,
    WorkflowFailure,
)
from paperops.quality.rules import QualityPolicy, evaluate_markdown
from paperops.retrieval.chunking import build_index_probe
from paperops.settings import Settings
from paperops.state import DocumentJobState


def _hash_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the PDF into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_and_hash_pdf(path: Path) -> str:
    """Validate and hash a source PDF outside the event loop."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return _hash_file(path)


def _write_text_atomically(path: Path, content: str) -> None:
    """Write a UTF-8 artifact atomically outside the event loop."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _job_id(file_hash: str, knowledge_base: str) -> str:
    """Derive a stable job identity from the source and destination."""
    identity = f"{file_hash}:{knowledge_base}".encode()
    return f"job-{hashlib.sha256(identity).hexdigest()[:16]}"


def _failure_update(
    failure: WorkflowFailure,
    message: str,
    *,
    attempt: int | None = None,
) -> dict[str, Any]:
    """Return a terminal update with structured failure and audit entries."""
    return {
        "status": JobStatus.FAILED,
        "failure": failure,
        "errors": [failure],
        "events": [
            WorkflowEvent(
                status=JobStatus.FAILED,
                message=message,
                attempt=attempt,
            )
        ],
    }


@dataclass(slots=True)
class WorkflowNodes:
    """Hold workflow dependencies while keeping graph state serialisable."""

    parser: ParserClient
    knowledge_base: RetrievalBackend
    settings: Settings
    quality_policy: QualityPolicy

    async def initialize(self, state: DocumentJobState) -> dict[str, Any]:
        """Validate input and derive deterministic file and job identities."""
        source_pdf = Path(state.get("source_pdf", ""))
        knowledge_base = state.get("target_knowledge_base", "").strip()
        if source_pdf.suffix.lower() != ".pdf":
            failure = WorkflowFailure(
                stage=JobStatus.PENDING,
                code=FailureCode.INVALID_SOURCE,
                message=f"Source PDF does not exist or is not a PDF: {source_pdf}",
            )
            return _failure_update(failure, "Source validation failed.")
        if not knowledge_base:
            failure = WorkflowFailure(
                stage=JobStatus.PENDING,
                code=FailureCode.INVALID_SOURCE,
                message="A target knowledge base is required.",
            )
            return _failure_update(failure, "Destination validation failed.")

        try:
            file_hash = await asyncio.to_thread(_validate_and_hash_pdf, source_pdf)
        except OSError:
            failure = WorkflowFailure(
                stage=JobStatus.PENDING,
                code=FailureCode.INVALID_SOURCE,
                message=f"Source PDF does not exist or cannot be read: {source_pdf}",
            )
            return _failure_update(failure, "Source validation failed.")
        job_id = _job_id(file_hash, knowledge_base)
        return {
            "job_id": job_id,
            "file_hash": file_hash,
            "status": JobStatus.PENDING,
            "parse_attempts": 0,
            "events": [
                WorkflowEvent(
                    status=JobStatus.PENDING,
                    message="Source and destination validated.",
                )
            ],
        }

    async def parse_document(self, state: DocumentJobState) -> dict[str, Any]:
        """Parse a PDF using one idempotency key per intentional attempt."""
        attempt = state.get("parse_attempts", 0) + 1
        request = ParseRequest(
            job_id=state["job_id"],
            source_pdf=state["source_pdf"],
            file_hash=state["file_hash"],
            attempt=attempt,
            idempotency_key=f"parse:{state['job_id']}:{attempt}",
        )
        try:
            result = await self.parser.parse(request)
        except Exception as exc:
            failure = WorkflowFailure(
                stage=JobStatus.PARSING,
                code=FailureCode.PARSER_ERROR,
                message=f"{type(exc).__name__}: {exc}",
                retryable=attempt < self.settings.max_parse_attempts,
            )
            return {
                "status": JobStatus.QUALITY_CHECK,
                "parse_attempts": attempt,
                "parsed_markdown_path": "",
                "errors": [failure],
                "events": [
                    WorkflowEvent(
                        status=JobStatus.PARSING,
                        message="Parser call failed; quality routing will decide retry.",
                        attempt=attempt,
                    )
                ],
            }

        return {
            "status": JobStatus.QUALITY_CHECK,
            "parse_attempts": attempt,
            "parsed_markdown_path": result.markdown_path,
            "events": [
                WorkflowEvent(
                    status=JobStatus.PARSING,
                    message=(
                        "Parser artifact created."
                        if result.created
                        else "Existing parser artifact reused."
                    ),
                    attempt=attempt,
                )
            ],
        }

    async def quality_check(self, state: DocumentJobState) -> dict[str, Any]:
        """Apply deterministic rules to the referenced Markdown artifact."""
        markdown_path = Path(state.get("parsed_markdown_path", ""))
        try:
            decision = await asyncio.to_thread(
                evaluate_markdown,
                markdown_path,
                self.quality_policy,
            )
            errors: list[WorkflowFailure] = []
        except Exception as exc:
            failure = WorkflowFailure(
                stage=JobStatus.QUALITY_CHECK,
                code=FailureCode.QUALITY_CHECK_ERROR,
                message=f"{type(exc).__name__}: {exc}",
                retryable=state.get("parse_attempts", 0)
                < self.settings.max_parse_attempts,
            )
            errors = [failure]
            decision = QualityDecision(
                verdict=QualityVerdict.RETRY,
                confidence=1.0,
                issues=["quality_check_error"],
                retry_reason="quality_check_error",
            )

        return {
            "status": JobStatus.QUALITY_CHECK,
            "quality_decision": decision,
            "errors": errors,
            "events": [
                WorkflowEvent(
                    status=JobStatus.QUALITY_CHECK,
                    message=f"Quality verdict: {decision.verdict}.",
                    attempt=state.get("parse_attempts"),
                )
            ],
        }

    async def mark_waiting_approval(self, state: DocumentJobState) -> dict[str, Any]:
        """Persist the waiting state before entering a human interrupt."""
        return {
            "status": JobStatus.WAITING_APPROVAL,
            "events": [
                WorkflowEvent(
                    status=JobStatus.WAITING_APPROVAL,
                    message="Human review is required before indexing.",
                    attempt=state.get("parse_attempts"),
                )
            ],
        }

    def request_approval(self, state: DocumentJobState) -> dict[str, Any]:
        """Pause for a validated approve or reject decision."""
        raw_decision = interrupt(
            {
                "job_id": state["job_id"],
                "artifact_path": state.get("parsed_markdown_path", ""),
                "quality_decision": state["quality_decision"].model_dump(mode="json"),
                "allowed_actions": [
                    ApprovalAction.APPROVE.value,
                    ApprovalAction.REJECT.value,
                ],
            }
        )
        try:
            decision = ApprovalDecision.model_validate(raw_decision)
        except ValidationError as exc:
            failure = WorkflowFailure(
                stage=JobStatus.WAITING_APPROVAL,
                code=FailureCode.INVALID_APPROVAL,
                message=f"Approval payload failed validation: {exc.errors(include_url=False)}",
            )
            return _failure_update(failure, "Invalid human approval payload.")

        if decision.action == ApprovalAction.REJECT:
            failure = WorkflowFailure(
                stage=JobStatus.WAITING_APPROVAL,
                code=FailureCode.APPROVAL_REJECTED,
                message=decision.note or "The reviewer rejected the parsed artifact.",
            )
            update = _failure_update(failure, "Reviewer rejected the artifact.")
            update["approval_decision"] = decision
            return update

        return {
            "status": JobStatus.INDEXING,
            "approval_decision": decision,
            "events": [
                WorkflowEvent(
                    status=JobStatus.INDEXING,
                    message="Reviewer approved the artifact for indexing.",
                )
            ],
        }

    async def fail_quality(self, state: DocumentJobState) -> dict[str, Any]:
        """Fail closed after deterministic retry capacity is exhausted."""
        decision = state["quality_decision"]
        issues = ", ".join(decision.issues) or "unknown quality issue"
        failure = WorkflowFailure(
            stage=JobStatus.QUALITY_CHECK,
            code=FailureCode.QUALITY_RETRIES_EXHAUSTED,
            message=(
                f"Quality gate still failed after {state.get('parse_attempts', 0)} "
                f"attempt(s): {issues}"
            ),
        )
        return _failure_update(
            failure,
            "Automatic parse retries were exhausted.",
            attempt=state.get("parse_attempts"),
        )

    async def ingest_document(self, state: DocumentJobState) -> dict[str, Any]:
        """Ingest an approved artifact through an idempotent client request."""
        request = IngestRequest(
            job_id=state["job_id"],
            knowledge_base=state["target_knowledge_base"],
            file_hash=state["file_hash"],
            markdown_path=state["parsed_markdown_path"],
            idempotency_key=(
                f"ingest:{state['target_knowledge_base']}:{state['file_hash']}"
            ),
        )
        try:
            result = await self.knowledge_base.ingest(request)
        except Exception as exc:
            failure = WorkflowFailure(
                stage=JobStatus.INDEXING,
                code=FailureCode.INDEX_ERROR,
                message=f"{type(exc).__name__}: {exc}",
            )
            return _failure_update(failure, "Document indexing failed.")

        return {
            "status": JobStatus.RETRIEVAL_EVAL,
            "indexed_document_id": result.document_id,
            "indexed_chunk_count": result.chunk_count,
            "events": [
                WorkflowEvent(
                    status=JobStatus.INDEXING,
                    message=(
                        "Indexed document created."
                        if result.created
                        else "Existing indexed document reused."
                    ),
                )
            ],
        }

    async def evaluate_retrieval(self, state: DocumentJobState) -> dict[str, Any]:
        """Verify that the expected document can supply retrieval evidence."""
        try:
            markdown = await asyncio.to_thread(
                Path(state["parsed_markdown_path"]).read_text,
                encoding="utf-8",
            )
            source_name = Path(state["source_pdf"]).stem.replace("_", " ")
            query = build_index_probe(markdown, source_name)
            request = SearchRequest(
                knowledge_base=state["target_knowledge_base"],
                query=query,
                expected_document_id=state["indexed_document_id"],
                top_k=self.settings.retrieval_probe_top_k,
            )
            hits = await self.knowledge_base.search(request)
        except Exception as exc:
            failure = WorkflowFailure(
                stage=JobStatus.RETRIEVAL_EVAL,
                code=FailureCode.RETRIEVAL_ERROR,
                message=f"{type(exc).__name__}: {exc}",
            )
            return _failure_update(failure, "Retrieval verification failed to run.")

        matching_hits = [
            hit for hit in hits if hit.document_id == state["indexed_document_id"]
        ]
        passed = len(matching_hits) >= self.settings.min_retrieval_hits
        report = RetrievalReport(
            passed=passed,
            query=query,
            document_id=state["indexed_document_id"],
            hit_count=len(matching_hits),
            backend=self.knowledge_base.name,
            strategy="document_scoped_index_probe",
            evidence=[hit.content for hit in matching_hits[:3]],
        )
        report_path = (
            self.settings.artifacts_dir / state["job_id"] / "retrieval-evaluation.json"
        )
        await asyncio.to_thread(
            _write_text_atomically,
            report_path,
            report.model_dump_json(indent=2),
        )

        if not passed:
            failure = WorkflowFailure(
                stage=JobStatus.RETRIEVAL_EVAL,
                code=FailureCode.RETRIEVAL_FAILED,
                message="The expected document was not present in retrieval results.",
            )
            update = _failure_update(failure, "Retrieval evidence did not meet policy.")
            update.update(
                {
                    "retrieval_report": report,
                    "evaluation_report_path": str(report_path),
                }
            )
            return update

        return {
            "status": JobStatus.COMPLETED,
            "retrieval_report": report,
            "evaluation_report_path": str(report_path),
            "events": [
                WorkflowEvent(
                    status=JobStatus.COMPLETED,
                    message="Document indexing and retrieval probe completed.",
                )
            ],
        }
