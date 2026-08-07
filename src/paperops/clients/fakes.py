"""Deterministic fake service clients for PR2 development and tests."""

from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path

from paperops.models import (
    IngestRequest,
    IngestResult,
    ParseRequest,
    ParseResult,
    SearchHit,
    SearchRequest,
)


def _default_markdown(request: ParseRequest) -> str:
    """Return a valid deterministic artifact for interactive local runs."""
    title = Path(request.source_pdf).stem.replace("_", " ").strip() or "Research paper"
    return (
        f"# {title}\n\n"
        "## Abstract\n\n"
        "This deterministic Fake Parser artifact exercises the PaperOps workflow "
        "without claiming to parse the source PDF. It is used only in PR2 tests "
        "and local workflow demonstrations.\n\n"
        "## Method\n\n"
        "The document is validated, ingested with an idempotency key, and then "
        "queried to verify that supporting evidence can be retrieved.\n"
    )


class FakeParserClient:
    """Write scripted Markdown outcomes to deterministic artifact paths."""

    def __init__(
        self,
        artifacts_dir: Path,
        outcomes: list[str | Exception] | None = None,
    ) -> None:
        """Configure the artifact directory and optional scripted outcomes."""
        self.artifacts_dir = artifacts_dir
        self._outcomes: deque[str | Exception] = deque(outcomes or [])
        self.calls: list[ParseRequest] = []
        self.created_artifacts = 0

    async def parse(self, request: ParseRequest) -> ParseResult:
        """Create one Markdown artifact per job attempt, or reuse it on replay."""
        self.calls.append(request)
        markdown_path = (
            self.artifacts_dir / request.job_id / f"parse-attempt-{request.attempt}.md"
        )

        if markdown_path.is_file():
            return ParseResult(
                markdown_path=str(markdown_path),
                idempotency_key=request.idempotency_key,
                created=False,
            )

        outcome = (
            self._outcomes.popleft() if self._outcomes else _default_markdown(request)
        )
        if isinstance(outcome, Exception):
            raise outcome

        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = markdown_path.with_suffix(".md.tmp")
        temporary_path.write_text(outcome, encoding="utf-8")
        temporary_path.replace(markdown_path)
        self.created_artifacts += 1
        return ParseResult(
            markdown_path=str(markdown_path),
            idempotency_key=request.idempotency_key,
            created=True,
        )


class FakeKnowledgeBaseClient:
    """Provide idempotent in-memory ingestion and deterministic retrieval."""

    def __init__(
        self,
        *,
        fail_ingest_times: int = 0,
        fail_search_times: int = 0,
        return_hits: bool = True,
    ) -> None:
        """Configure deterministic failure injection and retrieval behavior."""
        self.fail_ingest_times = fail_ingest_times
        self.fail_search_times = fail_search_times
        self.return_hits = return_hits
        self.ingest_calls: list[IngestRequest] = []
        self.search_calls: list[SearchRequest] = []
        self._results_by_key: dict[str, IngestResult] = {}
        self._documents: dict[str, tuple[str, str]] = {}
        self.created_documents = 0

    async def ingest(self, request: IngestRequest) -> IngestResult:
        """Create one deterministic document for each idempotency key."""
        self.ingest_calls.append(request)
        existing = self._results_by_key.get(request.idempotency_key)
        if existing is not None:
            return existing.model_copy(update={"created": False})

        if self.fail_ingest_times > 0:
            self.fail_ingest_times -= 1
            raise RuntimeError("scripted knowledge-base ingestion failure")

        markdown_path = Path(request.markdown_path)
        content = markdown_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(request.idempotency_key.encode("utf-8")).hexdigest()
        document_id = f"doc-{digest[:16]}"
        result = IngestResult(
            document_id=document_id,
            idempotency_key=request.idempotency_key,
            created=True,
        )
        self._results_by_key[request.idempotency_key] = result
        self._documents[document_id] = (request.knowledge_base, content)
        self.created_documents += 1
        return result

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        """Return the expected document when it exists and retrieval is enabled."""
        self.search_calls.append(request)
        if self.fail_search_times > 0:
            self.fail_search_times -= 1
            raise RuntimeError("scripted knowledge-base search failure")
        if not self.return_hits:
            return []

        stored = self._documents.get(request.expected_document_id)
        if stored is None:
            return []
        knowledge_base, content = stored
        if knowledge_base != request.knowledge_base:
            return []
        return [
            SearchHit(
                document_id=request.expected_document_id,
                content=content[:240],
                score=1.0,
            )
        ]
