"""Service boundaries consumed by PaperOps workflow nodes."""

from typing import Protocol

from paperops.models import (
    IngestRequest,
    IngestResult,
    ParseRequest,
    ParseResult,
    SearchHit,
    SearchRequest,
)


class ParserClient(Protocol):
    """Parse a source PDF into an artifact referenced by path."""

    async def parse(self, request: ParseRequest) -> ParseResult:
        """Return an idempotent parse result."""


class RetrievalBackend(Protocol):
    """Own the indexing and retrieval boundary used by the workflow."""

    name: str

    async def ingest(self, request: IngestRequest) -> IngestResult:
        """Create or reuse a knowledge-base document."""

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        """Return evidence snippets for retrieval verification."""


# Kept as a compatibility alias for the PR2 public boundary. New code should use
# RetrievalBackend because an implementation need not be a hosted knowledge base.
KnowledgeBaseClient = RetrievalBackend
