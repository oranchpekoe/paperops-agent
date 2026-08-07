"""External service boundaries and deterministic PR2 fakes."""

from paperops.clients.fakes import FakeKnowledgeBaseClient, FakeParserClient
from paperops.clients.protocols import KnowledgeBaseClient, ParserClient

__all__ = [
    "FakeKnowledgeBaseClient",
    "FakeParserClient",
    "KnowledgeBaseClient",
    "ParserClient",
]
