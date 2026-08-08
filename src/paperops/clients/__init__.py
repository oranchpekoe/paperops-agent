"""External service boundaries, concrete adapters, and deterministic fakes."""

from paperops.clients.errors import (
    ExternalServiceError,
    ExternalServiceTimeout,
    MinerUError,
    MinerUTimeout,
    RAGFlowError,
    RAGFlowTimeout,
    ResearchModelError,
)
from paperops.clients.fakes import FakeKnowledgeBaseClient, FakeParserClient
from paperops.clients.protocols import (
    KnowledgeBaseClient,
    ParserClient,
    RetrievalBackend,
)

__all__ = [
    "ExternalServiceError",
    "ExternalServiceTimeout",
    "FakeKnowledgeBaseClient",
    "FakeParserClient",
    "KnowledgeBaseClient",
    "MinerUError",
    "MinerUTimeout",
    "ParserClient",
    "RetrievalBackend",
    "RAGFlowError",
    "RAGFlowTimeout",
    "ResearchModelError",
]
