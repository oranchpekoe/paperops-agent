"""Document chunking and local retrieval implementations."""

from paperops.retrieval.chunking import MarkdownChunk, build_index_probe, chunk_markdown
from paperops.retrieval.dense import DenseRetrievalBackend
from paperops.retrieval.hybrid import HybridRetrievalBackend, RerankedRetrievalBackend
from paperops.retrieval.native import NativeRetrievalBackend
from paperops.retrieval.providers import FastEmbedProvider, FastEmbedReranker

__all__ = [
    "DenseRetrievalBackend",
    "FastEmbedProvider",
    "FastEmbedReranker",
    "HybridRetrievalBackend",
    "MarkdownChunk",
    "NativeRetrievalBackend",
    "RerankedRetrievalBackend",
    "build_index_probe",
    "chunk_markdown",
]
