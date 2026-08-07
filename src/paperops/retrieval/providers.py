"""Optional local model providers for dense retrieval and reranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    """Generate compatible document and query vectors."""

    name: str

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Encode passages in their retrieval role."""

    def embed_query(self, query: str) -> list[float]:
        """Encode one search query."""


class Reranker(Protocol):
    """Score query-passage pairs with a cross encoder."""

    name: str

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document."""


def _fastembed_import_error() -> RuntimeError:
    return RuntimeError(
        "Local retrieval models are optional. Install them with "
        "`uv sync --extra retrieval-models`."
    )


class FastEmbedProvider:
    """Run a bi-encoder locally through ONNX Runtime."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        threads: int | None = None,
    ) -> None:
        """Load one supported FastEmbed model, downloading it if absent."""
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise _fastembed_import_error() from exc
        self.model_name = model_name
        self.name = f"fastembed:{model_name}"
        self._model: Any = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
            threads=threads,
        )

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Encode passages with the model's document path."""
        return [vector.tolist() for vector in self._model.embed(documents)]

    def embed_query(self, query: str) -> list[float]:
        """Encode a query with any model-specific query prefixing."""
        vectors = list(self._model.query_embed(query))
        if len(vectors) != 1:
            raise ValueError("embedding provider returned an unexpected query count")
        return vectors[0].tolist()


class FastEmbedReranker:
    """Run a cross encoder locally through ONNX Runtime."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        threads: int | None = None,
    ) -> None:
        """Load one supported FastEmbed cross encoder."""
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise _fastembed_import_error() from exc
        self.model_name = model_name
        self.name = f"fastembed-reranker:{model_name}"
        self._model: Any = TextCrossEncoder(
            model_name=model_name,
            cache_dir=str(cache_dir),
            threads=threads,
        )

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score all candidates without changing their text."""
        return [float(score) for score in self._model.rerank(query, documents)]
