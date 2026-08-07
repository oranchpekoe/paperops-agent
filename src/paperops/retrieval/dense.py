"""Local dense retrieval backed by FastEmbed-compatible vectors and sqlite-vec."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from threading import Lock

import sqlite_vec  # type: ignore[import-untyped]

from paperops.models import IngestRequest, IngestResult, SearchHit, SearchRequest
from paperops.retrieval.chunking import chunk_markdown
from paperops.retrieval.native import stable_chunk_id, stable_document_id
from paperops.retrieval.providers import EmbeddingProvider
from paperops.settings import Settings


class DenseRetrievalBackend:
    """Persist normalized dense vectors in a filtered sqlite-vec index."""

    def __init__(self, settings: Settings, provider: EmbeddingProvider) -> None:
        """Configure the shared chunk policy and embedding provider."""
        if settings.native_chunk_overlap_chars >= settings.native_chunk_size_chars:
            raise ValueError(
                "PAPEROPS_NATIVE_CHUNK_OVERLAP_CHARS must be smaller than "
                "PAPEROPS_NATIVE_CHUNK_SIZE_CHARS"
            )
        self.settings = settings
        self.provider = provider
        self.database_path = settings.native_index_db
        self.name = f"dense_sqlite_vec:{provider.name}"
        self._write_lock = Lock()
        self._model_lock = Lock()

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS dense_documents (
                document_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                knowledge_base TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                markdown_path TEXT NOT NULL,
                chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS dense_index_config (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                model_name TEXT NOT NULL,
                dimension INTEGER NOT NULL CHECK (dimension > 0)
            );
            """
        )
        return connection

    def _ensure_vector_schema(
        self,
        connection: sqlite3.Connection,
        dimension: int,
    ) -> None:
        existing = connection.execute(
            "SELECT model_name, dimension FROM dense_index_config WHERE singleton = 1"
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO dense_index_config (singleton, model_name, dimension)
                VALUES (1, ?, ?)
                """,
                (self.provider.name, dimension),
            )
        elif (
            str(existing["model_name"]) != self.provider.name
            or int(existing["dimension"]) != dimension
        ):
            raise ValueError(
                "Dense index configuration does not match the selected embedding "
                "model; use a new index database or rebuild the existing index"
            )
        connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS dense_chunks USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding float[{dimension}],
                knowledge_base TEXT PARTITION KEY,
                document_id TEXT,
                +heading_path TEXT,
                +content TEXT
            )
            """
        )

    async def ingest(self, request: IngestRequest) -> IngestResult:
        """Chunk, embed, and atomically persist one document."""
        return await asyncio.to_thread(self._ingest_sync, request)

    def _ingest_sync(self, request: IngestRequest) -> IngestResult:
        markdown_path = Path(request.markdown_path)
        markdown = markdown_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(
            markdown,
            max_chars=self.settings.native_chunk_size_chars,
            overlap_chars=self.settings.native_chunk_overlap_chars,
        )
        if not chunks:
            raise ValueError("The parsed Markdown did not produce any indexable chunks")
        document_id = stable_document_id(request.idempotency_key)

        with self._write_lock, self._connect() as connection:
            existing = connection.execute(
                """
                SELECT document_id, chunk_count
                FROM dense_documents
                WHERE idempotency_key = ?
                """,
                (request.idempotency_key,),
            ).fetchone()
            if existing is not None:
                return IngestResult(
                    document_id=str(existing["document_id"]),
                    idempotency_key=request.idempotency_key,
                    created=False,
                    chunk_count=int(existing["chunk_count"]),
                )

        with self._model_lock:
            embeddings = self.provider.embed_documents(
                [chunk.content for chunk in chunks]
            )
        if len(embeddings) != len(chunks) or not embeddings or not embeddings[0]:
            raise ValueError("embedding provider returned an invalid document batch")
        dimension = len(embeddings[0])
        if any(len(vector) != dimension for vector in embeddings):
            raise ValueError("embedding provider returned inconsistent dimensions")

        with self._write_lock, self._connect() as connection:
            existing = connection.execute(
                """
                SELECT document_id, chunk_count
                FROM dense_documents
                WHERE idempotency_key = ?
                """,
                (request.idempotency_key,),
            ).fetchone()
            if existing is not None:
                return IngestResult(
                    document_id=str(existing["document_id"]),
                    idempotency_key=request.idempotency_key,
                    created=False,
                    chunk_count=int(existing["chunk_count"]),
                )
            self._ensure_vector_schema(connection, dimension)
            connection.execute(
                """
                INSERT INTO dense_documents (
                    document_id,
                    idempotency_key,
                    knowledge_base,
                    file_hash,
                    markdown_path,
                    chunk_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    request.idempotency_key,
                    request.knowledge_base,
                    request.file_hash,
                    str(markdown_path),
                    len(chunks),
                ),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                connection.execute(
                    """
                    INSERT INTO dense_chunks (
                        chunk_id,
                        embedding,
                        knowledge_base,
                        document_id,
                        heading_path,
                        content
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_chunk_id(document_id, chunk.ordinal, chunk.content),
                        sqlite_vec.serialize_float32(embedding),
                        request.knowledge_base,
                        document_id,
                        "\n".join(chunk.heading_path),
                        chunk.content,
                    ),
                )
        return IngestResult(
            document_id=document_id,
            idempotency_key=request.idempotency_key,
            created=True,
            chunk_count=len(chunks),
        )

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        """Embed one query and run filtered exact KNN search."""
        return await asyncio.to_thread(self._search_sync, request)

    def _search_sync(self, request: SearchRequest) -> list[SearchHit]:
        with self._connect() as connection:
            configured = connection.execute(
                "SELECT model_name, dimension FROM dense_index_config WHERE singleton = 1"
            ).fetchone()
        if configured is None:
            return []
        if str(configured["model_name"]) != self.provider.name:
            raise ValueError("dense index was created by a different embedding model")

        with self._model_lock:
            query_embedding = self.provider.embed_query(request.query)
        if len(query_embedding) != int(configured["dimension"]):
            raise ValueError("query embedding dimension does not match the dense index")

        top_k = min(request.top_k, self.settings.native_search_top_k)
        conditions = ["embedding MATCH ?", "k = ?", "knowledge_base = ?"]
        parameters: list[object] = [
            sqlite_vec.serialize_float32(query_embedding),
            top_k,
            request.knowledge_base,
        ]
        if request.expected_document_id is not None:
            conditions.append("document_id = ?")
            parameters.append(request.expected_document_id)
        sql = f"""
            SELECT
                chunk_id,
                document_id,
                heading_path,
                content,
                distance
            FROM dense_chunks
            WHERE {" AND ".join(conditions)}
            ORDER BY distance
        """
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        rows.sort(key=lambda row: (float(row["distance"]), str(row["chunk_id"])))
        return [
            SearchHit(
                document_id=str(row["document_id"]),
                chunk_id=str(row["chunk_id"]),
                content=str(row["content"]),
                score=1.0 / (1.0 + max(0.0, float(row["distance"]))),
                heading_path=[
                    part for part in str(row["heading_path"]).splitlines() if part
                ],
            )
            for row in rows
        ]
