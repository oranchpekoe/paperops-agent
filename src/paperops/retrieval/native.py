"""SQLite FTS5 retrieval backend with explicit chunking and ranking."""

from __future__ import annotations

import asyncio
import hashlib
import re
import sqlite3
from pathlib import Path
from threading import Lock

from paperops.models import (
    IngestRequest,
    IngestResult,
    SearchHit,
    SearchRequest,
)
from paperops.retrieval.chunking import chunk_markdown
from paperops.settings import Settings

_ENGLISH_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{3,}")
_STOP_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "indexed",
    "into",
    "the",
    "this",
    "what",
    "with",
}


def stable_document_id(idempotency_key: str) -> str:
    """Derive one backend-independent local document identifier."""
    digest = hashlib.sha256(idempotency_key.encode()).hexdigest()
    return f"native-{digest[:20]}"


def stable_chunk_id(document_id: str, ordinal: int, content: str) -> str:
    """Derive a stable chunk identifier shared by local retrievers."""
    digest = hashlib.sha256(f"{document_id}:{ordinal}:{content}".encode()).hexdigest()
    return f"chunk-{digest[:20]}"


def _match_expression(query: str) -> str:
    """Convert untrusted query text into a bounded FTS5 OR expression."""
    terms: list[str] = []
    for term in _ENGLISH_TERM_RE.findall(query.lower()):
        if term not in _STOP_WORDS and term not in terms:
            terms.append(term)
    for run in _CJK_RUN_RE.findall(query):
        for index in range(len(run) - 2):
            term = run[index : index + 3]
            if term not in terms:
                terms.append(term)
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms[:32])


class NativeRetrievalBackend:
    """Persist chunks locally and expose a transparent BM25 retrieval baseline."""

    name = "native_fts5_bm25"

    def __init__(self, settings: Settings) -> None:
        """Configure the SQLite index and deterministic chunking policy."""
        if settings.native_chunk_overlap_chars >= settings.native_chunk_size_chars:
            raise ValueError(
                "PAPEROPS_NATIVE_CHUNK_OVERLAP_CHARS must be smaller than "
                "PAPEROPS_NATIVE_CHUNK_SIZE_CHARS"
            )
        self.settings = settings
        self.database_path = settings.native_index_db
        self._write_lock = Lock()

    def _connect(self) -> sqlite3.Connection:
        """Open one short-lived SQLite connection with bounded lock waiting."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        self._ensure_schema(connection)
        return connection

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        """Create the durable document table and FTS5 chunk index."""
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS native_documents (
                document_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                knowledge_base TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                markdown_path TEXT NOT NULL,
                chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS native_chunks USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                knowledge_base UNINDEXED,
                heading_path,
                content,
                tokenize='trigram'
            );
            """
        )

    async def ingest(self, request: IngestRequest) -> IngestResult:
        """Chunk and atomically index a Markdown artifact, or reuse it."""
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
                FROM native_documents
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

            connection.execute(
                """
                INSERT INTO native_documents (
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
            for chunk in chunks:
                connection.execute(
                    """
                    INSERT INTO native_chunks (
                        chunk_id,
                        document_id,
                        knowledge_base,
                        heading_path,
                        content
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        stable_chunk_id(document_id, chunk.ordinal, chunk.content),
                        document_id,
                        request.knowledge_base,
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
        """Run document-filtered or collection-wide BM25 retrieval."""
        return await asyncio.to_thread(self._search_sync, request)

    def _search_sync(self, request: SearchRequest) -> list[SearchHit]:
        expression = _match_expression(request.query)
        if not expression:
            return []
        top_k = min(request.top_k, self.settings.native_search_top_k)
        conditions = ["native_chunks MATCH ?", "knowledge_base = ?"]
        parameters: list[str | int] = [expression, request.knowledge_base]
        if request.expected_document_id is not None:
            conditions.append("document_id = ?")
            parameters.append(request.expected_document_id)
        parameters.append(top_k)
        sql = f"""
            SELECT
                chunk_id,
                document_id,
                heading_path,
                content,
                bm25(native_chunks, 0.0, 0.0, 0.0, 2.0, 1.0) AS rank
            FROM native_chunks
            WHERE {" AND ".join(conditions)}
            ORDER BY rank, chunk_id
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        if not rows:
            return []

        raw_scores = [max(0.0, -float(row["rank"])) for row in rows]
        best_score = max(raw_scores, default=0.0)
        hits: list[SearchHit] = []
        for position, (row, raw_score) in enumerate(zip(rows, raw_scores, strict=True)):
            score = raw_score / best_score if best_score > 0.0 else 1.0 / (position + 1)
            hits.append(
                SearchHit(
                    document_id=str(row["document_id"]),
                    chunk_id=str(row["chunk_id"]),
                    content=str(row["content"]),
                    score=max(0.0, min(1.0, score)),
                    heading_path=[
                        part for part in str(row["heading_path"]).splitlines() if part
                    ],
                )
            )
        return hits
