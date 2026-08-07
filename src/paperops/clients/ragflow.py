"""HTTP adapter for RAGFlow document ingestion and retrieval APIs."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from paperops.clients.errors import RAGFlowError, RAGFlowTimeout
from paperops.clients.http import require_json_object
from paperops.models import (
    IngestRequest,
    IngestResult,
    SearchHit,
    SearchRequest,
)
from paperops.settings import Settings


def _document_progress(document: dict[str, Any]) -> float:
    """Normalize RAGFlow's progress field for status decisions."""
    try:
        return float(document.get("progress", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _document_run(document: dict[str, Any]) -> str:
    """Normalize the human-readable run state returned by RAGFlow."""
    return str(document.get("run", "")).strip().upper()


class RAGFlowClient:
    """Upload Markdown, wait for indexing, and retrieve evidence from RAGFlow."""

    name = "ragflow"

    def __init__(
        self,
        settings: Settings,
        *,
        async_client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ) -> None:
        """Configure the official RAGFlow v1 HTTP boundary."""
        api_key = settings.ragflow_api_key.get_secret_value().strip()
        if not api_key:
            raise RAGFlowError(
                "PAPEROPS_RAGFLOW_API_KEY is required when the RAGFlow "
                "retrieval backend is selected"
            )
        self.settings = settings
        root = settings.ragflow_base_url.rstrip("/")
        self.api_url = root if root.endswith("/api/v1") else f"{root}/api/v1"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(
            connect=settings.external_connect_timeout_seconds,
            read=settings.external_read_timeout_seconds,
            write=settings.external_write_timeout_seconds,
            pool=settings.external_connect_timeout_seconds,
        )
        self._async_client = async_client or httpx.AsyncClient(
            timeout=timeout,
            trust_env=settings.external_trust_env,
        )
        self._sync_client = sync_client or httpx.Client(
            timeout=timeout,
            trust_env=settings.external_trust_env,
        )
        self._owns_async_client = async_client is None
        self._owns_sync_client = sync_client is None
        self._locks: dict[str, asyncio.Lock] = {}

    async def ingest(self, request: IngestRequest) -> IngestResult:
        """Create or reuse one deterministic document and wait for indexing."""
        lock = self._locks.setdefault(request.idempotency_key, asyncio.Lock())
        async with lock:
            return await self._ingest_locked(request)

    async def _ingest_locked(self, request: IngestRequest) -> IngestResult:
        upload_name = f"paper-{request.file_hash[:20]}.md"
        document = await self._find_document(request.knowledge_base, upload_name)
        created = document is None
        if document is None:
            document = await asyncio.to_thread(
                self._upload_document,
                request.knowledge_base,
                Path(request.markdown_path),
                upload_name,
            )

        document_id = document.get("id")
        if not isinstance(document_id, str) or not document_id:
            raise RAGFlowError("RAGFlow returned a document without a valid id")

        if self._is_indexed(document):
            return IngestResult(
                document_id=document_id,
                idempotency_key=request.idempotency_key,
                created=created,
                chunk_count=0,
            )

        if created or self._should_start_indexing(document):
            await self._start_indexing(request.knowledge_base, document_id)
        await self._wait_for_indexing(request.knowledge_base, document_id)
        return IngestResult(
            document_id=document_id,
            idempotency_key=request.idempotency_key,
            created=created,
            chunk_count=0,
        )

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        """Retrieve chunks from the expected document using RAGFlow retrieval."""
        payload = {
            "dataset_ids": [request.knowledge_base],
            "question": request.query,
            "page": 1,
            "page_size": min(request.top_k, self.settings.ragflow_page_size),
            "similarity_threshold": self.settings.ragflow_similarity_threshold,
        }
        if request.expected_document_id is not None:
            payload["document_ids"] = [request.expected_document_id]
        response = await self._post_json("/retrieval", payload)
        data = response.get("data")
        chunks = data.get("chunks", []) if isinstance(data, dict) else []
        if not isinstance(chunks, list):
            raise RAGFlowError("RAGFlow retrieval returned invalid chunks")

        hits: list[SearchHit] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            document_id = chunk.get("document_id")
            if not isinstance(document_id, str) or not document_id:
                continue
            content = chunk.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            try:
                raw_score = float(chunk.get("similarity", 0.0) or 0.0)
            except (TypeError, ValueError):
                raw_score = 0.0
            hits.append(
                SearchHit(
                    document_id=document_id,
                    chunk_id=(
                        str(chunk["id"]) if chunk.get("id") is not None else None
                    ),
                    content=content,
                    score=max(0.0, min(1.0, raw_score)),
                )
            )
        return hits

    async def _find_document(
        self,
        dataset_id: str,
        name: str,
    ) -> dict[str, Any] | None:
        payload = await self._get_json(
            f"/datasets/{quote(dataset_id, safe='')}/documents",
            params={"name": name, "page": 1, "page_size": 30},
        )
        data = payload.get("data")
        documents = data.get("docs", []) if isinstance(data, dict) else []
        if not isinstance(documents, list):
            raise RAGFlowError("RAGFlow returned an invalid document list")
        exact = [
            document
            for document in documents
            if isinstance(document, dict) and document.get("name") == name
        ]
        if len(exact) > 1:
            raise RAGFlowError(
                f"RAGFlow contains duplicate idempotent document names: {name}"
            )
        return exact[0] if exact else None

    def _upload_document(
        self,
        dataset_id: str,
        markdown_path: Path,
        upload_name: str,
    ) -> dict[str, Any]:
        try:
            with markdown_path.open("rb") as source:
                response = self._sync_client.post(
                    f"{self.api_url}/datasets/{quote(dataset_id, safe='')}/documents",
                    headers=self._headers,
                    files={"file": (upload_name, source, "text/markdown")},
                )
        except httpx.TimeoutException as exc:
            raise RAGFlowTimeout(
                "Timed out while uploading a RAGFlow document"
            ) from exc
        except httpx.HTTPError as exc:
            raise RAGFlowError(f"Failed to upload a RAGFlow document: {exc}") from exc

        payload = self._require_api_payload(response, "RAGFlow document upload")
        data = payload.get("data")
        if (
            not isinstance(data, list)
            or len(data) != 1
            or not isinstance(data[0], dict)
        ):
            raise RAGFlowError("RAGFlow upload returned an invalid document payload")
        return data[0]

    async def _start_indexing(self, dataset_id: str, document_id: str) -> None:
        await self._post_json(
            f"/datasets/{quote(dataset_id, safe='')}/chunks",
            {"document_ids": [document_id]},
        )

    async def _wait_for_indexing(self, dataset_id: str, document_id: str) -> None:
        deadline = time.monotonic() + self.settings.ragflow_index_timeout_seconds
        while time.monotonic() < deadline:
            document = await self._get_document(dataset_id, document_id)
            if self._is_indexed(document):
                return
            run_state = _document_run(document)
            progress = _document_progress(document)
            if run_state in {"FAIL", "FAILED", "CANCEL", "CANCELLED"} or progress < 0:
                message = document.get("progress_msg") or "no error detail"
                raise RAGFlowError(
                    f"RAGFlow indexing failed for {document_id}: {message}"
                )
            await asyncio.sleep(self.settings.ragflow_poll_interval_seconds)
        raise RAGFlowTimeout(
            f"RAGFlow document {document_id} did not finish indexing within "
            f"{self.settings.ragflow_index_timeout_seconds:g}s"
        )

    async def _get_document(
        self,
        dataset_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        payload = await self._get_json(
            f"/datasets/{quote(dataset_id, safe='')}/documents",
            params={"id": document_id, "page": 1, "page_size": 1},
        )
        data = payload.get("data")
        documents = data.get("docs", []) if isinstance(data, dict) else []
        if not isinstance(documents, list) or not documents:
            raise RAGFlowError(
                f"RAGFlow document disappeared while indexing: {document_id}"
            )
        document = documents[0]
        if not isinstance(document, dict) or document.get("id") != document_id:
            raise RAGFlowError("RAGFlow returned the wrong document while polling")
        return document

    @staticmethod
    def _is_indexed(document: dict[str, Any]) -> bool:
        return _document_run(document) == "DONE" or _document_progress(document) >= 1.0

    @staticmethod
    def _should_start_indexing(document: dict[str, Any]) -> bool:
        run_state = _document_run(document)
        progress = _document_progress(document)
        return progress <= 0.0 and run_state not in {"RUNNING", "PROCESSING", "1"}

    async def _get_json(
        self,
        path: str,
        *,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = await self._async_client.get(
                f"{self.api_url}{path}",
                headers=self._headers,
                params=params,
            )
        except httpx.TimeoutException as exc:
            raise RAGFlowTimeout(f"Timed out while calling RAGFlow {path}") from exc
        except httpx.HTTPError as exc:
            raise RAGFlowError(f"Failed to call RAGFlow {path}: {exc}") from exc
        return self._require_api_payload(response, f"RAGFlow {path}")

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = await self._async_client.post(
                f"{self.api_url}{path}",
                headers=self._headers,
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise RAGFlowTimeout(f"Timed out while calling RAGFlow {path}") from exc
        except httpx.HTTPError as exc:
            raise RAGFlowError(f"Failed to call RAGFlow {path}: {exc}") from exc
        return self._require_api_payload(response, f"RAGFlow {path}")

    @staticmethod
    def _require_api_payload(
        response: httpx.Response,
        service: str,
    ) -> dict[str, Any]:
        payload = require_json_object(
            response,
            service=service,
            error_type=RAGFlowError,
        )
        if payload.get("code") != 0:
            raise RAGFlowError(
                f"{service} failed with code {payload.get('code')}: "
                f"{payload.get('message', 'unknown error')}"
            )
        return payload

    async def aclose(self) -> None:
        """Close owned HTTP connection pools."""
        if self._owns_async_client:
            await self._async_client.aclose()
        if self._owns_sync_client:
            await asyncio.to_thread(self._sync_client.close)
