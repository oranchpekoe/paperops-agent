"""Opt-in smoke test against MinerU and the native retrieval backend."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from paperops.clients.mineru import MinerUClient
from paperops.models import IngestRequest, ParseRequest, SearchRequest
from paperops.retrieval.chunking import build_index_probe
from paperops.retrieval.native import NativeRetrievalBackend
from paperops.settings import Settings


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_mineru_to_native_retrieval_round_trip(tmp_path: Path) -> None:
    """Parse with MinerU, then chunk, index, and retrieve locally."""
    if os.getenv("PAPEROPS_RUN_LIVE_INTEGRATION") != "1":
        pytest.skip("Set PAPEROPS_RUN_LIVE_INTEGRATION=1 to allow external writes")
    source_value = os.getenv("PAPEROPS_INTEGRATION_PDF", "")
    collection_id = os.getenv("PAPEROPS_INTEGRATION_COLLECTION_ID", "live-papers")
    source = Path(source_value)
    if not source.is_file() or source.suffix.lower() != ".pdf":
        pytest.skip("PAPEROPS_INTEGRATION_PDF must point to a readable PDF")
    settings = Settings(
        artifacts_dir=tmp_path / "artifacts",
        native_index_db=tmp_path / "native-index.db",
        client_mode="real",
        retrieval_backend="native",
    )

    file_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    job_id = f"live-{file_hash[:16]}"
    parser = MinerUClient(settings)
    knowledge_base = NativeRetrievalBackend(settings)
    try:
        parsed = await parser.parse(
            ParseRequest(
                job_id=job_id,
                source_pdf=str(source),
                file_hash=file_hash,
                attempt=1,
                idempotency_key=f"parse:{job_id}:1",
            )
        )
        ingested = await knowledge_base.ingest(
            IngestRequest(
                job_id=job_id,
                knowledge_base=collection_id,
                file_hash=file_hash,
                markdown_path=parsed.markdown_path,
                idempotency_key=f"ingest:{collection_id}:{file_hash}",
            )
        )
        hits = await knowledge_base.search(
            SearchRequest(
                knowledge_base=collection_id,
                query=build_index_probe(
                    Path(parsed.markdown_path).read_text(encoding="utf-8"),
                    source.stem,
                ),
                expected_document_id=ingested.document_id,
            )
        )
    finally:
        await parser.aclose()

    assert Path(parsed.markdown_path).is_file()
    assert any(hit.document_id == ingested.document_id for hit in hits)
