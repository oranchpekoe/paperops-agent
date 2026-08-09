"""Shared evidence normalization for research workflows."""

from __future__ import annotations

import hashlib

from paperops.models import SearchHit
from paperops.research.models import EvidenceCitation


def append_single_inline_citation(text: str, citation_id: str) -> str:
    """Repair one unambiguous omitted marker without guessing claim boundaries."""
    normalized = text.rstrip()
    if normalized[-1:] in {".", "!", "?", "。", "！", "？"}:
        return f"{normalized[:-1].rstrip()} [{citation_id}]{normalized[-1]}"
    return f"{normalized} [{citation_id}]"


def hit_key(hit: SearchHit) -> str:
    """Build a stable identity even when a backend omits chunk ids."""
    if hit.chunk_id:
        return f"{hit.document_id}:{hit.chunk_id}"
    digest = hashlib.sha256(hit.content.encode()).hexdigest()
    return f"{hit.document_id}:content-{digest[:16]}"


def merge_evidence(
    existing: list[EvidenceCitation],
    hits: list[SearchHit],
    *,
    query: str,
    retrieval_round: int,
    max_chunk_chars: int,
    max_evidence_chars: int,
) -> list[EvidenceCitation]:
    """Deduplicate and bound checkpointed retrieval payloads."""
    merged = list(existing)
    seen = {f"{item.document_id}:{item.chunk_id}" for item in existing}
    used_chars = sum(len(item.content) for item in existing)
    for hit in hits:
        key = hit_key(hit)
        if key in seen or used_chars >= max_evidence_chars:
            continue
        content = hit.content.strip()[:max_chunk_chars]
        remaining = max_evidence_chars - used_chars
        content = content[:remaining].strip()
        if not content:
            continue
        chunk_id = hit.chunk_id or key.split(":", 1)[1]
        merged.append(
            EvidenceCitation(
                citation_id=f"E{len(merged) + 1}",
                document_id=hit.document_id,
                chunk_id=chunk_id,
                content=content,
                score=hit.score,
                heading_path=hit.heading_path,
                retrieval_query=query,
                retrieval_round=retrieval_round,
            )
        )
        seen.add(key)
        used_chars += len(content)
    return merged
