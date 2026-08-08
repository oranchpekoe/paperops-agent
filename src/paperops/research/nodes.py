"""Dependency-injected nodes for bounded evidence gathering and answering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from paperops.clients.protocols import RetrievalBackend
from paperops.models import SearchHit, SearchRequest
from paperops.research.models import (
    AnswerSynthesisRequest,
    EvidenceAssessment,
    EvidenceAssessmentRequest,
    EvidenceCitation,
    QueryRewrite,
    QueryRewriteRequest,
    ResearchAnswer,
    ResearchEvent,
    ResearchFailure,
    ResearchFailureCode,
    ResearchStatus,
)
from paperops.research.protocols import ResearchModel
from paperops.research.state import ResearchQueryState
from paperops.settings import Settings


def _query_id(knowledge_base: str, question: str) -> str:
    """Derive a stable logical identity without replacing checkpoint thread ids."""
    digest = hashlib.sha256(f"{knowledge_base}:{question}".encode()).hexdigest()
    return f"query-{digest[:16]}"


def _failure_update(
    stage: ResearchStatus,
    code: ResearchFailureCode,
    message: str,
    *,
    retryable: bool = False,
) -> dict[str, Any]:
    """Return a terminal structured failure update."""
    failure = ResearchFailure(
        stage=stage,
        code=code,
        message=message,
        retryable=retryable,
    )
    return {
        "status": ResearchStatus.FAILED,
        "failure": failure,
        "errors": [failure],
        "events": [
            ResearchEvent(
                status=ResearchStatus.FAILED,
                message=f"{stage.value} failed: {code.value}.",
            )
        ],
    }


def _hit_key(hit: SearchHit) -> str:
    """Build a stable identity even when an external backend omits chunk ids."""
    if hit.chunk_id:
        return f"{hit.document_id}:{hit.chunk_id}"
    digest = hashlib.sha256(hit.content.encode()).hexdigest()
    return f"{hit.document_id}:content-{digest[:16]}"


def _merge_evidence(
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
        key = _hit_key(hit)
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


@dataclass(slots=True)
class ResearchNodes:
    """Hold retrieval and semantic dependencies outside serialised graph state."""

    retrieval: RetrievalBackend
    model: ResearchModel
    settings: Settings

    async def initialize(self, state: ResearchQueryState) -> dict[str, Any]:
        """Validate and normalize the immutable query inputs."""
        knowledge_base = state.get("knowledge_base", "").strip()
        question = state.get("question", "").strip()
        if not knowledge_base or not question:
            return _failure_update(
                ResearchStatus.PENDING,
                ResearchFailureCode.INVALID_QUERY,
                "Both knowledge_base and question are required.",
            )
        return {
            "query_id": _query_id(knowledge_base, question),
            "knowledge_base": knowledge_base,
            "question": question,
            "current_query": question,
            "status": ResearchStatus.RETRIEVING,
            "retrieval_round": 0,
            "rewrite_count": 0,
            "retrieval_calls": 0,
            "model_calls": 0,
            "attempted_queries": [],
            "evidence": [],
            "events": [
                ResearchEvent(
                    status=ResearchStatus.PENDING,
                    message="Research query validated.",
                )
            ],
        }

    async def retrieve(self, state: ResearchQueryState) -> dict[str, Any]:
        """Execute one bounded retrieval round and checkpoint deduplicated evidence."""
        retrieval_round = state.get("retrieval_round", 0) + 1
        query = state["current_query"].strip()
        try:
            hits = await self.retrieval.search(
                SearchRequest(
                    knowledge_base=state["knowledge_base"],
                    query=query,
                    top_k=self.settings.research_search_top_k,
                )
            )
        except Exception as exc:
            update = _failure_update(
                ResearchStatus.RETRIEVING,
                ResearchFailureCode.RETRIEVAL_ERROR,
                f"{type(exc).__name__}: {exc}",
                retryable=True,
            )
            update["retrieval_calls"] = state.get("retrieval_calls", 0) + 1
            return update

        evidence = _merge_evidence(
            state.get("evidence", []),
            hits,
            query=query,
            retrieval_round=retrieval_round,
            max_chunk_chars=self.settings.research_max_chunk_chars,
            max_evidence_chars=self.settings.research_max_evidence_chars,
        )
        return {
            "status": ResearchStatus.ASSESSING,
            "retrieval_round": retrieval_round,
            "retrieval_calls": state.get("retrieval_calls", 0) + 1,
            "attempted_queries": [*state.get("attempted_queries", []), query],
            "evidence": evidence,
            "events": [
                ResearchEvent(
                    status=ResearchStatus.RETRIEVING,
                    message=(
                        f"Retrieved {len(hits)} hit(s); retained "
                        f"{len(evidence)} unique evidence chunk(s)."
                    ),
                    retrieval_round=retrieval_round,
                )
            ],
        }

    async def assess(self, state: ResearchQueryState) -> dict[str, Any]:
        """Use a typed model decision to route, without delegating loop control."""
        evidence = state.get("evidence", [])
        if len(evidence) < self.settings.research_min_evidence_hits:
            assessment = EvidenceAssessment(
                sufficient=False,
                confidence=1.0,
                rationale="The deterministic minimum evidence policy was not met.",
                missing_aspects=["retrievable supporting evidence"],
            )
            model_calls = state.get("model_calls", 0)
        else:
            try:
                assessment = EvidenceAssessment.model_validate(
                    await self.model.assess_evidence(
                        EvidenceAssessmentRequest(
                            question=state["question"],
                            attempted_queries=state["attempted_queries"],
                            evidence=evidence,
                        )
                    )
                )
            except ValidationError as exc:
                update = _failure_update(
                    ResearchStatus.ASSESSING,
                    ResearchFailureCode.INVALID_MODEL_OUTPUT,
                    f"Evidence assessment failed validation: {exc}",
                )
                update["model_calls"] = state.get("model_calls", 0) + 1
                return update
            except Exception as exc:
                update = _failure_update(
                    ResearchStatus.ASSESSING,
                    ResearchFailureCode.MODEL_ERROR,
                    f"{type(exc).__name__}: {exc}",
                    retryable=True,
                )
                update["model_calls"] = state.get("model_calls", 0) + 1
                return update
            model_calls = state.get("model_calls", 0) + 1

        if (
            assessment.sufficient
            and assessment.confidence < self.settings.research_min_assessment_confidence
        ):
            assessment = EvidenceAssessment(
                sufficient=False,
                confidence=assessment.confidence,
                rationale=(
                    "The semantic judge confidence was below the configured "
                    "answer threshold."
                ),
                missing_aspects=["higher-confidence supporting evidence"],
            )
        return {
            "status": ResearchStatus.ASSESSING,
            "assessment": assessment,
            "model_calls": model_calls,
            "events": [
                ResearchEvent(
                    status=ResearchStatus.ASSESSING,
                    message=(
                        "Evidence accepted for synthesis."
                        if assessment.sufficient
                        else "Evidence gap recorded for bounded query rewriting."
                    ),
                    retrieval_round=state.get("retrieval_round"),
                )
            ],
        }

    async def rewrite(self, state: ResearchQueryState) -> dict[str, Any]:
        """Generate one focused, non-duplicate retrieval query."""
        try:
            rewrite = QueryRewrite.model_validate(
                await self.model.rewrite_query(
                    QueryRewriteRequest(
                        question=state["question"],
                        attempted_queries=state["attempted_queries"],
                        missing_aspects=state["assessment"].missing_aspects,
                    )
                )
            )
        except ValidationError as exc:
            update = _failure_update(
                ResearchStatus.REWRITING,
                ResearchFailureCode.INVALID_MODEL_OUTPUT,
                f"Query rewrite failed validation: {exc}",
            )
            update["model_calls"] = state.get("model_calls", 0) + 1
            return update
        except Exception as exc:
            update = _failure_update(
                ResearchStatus.REWRITING,
                ResearchFailureCode.MODEL_ERROR,
                f"{type(exc).__name__}: {exc}",
                retryable=True,
            )
            update["model_calls"] = state.get("model_calls", 0) + 1
            return update

        normalized = rewrite.query.strip()
        if not normalized:
            update = _failure_update(
                ResearchStatus.REWRITING,
                ResearchFailureCode.INVALID_MODEL_OUTPUT,
                "Query rewrite was blank after normalization.",
            )
            update["model_calls"] = state.get("model_calls", 0) + 1
            return update
        attempted = {query.casefold() for query in state["attempted_queries"]}
        if normalized.casefold() in attempted:
            return {
                "status": ResearchStatus.INSUFFICIENT_EVIDENCE,
                "last_rewrite": rewrite,
                "model_calls": state.get("model_calls", 0) + 1,
                "events": [
                    ResearchEvent(
                        status=ResearchStatus.INSUFFICIENT_EVIDENCE,
                        message="Model produced a duplicate query; stopped safely.",
                        retrieval_round=state.get("retrieval_round"),
                    )
                ],
            }
        return {
            "status": ResearchStatus.RETRIEVING,
            "current_query": normalized,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "last_rewrite": rewrite,
            "model_calls": state.get("model_calls", 0) + 1,
            "events": [
                ResearchEvent(
                    status=ResearchStatus.REWRITING,
                    message="Focused query rewrite accepted.",
                    retrieval_round=state.get("retrieval_round"),
                )
            ],
        }

    async def synthesize(self, state: ResearchQueryState) -> dict[str, Any]:
        """Generate an evidence-only answer and validate every citation."""
        try:
            answer = ResearchAnswer.model_validate(
                await self.model.synthesize_answer(
                    AnswerSynthesisRequest(
                        question=state["question"],
                        evidence=state["evidence"],
                    )
                )
            )
        except ValidationError as exc:
            update = _failure_update(
                ResearchStatus.ANSWERING,
                ResearchFailureCode.INVALID_MODEL_OUTPUT,
                f"Research answer failed validation: {exc}",
            )
            update["model_calls"] = state.get("model_calls", 0) + 1
            return update
        except Exception as exc:
            update = _failure_update(
                ResearchStatus.ANSWERING,
                ResearchFailureCode.MODEL_ERROR,
                f"{type(exc).__name__}: {exc}",
                retryable=True,
            )
            update["model_calls"] = state.get("model_calls", 0) + 1
            return update

        available = {item.citation_id for item in state["evidence"]}
        cited = answer.citation_ids
        invalid = set(cited) - available
        missing_inline = [item for item in cited if f"[{item}]" not in answer.text]
        if invalid or missing_inline or len(cited) != len(set(cited)):
            details = []
            if invalid:
                details.append(f"unknown citations: {sorted(invalid)}")
            if missing_inline:
                details.append(f"missing inline markers: {missing_inline}")
            if len(cited) != len(set(cited)):
                details.append("duplicate citation ids")
            update = _failure_update(
                ResearchStatus.ANSWERING,
                ResearchFailureCode.CITATION_VALIDATION_ERROR,
                "; ".join(details),
            )
            update["model_calls"] = state.get("model_calls", 0) + 1
            return update
        return {
            "status": ResearchStatus.COMPLETED,
            "answer": answer,
            "model_calls": state.get("model_calls", 0) + 1,
            "events": [
                ResearchEvent(
                    status=ResearchStatus.COMPLETED,
                    message="Evidence-grounded answer and citations validated.",
                    retrieval_round=state.get("retrieval_round"),
                )
            ],
        }

    async def refuse(self, state: ResearchQueryState) -> dict[str, Any]:
        """Stop after the configured rewrite budget without fabricating an answer."""
        return {
            "status": ResearchStatus.INSUFFICIENT_EVIDENCE,
            "events": [
                ResearchEvent(
                    status=ResearchStatus.INSUFFICIENT_EVIDENCE,
                    message=(
                        "Evidence remained insufficient after the configured "
                        "retrieval budget."
                    ),
                    retrieval_round=state.get("retrieval_round"),
                )
            ],
        }
