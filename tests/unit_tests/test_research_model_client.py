"""HTTP contract tests for the OpenAI-compatible research model adapter."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from paperops.clients.errors import ResearchModelError
from paperops.comparison.models import (
    ComparisonDimension,
    ComparisonDocument,
    ComparisonExtractionRequest,
)
from paperops.research.models import (
    EvidenceAssessmentRequest,
    EvidenceCitation,
)
from paperops.research.openai_compatible import OpenAICompatibleResearchModel
from paperops.settings import Settings


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        research_model_mode="openai_compatible",
        research_model_base_url="https://model.example/v1",
        research_model_api_key=SecretStr("test-only-key"),
        research_model_name="test-model",
    )


def _assessment_request() -> EvidenceAssessmentRequest:
    return EvidenceAssessmentRequest(
        question="What improves recall?",
        attempted_queries=["recall"],
        evidence=[
            EvidenceCitation(
                citation_id="E1",
                document_id="paper-1",
                chunk_id="chunk-1",
                content="Hybrid retrieval improves recall.",
                score=0.9,
                retrieval_query="recall",
                retrieval_round=1,
            )
        ],
    )


@pytest.mark.asyncio
async def test_adapter_keeps_user_payload_out_of_system_prompt() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                },
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "sufficient": True,
                                    "confidence": 0.91,
                                    "rationale": "The evidence answers the question.",
                                    "missing_aspects": [],
                                    "relevant_citation_ids": ["E1"],
                                }
                            )
                        }
                    }
                ],
            },
        )

    client = OpenAICompatibleResearchModel(
        _settings(),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.assess_evidence(_assessment_request())
    finally:
        await client.aclose()

    payload = captured["payload"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert "What improves recall?" not in messages[0]["content"]
    assert "What improves recall?" in messages[1]["content"]
    assert payload["response_format"] == {"type": "json_object"}
    assert captured["authorization"] == "Bearer test-only-key"
    assert result.sufficient is True
    usage = client.drain_usage()
    assert len(usage) == 1
    assert usage[0].operation == "assess_evidence"
    assert usage[0].success is True
    assert usage[0].total_tokens == 150


@pytest.mark.asyncio
async def test_adapter_rejects_non_json_message_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not json"}}]},
        )

    client = OpenAICompatibleResearchModel(
        _settings(),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ResearchModelError, match="non-JSON"):
            await client.assess_evidence(_assessment_request())
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_adapter_extracts_typed_comparison_cells() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "exactly one cell" in payload["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "document_id": "paper-1",
                                    "cells": [
                                        {
                                            "document_id": "paper-1",
                                            "dimension_id": "method",
                                            "status": "supported",
                                            "claim": "It uses hybrid retrieval [E1].",
                                            "citation_ids": ["E1"],
                                            "confidence": 0.93,
                                            "missing_reason": None,
                                            "suggested_query": None,
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = OpenAICompatibleResearchModel(
        _settings(),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.extract_comparison(
            ComparisonExtractionRequest(
                document=ComparisonDocument(
                    document_id="paper-1",
                    label="Paper One",
                ),
                dimensions=[
                    ComparisonDimension(
                        dimension_id="method",
                        description="Which method is proposed?",
                    )
                ],
                evidence=_assessment_request().evidence,
            )
        )
    finally:
        await client.aclose()

    assert result.cells[0].citation_ids == ["E1"]
    usage = client.drain_usage()
    assert usage[0].operation == "extract_comparison"
    assert usage[0].success is True


@pytest.mark.asyncio
async def test_adapter_restores_unambiguous_comparison_document_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cells": [
                                        {
                                            "dimension_id": "method",
                                            "status": "missing",
                                            "confidence": 0.8,
                                            "missing_reason": "No method evidence.",
                                            "suggested_query": "method architecture",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = OpenAICompatibleResearchModel(
        _settings(),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.extract_comparison(
            ComparisonExtractionRequest(
                document=ComparisonDocument(
                    document_id="paper-1",
                    label="Paper One",
                ),
                dimensions=[
                    ComparisonDimension(
                        dimension_id="method",
                        description="Which method is proposed?",
                    )
                ],
                evidence=_assessment_request().evidence,
            )
        )
    finally:
        await client.aclose()

    assert result.document_id == "paper-1"
    assert result.cells[0].document_id == "paper-1"
