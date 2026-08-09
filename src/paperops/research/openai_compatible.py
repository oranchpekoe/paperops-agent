"""OpenAI-compatible JSON chat adapter for typed research decisions."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Callable, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from paperops.clients.errors import ResearchModelError
from paperops.clients.http import require_json_object
from paperops.comparison.models import (
    ComparisonExtraction,
    ComparisonExtractionRequest,
)
from paperops.research.models import (
    AnswerSynthesisRequest,
    EvidenceAssessment,
    EvidenceAssessmentRequest,
    ModelCallUsage,
    QueryRewrite,
    QueryRewriteRequest,
    ResearchAnswer,
)
from paperops.research.prompts import (
    ASSESS_EVIDENCE,
    EXTRACT_COMPARISON,
    REWRITE_QUERY,
    SYNTHESIZE_ANSWER,
    system_prompt,
)
from paperops.settings import Settings

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class OpenAICompatibleResearchModel:
    """Call a JSON-mode chat-completions endpoint with strict typed outputs."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create a client from explicit model configuration."""
        api_key = settings.research_model_api_key.get_secret_value()
        if not api_key:
            raise ValueError(
                "PAPEROPS_RESEARCH_MODEL_API_KEY is required when "
                "research_model_mode=openai_compatible"
            )
        self.name = f"openai-compatible:{settings.research_model_name}"
        self._model = settings.research_model_name
        self._endpoint = (
            f"{settings.research_model_base_url.rstrip('/')}/chat/completions"
        )
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=settings.research_model_timeout_seconds,
            trust_env=settings.external_trust_env,
            proxy=settings.research_model_proxy_url or None,
            transport=transport,
        )
        self._usage: list[ModelCallUsage] = []

    async def aclose(self) -> None:
        """Close the owned HTTP connection pool."""
        await self._client.aclose()

    async def assess_evidence(
        self,
        request: EvidenceAssessmentRequest,
    ) -> EvidenceAssessment:
        """Judge evidence sufficiency using a strict response schema."""
        return await self._invoke(
            operation="assess_evidence",
            purpose=ASSESS_EVIDENCE,
            request=request,
            response_type=EvidenceAssessment,
        )

    async def rewrite_query(self, request: QueryRewriteRequest) -> QueryRewrite:
        """Generate one focused search query for an identified evidence gap."""
        return await self._invoke(
            operation="rewrite_query",
            purpose=REWRITE_QUERY,
            request=request,
            response_type=QueryRewrite,
        )

    async def synthesize_answer(
        self,
        request: AnswerSynthesisRequest,
    ) -> ResearchAnswer:
        """Synthesize only supported claims with inline evidence markers."""
        return await self._invoke(
            operation="synthesize_answer",
            purpose=SYNTHESIZE_ANSWER,
            request=request,
            response_type=ResearchAnswer,
        )

    async def extract_comparison(
        self,
        request: ComparisonExtractionRequest,
    ) -> ComparisonExtraction:
        """Extract a typed evidence matrix row for one document."""
        return await self._invoke(
            operation="extract_comparison",
            purpose=EXTRACT_COMPARISON,
            request=request,
            response_type=ComparisonExtraction,
            normalize=lambda payload: self._restore_comparison_document_ids(
                payload,
                request.document.document_id,
            ),
        )

    async def _invoke(
        self,
        *,
        operation: str,
        purpose: str,
        request: BaseModel,
        response_type: type[ResponseModel],
        normalize: Callable[[Any], Any] | None = None,
    ) -> ResponseModel:
        """Call JSON mode and validate both transport and semantic shape."""
        schema = response_type.model_json_schema()
        payload = {
            "model": self._model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt(purpose, schema),
                },
                {
                    "role": "user",
                    "content": request.model_dump_json(),
                },
            ],
        }
        started = perf_counter()
        body: dict[str, Any] | None = None
        success = False
        try:
            try:
                response = await self._client.post(self._endpoint, json=payload)
            except httpx.HTTPError as exc:
                raise ResearchModelError(
                    f"Research model request failed: {type(exc).__name__}: {exc}"
                ) from exc
            body = require_json_object(
                response,
                service="Research model",
                error_type=ResearchModelError,
            )
            content = self._message_content(body)
            try:
                parsed: Any = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ResearchModelError(
                    "Research model returned non-JSON message content"
                ) from exc
            if normalize is not None:
                parsed = normalize(parsed)
            try:
                result = response_type.model_validate(parsed)
            except ValidationError as exc:
                raise ResearchModelError(
                    "Research model output failed schema validation: "
                    f"{exc.errors(include_url=False)}"
                ) from exc
            success = True
            return result
        finally:
            self._usage.append(
                self._usage_record(
                    operation,
                    body,
                    success=success,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )

    @staticmethod
    def _restore_comparison_document_ids(
        payload: Any,
        document_id: str,
    ) -> Any:
        """Fill only omitted ids that are unambiguous in a single-document call."""
        if not isinstance(payload, dict):
            return payload
        normalized = dict(payload)
        if not normalized.get("document_id"):
            normalized["document_id"] = document_id
        cells = normalized.get("cells")
        if not isinstance(cells, list):
            return normalized
        normalized["cells"] = [
            (
                {**cell, "document_id": document_id}
                if isinstance(cell, dict) and not cell.get("document_id")
                else cell
            )
            for cell in cells
        ]
        return normalized

    def drain_usage(self) -> list[ModelCallUsage]:
        """Return and clear provider telemetry captured by the adapter."""
        usage, self._usage = self._usage, []
        return usage

    @staticmethod
    def _usage_record(
        operation: str,
        body: dict[str, Any] | None,
        *,
        success: bool,
        latency_ms: float,
    ) -> ModelCallUsage:
        raw_usage = body.get("usage") if body is not None else None
        usage = raw_usage if isinstance(raw_usage, dict) else {}

        def token_count(name: str) -> int | None:
            value = usage.get(name)
            return (
                value
                if isinstance(value, int) and not isinstance(value, bool)
                else None
            )

        return ModelCallUsage(
            operation=operation,
            success=success,
            latency_ms=latency_ms,
            prompt_tokens=token_count("prompt_tokens"),
            completion_tokens=token_count("completion_tokens"),
            total_tokens=token_count("total_tokens"),
        )

    @staticmethod
    def _message_content(body: dict[str, Any]) -> str:
        """Extract one string message without accepting ambiguous alternatives."""
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ResearchModelError(
                "Research model response must contain exactly one choice"
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ResearchModelError("Research model choice must be an object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ResearchModelError("Research model choice has no message object")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ResearchModelError("Research model message content is empty")
        return content
