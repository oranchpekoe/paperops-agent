"""OpenAI-compatible JSON chat adapter for typed research decisions."""

from __future__ import annotations

import json
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from paperops.clients.errors import ResearchModelError
from paperops.clients.http import require_json_object
from paperops.research.models import (
    AnswerSynthesisRequest,
    EvidenceAssessment,
    EvidenceAssessmentRequest,
    QueryRewrite,
    QueryRewriteRequest,
    ResearchAnswer,
)
from paperops.research.prompts import (
    ASSESS_EVIDENCE,
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

    async def aclose(self) -> None:
        """Close the owned HTTP connection pool."""
        await self._client.aclose()

    async def assess_evidence(
        self,
        request: EvidenceAssessmentRequest,
    ) -> EvidenceAssessment:
        """Judge evidence sufficiency using a strict response schema."""
        return await self._invoke(
            purpose=ASSESS_EVIDENCE,
            request=request,
            response_type=EvidenceAssessment,
        )

    async def rewrite_query(self, request: QueryRewriteRequest) -> QueryRewrite:
        """Generate one focused search query for an identified evidence gap."""
        return await self._invoke(
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
            purpose=SYNTHESIZE_ANSWER,
            request=request,
            response_type=ResearchAnswer,
        )

    async def _invoke(
        self,
        *,
        purpose: str,
        request: BaseModel,
        response_type: type[ResponseModel],
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
        try:
            return response_type.model_validate(parsed)
        except ValidationError as exc:
            raise ResearchModelError(
                "Research model output failed schema validation: "
                f"{exc.errors(include_url=False)}"
            ) from exc

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
