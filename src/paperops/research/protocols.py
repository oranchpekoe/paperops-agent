"""Model boundary consumed by semantic research workflow nodes."""

from typing import Protocol

from paperops.research.models import (
    AnswerSynthesisRequest,
    EvidenceAssessment,
    EvidenceAssessmentRequest,
    ModelCallUsage,
    QueryRewrite,
    QueryRewriteRequest,
    ResearchAnswer,
)


class ResearchModel(Protocol):
    """Provide typed semantic decisions without owning graph control flow."""

    name: str

    async def assess_evidence(
        self,
        request: EvidenceAssessmentRequest,
    ) -> EvidenceAssessment:
        """Judge whether the current evidence can answer the question."""

    async def rewrite_query(self, request: QueryRewriteRequest) -> QueryRewrite:
        """Return one focused query for the missing evidence."""

    async def synthesize_answer(
        self,
        request: AnswerSynthesisRequest,
    ) -> ResearchAnswer:
        """Answer strictly from the supplied evidence."""

    def drain_usage(self) -> list[ModelCallUsage]:
        """Return and clear telemetry recorded since the previous drain."""
