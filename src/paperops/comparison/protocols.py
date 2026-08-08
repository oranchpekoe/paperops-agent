"""Semantic boundary consumed by the comparison graph."""

from typing import Protocol

from paperops.comparison.models import (
    ComparisonExtraction,
    ComparisonExtractionRequest,
)
from paperops.research.models import ModelCallUsage


class ComparisonModel(Protocol):
    """Extract typed matrix cells without controlling graph execution."""

    name: str

    async def extract_comparison(
        self,
        request: ComparisonExtractionRequest,
    ) -> ComparisonExtraction:
        """Fill requested dimensions from one paper's supplied evidence."""

    def drain_usage(self) -> list[ModelCallUsage]:
        """Return and clear telemetry recorded since the previous drain."""
