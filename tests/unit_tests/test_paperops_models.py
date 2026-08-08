"""Unit tests for the initial PaperOps domain boundary."""

import pytest
from pydantic import ValidationError

from paperops import JobStatus, QualityDecision, QualityVerdict
from paperops.settings import Settings


def test_quality_decision_defaults() -> None:
    decision = QualityDecision(
        verdict=QualityVerdict.PASS,
        confidence=0.9,
    )

    assert decision.issues == []
    assert decision.retry_reason is None


def test_quality_decision_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        QualityDecision(
            verdict=QualityVerdict.REVIEW,
            confidence=1.1,
        )


def test_settings_have_safe_local_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.artifacts_dir.name == "artifacts"
    assert settings.knowledge_dir.name == "knowledge"
    assert settings.max_parse_attempts == 2
    assert settings.research_max_rewrites == 0
    assert JobStatus.PENDING == "pending"
