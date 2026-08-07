"""Apply deterministic quality rules before document indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from paperops.models import QualityDecision, QualityMetrics, QualityVerdict
from paperops.settings import Settings

_HEADING_PATTERN = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_SECTION_PATTERN = re.compile(r"^#{2,6}\s+\S", re.MULTILINE)
_TITLE_PATTERN = re.compile(r"^#\s+\S", re.MULTILINE)
_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Thresholds used by the deterministic Markdown quality gate."""

    min_characters: int
    min_sections: int
    max_replacement_character_ratio: float

    @classmethod
    def from_settings(cls, settings: Settings) -> QualityPolicy:
        """Build a policy from application settings."""
        return cls(
            min_characters=settings.min_markdown_characters,
            min_sections=settings.min_section_count,
            max_replacement_character_ratio=settings.max_replacement_character_ratio,
        )


def _count_broken_images(markdown_path: Path, text: str) -> int:
    """Count missing local image targets while ignoring remote and data URLs."""
    broken = 0
    for raw_target in _IMAGE_PATTERN.findall(text):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>\"'")
        if not target or target.startswith(("http://", "https://", "data:", "#")):
            continue
        if not (markdown_path.parent / target).is_file():
            broken += 1
    return broken


def evaluate_markdown(markdown_path: Path, policy: QualityPolicy) -> QualityDecision:
    """Return a deterministic retry, review, or pass decision for an artifact."""
    if not markdown_path.is_file():
        return QualityDecision(
            verdict=QualityVerdict.RETRY,
            confidence=1.0,
            issues=["artifact_missing"],
            retry_reason="artifact_missing",
            metrics=QualityMetrics(
                character_count=0,
                heading_count=0,
                section_count=0,
                broken_image_references=0,
                replacement_character_ratio=0.0,
            ),
        )

    text = markdown_path.read_text(encoding="utf-8", errors="replace")
    character_count = len(text.strip())
    replacement_count = text.count("\ufffd")
    replacement_ratio = replacement_count / max(len(text), 1)
    heading_count = len(_HEADING_PATTERN.findall(text))
    section_count = len(_SECTION_PATTERN.findall(text))
    broken_images = _count_broken_images(markdown_path, text)
    metrics = QualityMetrics(
        character_count=character_count,
        heading_count=heading_count,
        section_count=section_count,
        broken_image_references=broken_images,
        replacement_character_ratio=replacement_ratio,
    )

    retry_issues: list[str] = []
    if character_count < policy.min_characters:
        retry_issues.append("document_too_short")
    if replacement_ratio > policy.max_replacement_character_ratio:
        retry_issues.append("garbled_text")
    if retry_issues:
        return QualityDecision(
            verdict=QualityVerdict.RETRY,
            confidence=1.0,
            issues=retry_issues,
            retry_reason=retry_issues[0],
            metrics=metrics,
        )

    review_issues: list[str] = []
    if _TITLE_PATTERN.search(text) is None:
        review_issues.append("missing_title")
    if section_count < policy.min_sections:
        review_issues.append("insufficient_sections")
    if broken_images:
        review_issues.append("broken_image_references")
    if review_issues:
        return QualityDecision(
            verdict=QualityVerdict.REVIEW,
            confidence=0.5,
            issues=review_issues,
            metrics=metrics,
        )

    return QualityDecision(
        verdict=QualityVerdict.PASS,
        confidence=1.0,
        metrics=metrics,
    )
