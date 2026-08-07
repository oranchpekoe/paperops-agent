from pathlib import Path

from paperops import QualityVerdict
from paperops.quality import QualityPolicy, evaluate_markdown


def _policy() -> QualityPolicy:
    return QualityPolicy(
        min_characters=80,
        min_sections=1,
        max_replacement_character_ratio=0.01,
    )


def test_quality_rules_pass_a_well_formed_artifact(tmp_path: Path) -> None:
    markdown = tmp_path / "paper.md"
    markdown.write_text(
        "# Reliable parsing\n\n## Abstract\n\n"
        + "Evidence-backed research content. " * 8,
        encoding="utf-8",
    )

    decision = evaluate_markdown(markdown, _policy())

    assert decision.verdict is QualityVerdict.PASS
    assert decision.metrics is not None
    assert decision.metrics.section_count == 1


def test_quality_rules_retry_a_short_artifact(tmp_path: Path) -> None:
    markdown = tmp_path / "paper.md"
    markdown.write_text("# Incomplete", encoding="utf-8")

    decision = evaluate_markdown(markdown, _policy())

    assert decision.verdict is QualityVerdict.RETRY
    assert "document_too_short" in decision.issues


def test_quality_rules_review_a_missing_local_image(tmp_path: Path) -> None:
    markdown = tmp_path / "paper.md"
    markdown.write_text(
        "# Visual paper\n\n## Results\n\n"
        + "Substantive experimental evidence. " * 8
        + "\n\n![plot](images/missing.png)",
        encoding="utf-8",
    )

    decision = evaluate_markdown(markdown, _policy())

    assert decision.verdict is QualityVerdict.REVIEW
    assert "broken_image_references" in decision.issues
