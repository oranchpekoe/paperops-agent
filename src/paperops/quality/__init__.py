"""Deterministic quality checks for parsed paper artifacts."""

from paperops.quality.rules import QualityPolicy, evaluate_markdown

__all__ = ["QualityPolicy", "evaluate_markdown"]
