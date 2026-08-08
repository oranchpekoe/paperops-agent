"""Stable system instructions for typed research-model calls."""

from __future__ import annotations

import json
from typing import Any

ASSESS_EVIDENCE = (
    "Judge whether the supplied evidence directly answers every material part of "
    "the question. Mark insufficient when claims would require outside knowledge."
)
REWRITE_QUERY = (
    "Write one concise retrieval query targeting the missing aspects. Do not "
    "repeat any attempted query."
)
SYNTHESIZE_ANSWER = (
    "Answer only from the supplied evidence. Put a marker such as [E1] "
    "immediately after each supported claim, list every used id in citation_ids, "
    "and disclose material limitations."
)


def system_prompt(purpose: str, schema: dict[str, Any]) -> str:
    """Combine stable instructions with the validated response schema."""
    return (
        f"{purpose}\nReturn exactly one JSON object matching this schema: "
        f"{json.dumps(schema, ensure_ascii=False)}"
    )
