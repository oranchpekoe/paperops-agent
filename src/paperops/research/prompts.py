"""Stable system instructions for typed research-model calls."""

from __future__ import annotations

import json
from typing import Any

ASSESS_EVIDENCE = (
    "Judge whether the supplied evidence directly answers every material part of "
    "the question. Mark insufficient when claims would require outside knowledge. "
    "Select only the minimal directly relevant evidence IDs; do not select a chunk "
    "merely because it is topically related. Return sufficient, confidence, "
    "rationale, missing_aspects, and relevant_citation_ids directly at the root of "
    "the JSON object."
)
REWRITE_QUERY = (
    "Write one concise retrieval query targeting the missing aspects. Do not "
    "repeat any attempted query. Return exactly the data instance "
    '{"query":"focused search text","reason":"why this targets the gap"}.'
)
SYNTHESIZE_ANSWER = (
    "Answer only from the supplied evidence. Put a marker such as [E1] "
    "immediately after each supported claim, list every used id in citation_ids, "
    "and disclose material limitations. Return text, citation_ids, and limitations "
    "directly at the root of the JSON object."
)
EXTRACT_COMPARISON = (
    "Extract every requested comparison dimension for exactly one document. "
    "Use only the supplied evidence. For a supported cell, write a concise claim "
    "with inline markers such as [E1] and list only the evidence IDs used. For a "
    "missing cell, return no claim or citations, explain the exact evidence gap, "
    "and propose one focused document-scoped retrieval query. Return the document_id "
    "and exactly one cell for each requested dimension; never use outside knowledge."
)


def system_prompt(purpose: str, schema: dict[str, Any]) -> str:
    """Combine stable instructions with the validated response schema."""
    return (
        f"{purpose}\nReturn exactly one data instance matching this schema: "
        f"{json.dumps(schema, ensure_ascii=False)}\nDo not echo or wrap the schema. "
        "The root object must contain the response data fields themselves; never "
        "return schema keywords such as description, properties, required, title, "
        "type, or $defs."
    )
