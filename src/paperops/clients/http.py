"""Shared helpers for strict JSON-over-HTTP service adapters."""

from __future__ import annotations

from typing import Any

import httpx

from paperops.clients.errors import ExternalServiceError


def response_detail(response: httpx.Response) -> str:
    """Return a compact upstream error without assuming a JSON response."""
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or response.reason_phrase

    if isinstance(payload, dict):
        for key in ("detail", "message", "error"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return str(payload)


def require_json_object(
    response: httpx.Response,
    *,
    service: str,
    error_type: type[ExternalServiceError],
    expected_status: int = 200,
) -> dict[str, Any]:
    """Validate the HTTP status and require an object-shaped JSON body."""
    if response.status_code != expected_status:
        raise error_type(
            f"{service} returned HTTP {response.status_code}: "
            f"{response_detail(response)}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise error_type(f"{service} returned a non-JSON response") from exc
    if not isinstance(payload, dict):
        raise error_type(f"{service} returned a non-object JSON response")
    return payload
