"""MCP (Model Context Protocol) tool loader.

Loads external tools from MCP-compatible servers and converts them to
LangChain tools via ``langchain-mcp-adapters``.

Configuration is read from the ``MCP_CONFIG`` environment variable, which
can be a JSON string or a path to a JSON file:

.. code-block:: json

    {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "transport": "stdio"
        },
        "weather": {
            "url": "http://localhost:8000/mcp",
            "transport": "http"
        }
    }

If the package is not installed or a server is unreachable, MCP tools are
skipped gracefully — the agent continues with its built-in tools.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def has_mcp_config() -> bool:
    """Return ``True`` if ``MCP_CONFIG`` is set and non-empty."""
    return bool(os.environ.get("MCP_CONFIG", "").strip())


async def load_mcp_tools(config_raw: str) -> list:
    """Connect to MCP servers and return merged LangChain-compatible tools.

    Parameters
    ----------
    config_raw : str
        Either an inline JSON string or an absolute path to a JSON file
        describing the MCP servers (see module docstring for format).

    Returns:
    -------
    list
        LangChain tool objects from all reachable servers.  Servers that
        cannot be connected to are skipped with a warning.
    """
    servers = _parse_config(config_raw)
    if not servers:
        return []

    # Gracefully degrade if the package is not installed
    try:
        from langchain_mcp_adapters.client import (
            MultiServerMCPClient,  # type: ignore[import-untyped]
        )
    except ImportError:
        _logger.warning(
            "langchain-mcp-adapters is not installed — skipping MCP tools. "
            "Install it with: pip install langchain-mcp-adapters"
        )
        return []

    tools: list = []
    try:
        client = MultiServerMCPClient(servers)
        tools = await client.get_tools()
    except Exception as exc:
        _logger.warning("Failed to connect to MCP servers: %s", exc)

    return tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_config(raw: str) -> dict[str, Any]:
    """Parse *raw* into a server-config dict.

    Tries JSON-string first, then file-path, then returns ``{}``.
    """
    # 1. Try inline JSON
    try:
        config = json.loads(raw)
        if isinstance(config, dict):
            return config
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Try as file path
    expanded = os.path.expanduser(raw)
    if os.path.isfile(expanded):
        try:
            with open(expanded, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            _logger.warning("Failed to read MCP config file %s: %s", expanded, exc)
            return {}

    _logger.warning("MCP_CONFIG is set but could not be parsed as JSON or file path: %s", raw)
    return {}
