"""Minimal MCP demo server — exposes two tools via stdio transport.

This server is designed to be launched by the MCP client (langchain-mcp-adapters)
as a subprocess.  Communication happens over stdin/stdout using the MCP JSON-RPC
protocol.

Tools exposed:
- add(a, b)        — add two numbers
- word_count(text) — count words in a string
"""

import asyncio

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("demo-tools")


# ---------------------------------------------------------------------------
# Tool definitions (visible to the LLM)
# ---------------------------------------------------------------------------


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Return the catalogue of tools this server provides."""
    return [
        types.Tool(
            name="add",
            description="Add two numbers together. Use this for arithmetic addition.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        types.Tool(
            name="word_count",
            description="Count the number of words in a text string.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyse"},
                },
                "required": ["text"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool execution handler
# ---------------------------------------------------------------------------


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute the requested tool and return its result."""
    if name == "add":
        result = float(arguments["a"]) + float(arguments["b"])
        return [types.TextContent(type="text", text=str(result))]

    if name == "word_count":
        count = len(str(arguments["text"]).split())
        return [types.TextContent(type="text", text=str(count))]

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
