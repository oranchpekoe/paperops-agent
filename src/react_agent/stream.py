"""Streaming output support for the Multi-Mode Agent Framework.

Provides two streaming interfaces built on LangGraph's ``astream_events()``:

1. **``stream_events()``** — low-level event stream exposing every graph event
   (LLM token streaming, tool calls, node transitions).  Best for debugging,
   observability, and building custom UIs.

2. **``stream_tokens()``** — high-level async generator that yields only
   LLM-generated text tokens.  Best for simple "typewriter effect" frontends.

Both functions invoke the compiled graph from :file:`graph.py` and require
the same environment setup (``MODEL``, API keys, etc.) as the normal
``graph.ainvoke()`` path.

Usage
-----
.. code-block:: python

    from react_agent.stream import stream_tokens

    async for token in stream_tokens("What is the capital of France?"):
        print(token, end="", flush=True)

.. code-block:: python

    from react_agent.stream import stream_events

    async for event in stream_events("Calculate 15% of 200"):
        print(event)
"""

from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import HumanMessage

from react_agent.graph import graph


async def stream_events(query: str) -> AsyncIterator[dict]:
    """Stream ALL LangGraph events for *query*.

    Yields raw event dicts from ``graph.astream_events()``.  Events include:

    - ``on_chat_model_stream`` — LLM token streaming (``event["data"]["chunk"]``)
    - ``on_tool_start`` — tool execution begins
    - ``on_tool_end`` — tool execution completes
    - ``on_chain_start`` / ``on_chain_end`` — node transitions

    Parameters
    ----------
    query : str
        The user query to send to the agent.

    Yields
    ------
    dict
        Raw LangGraph event dicts.
    """
    config = {"configurable": {"thread_id": "stream"}}
    input_state = {"messages": [HumanMessage(content=query)]}

    async for event in graph.astream_events(input_state, config=config, version="v2"):
        yield event


async def stream_tokens(query: str) -> AsyncIterator[str]:
    """Stream LLM-generated text tokens only.

    Convenience wrapper around ``astream_events()`` that filters for
    ``on_chat_model_stream`` events and extracts the text content.

    Parameters
    ----------
    query : str
        The user query to send to the agent.

    Yields
    ------
    str
        Individual text tokens from the LLM's streaming output.
    """
    config = {"configurable": {"thread_id": "stream"}}

    # Use the same message format as normal graph invocations
    input_state = {
        "messages": [{"role": "user", "content": query}],
    }

    async for event in graph.astream_events(input_state, config=config, version="v2"):
        kind = event.get("event", "")

        # Only forward LLM streaming tokens
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield chunk.content
