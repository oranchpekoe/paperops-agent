"""ReAct (Reasoning + Acting) agent subgraph.

The classic ReAct loop: the LLM decides whether to call a tool or return a
final answer.  Tools execute and results feed back into the model until no
more tool calls are requested (or ``recursion_limit`` is hit).

This is the simplest and fastest mode — ideal for factual queries,
single-step lookups, and casual conversation.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import MainState
from react_agent.tools import _ensure_mcp_loaded, get_all_tools
from react_agent.utils import load_chat_model, resolve_model, resolve_system_prompt

_trace = logging.getLogger("trace")


# ---------------------------------------------------------------------------
# Dynamic tool executor — tools are loaded lazily at runtime (MCP support).
# This replaces the static ``ToolNode(TOOLS)`` pattern.
# ---------------------------------------------------------------------------


async def _execute_tools(
    state: MainState, runtime: Runtime[Context]
) -> Dict[str, list]:
    """Execute tool calls from the last AIMessage using the full tool set."""
    await _ensure_mcp_loaded()
    tools = get_all_tools()
    tool_node = ToolNode(tools)
    result = await tool_node.ainvoke({"messages": [state.messages[-1]]})
    return {"messages": result["messages"]}


async def _call_model(
    state: MainState, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Call the LLM with tools bound.  One invocation per graph step."""
    _trace.info("📍 [3/5] ReAct._call_model — 当前 messages 数: %d", len(state.messages))

    await _ensure_mcp_loaded()
    tools = get_all_tools()
    model = load_chat_model(resolve_model(runtime)).bind_tools(tools)

    system_message = resolve_system_prompt(runtime).format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    # 根据当前的运行环境（runtime）获取一个系统提示模板。比如可能是字符串 "You are a helpful assistant. Current time: {system_time}"。

    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    has_tools = bool(response.tool_calls)
    if has_tools:
        _trace.info("   → LLM 决定调用工具: %s", [t["name"] for t in response.tool_calls])
    else:
        _trace.info("   → LLM 输出最终答案 (无工具调用) — 前60字: %s", str(response.content)[:60])

    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer in the specified number of steps.",
                )
            ]
        }

    return {"messages": [response]}


def _route(state: MainState) -> Literal["__end__", "tools"]:
    _trace.info("   → _route 判断: 最后一条消息类型=%s", type(state.messages[-1]).__name__)
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage, got {type(last_message).__name__}"
        )
    if not last_message.tool_calls:
        _trace.info("📍 [4/5] 无工具调用 → __end__ (流回主图 → 结束)")
        return "__end__"
    _trace.info("   → 有 %d 个工具调用 → 路由到 ToolNode", len(last_message.tool_calls))
    return "tools"


def build_react_subgraph() -> StateGraph:
    """Create a compiled ReAct subgraph.

    Returns a ``CompiledStateGraph`` that can be added as a node in a parent
    graph.  The subgraph shares the parent's ``MainState`` schema.
    """
    builder = StateGraph(MainState)

    builder.add_node("call_model", _call_model)
    builder.add_node("tools", _execute_tools)

    builder.add_edge("__start__", "call_model")
    builder.add_conditional_edges("call_model", _route)
    builder.add_edge("tools", "call_model")

    return builder.compile(name="ReAct")
