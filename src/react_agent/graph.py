"""Multi-mode Agent Framework — main orchestrator.

Three agent architectures (ReAct, Reflection, Plan-and-Solve) are implemented
as LangGraph subgraphs. A Mode Router classifies each user query and delegates
to the most appropriate mode.

Architecture:
    User Query → [Inject Memory] → [Mode Router] → Agent → [Extract Memory] → Response

Memory flows:
    Before routing: recall relevant facts from long-term storage, inject as context.
    After agent completes: auto-extract key facts, store for future sessions.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Literal, cast

# --- trace logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
_trace = logging.getLogger("trace")
_trace.setLevel(logging.INFO)

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import StateGraph

from react_agent.context import Context
from react_agent.modes.plan_solve import build_plan_solve_subgraph
from react_agent.modes.react import build_react_subgraph
from react_agent.modes.reflection import build_reflection_subgraph
from react_agent.modes.supervisor import build_supervisor_subgraph
from react_agent.state import MainState
from react_agent.utils import get_message_text, load_chat_model, resolve_model


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_last_user_message(state: MainState) -> str:
    """Return the last non-empty HumanMessage content, or ``""``."""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            text = get_message_text(msg).strip()
            if text:
                return text
    return ""

# ---------------------------------------------------------------------------
# Mode Router
# ---------------------------------------------------------------------------

ROUTER_PROMPT = """You are a query classifier. Analyze the user's message and decide which agent mode is best suited.

Available modes:
- react: Simple questions, factual lookups, casual conversation, single-step tool use.
- reflection: Tasks that benefit from self-critique — writing, analysis, code review, complex reasoning where the first answer is often imperfect.
- plan_solve: Multi-step problems requiring decomposition — math word problems, structured plans, anything with sequential dependencies.  Use this when the user wants YOU to produce the plan/answer (not just explain how).
- supervisor: Complex multi-faceted tasks that need different types of expertise working together — tasks requiring BOTH research AND computation, tasks mixing multiple domains, or multi-phase projects where different sub-problems need different approaches.

Rules of thumb:
- "What is X?" / "Tell me about Y" → react
- "Write a..." / "Analyze..." / "Review..." / "Is this correct?" → reflection
- "Plan a trip..." / "Solve this math problem..." / step-by-step outputs → plan_solve
- "Research X and calculate Y" / tasks mixing search + computation / complex multi-phase projects → supervisor
- "HOW to [do X]?" asking for methodology tips → react (this is a single factual lookup, not a planning task)
- "Make/Create a plan for [X]" asking for the actual plan → plan_solve

Respond with EXACTLY ONE WORD: react, reflection, plan_solve, or supervisor. No punctuation, no explanation."""


async def route_mode(state: MainState) -> Dict:
    """Classify the user query and set the mode."""
    _trace.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    _trace.info("📍 [1/5] ROUTER 节点被调用")
    _trace.info("   → 当前 messages 数量: %d", len(state.messages))

    fallback_model = os.environ.get("MODEL", "openai/deepseek-v4-flash")
    llm = load_chat_model(state.mode_router_model or fallback_model)

    last_user_msg = _get_last_user_message(state)

    _trace.info("   → 提取到的用户问题: %s", last_user_msg[:80] if last_user_msg else "(空)")

    response = await llm.ainvoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=last_user_msg),
    ])

    raw = str(response.content).strip().lower()
    mode: str
    if "supervisor" in raw or "super" in raw:
        mode = "supervisor"
    elif "plan" in raw:
        mode = "plan_solve"
    elif "reflect" in raw:
        mode = "reflection"
    else:
        mode = "react"

    _trace.info("   → LLM 路由决策: %s → mode='%s'", raw, mode)

    return {
        "mode": mode,
        "route_reason": raw,
        "user_query": last_user_msg,
        # 不追加任何消息——add_messages 是追加 reducer，
        # 旧消息（用户的 HumanMessage）会原样保留给子图使用。
    }



def _route_edge(state: MainState) -> Literal["react", "reflection", "plan_solve", "supervisor"]:
    _trace.info("📍 [2/5] 条件边路由 → '%s' 子图", state.mode)
    return cast(Literal["react", "reflection", "plan_solve", "supervisor"], state.mode)


# ---------------------------------------------------------------------------
# Memory injection (pre-routing)
# ---------------------------------------------------------------------------


async def inject_memory(state: MainState) -> Dict:
    """Recall relevant facts from long-term memory and inject them as context.

    Runs BEFORE the router so every mode benefits from recalled context.
    Degrades gracefully when Chroma is unavailable.
    """
    # Extract the user's query to use as the memory search key
    last_user_msg = _get_last_user_message(state)

    if not last_user_msg:
        return {"memory_status": "⏭️ 无用户消息，跳过记忆召回"}

    try:
        from react_agent.memory import _ensure_memory_loaded

        store = await _ensure_memory_loaded()
        if store is None:
            _trace.info("📍 [0/5] Memory: store unavailable — skipping recall")
            return {"memory_status": "⚠️ 记忆存储不可用（embedding API 无法访问）"}

        # ── Two-pass recall ──────────────────────────────────────────
        # Pass 1: strict threshold (0.6) — high precision.
        # Pass 2: relaxed threshold (0.75) — catches moderately related
        #   facts that use different phrasing (e.g. stored as "用户对
        #   三亚旅行感兴趣" but queried as "三亚有啥好玩的").
        #   Tagged separately so the LLM can weigh them appropriately.
        facts = await store.recall(last_user_msg, k=5)
        relaxed_facts: list[dict] = []

        if not facts:
            relaxed_facts = await store.recall(
                last_user_msg, k=5, score_threshold=0.75,
            )
            # Deduplicate against facts (which is empty here, but kept
            # for clarity if this logic ever changes).
            seen = {f["content"] for f in facts}
            relaxed_facts = [f for f in relaxed_facts if f["content"] not in seen]

        if not facts and not relaxed_facts:
            _trace.info("📍 [0/5] Memory: no relevant facts found")
            return {"memory_status": "🔍 未找到相关记忆"}

        # Build a compact context injection
        lines = ["[Long-term memory — relevant facts from previous conversations]"]
        all_facts = list(facts)
        recalled_count = len(facts)
        if relaxed_facts:
            lines.append("(部分匹配度较低，仅供参考)")
            for f in relaxed_facts:
                lines.append(f"• {f['content']}")
            all_facts.extend(relaxed_facts)
            recalled_count = len(all_facts)

        for i, f in enumerate(facts, 1):
            lines.append(f"{i}. {f['content']}")

        recalled_text = "\n".join(lines)
        _trace.info(
            "📍 [0/5] Memory: recalled %d strict + %d relaxed facts",
            len(facts), len(relaxed_facts),
        )

        preview = " | ".join(f["content"][:50] for f in all_facts)
        return {
            "recalled_facts": recalled_text,
            "memory_status": f"✅ 召回 {recalled_count} 条: {preview}",
            "messages": [SystemMessage(content=recalled_text)],
        }

    except ImportError:
        _trace.info("📍 [0/5] Memory: chromadb not installed — skipping recall")
        return {"memory_status": "⚠️ chromadb 未安装，跳过记忆召回"}
    except Exception as exc:
        _trace.info("📍 [0/5] Memory: recall failed (%s) — continuing", exc)
        return {"memory_status": f"❌ 记忆召回失败: {exc}"}


# ---------------------------------------------------------------------------
# Memory extraction (post-subgraph)
# ---------------------------------------------------------------------------


async def extract_memory(state: MainState) -> Dict:
    """Auto-extract key facts from the completed conversation and store them.

    Runs AFTER each subgraph completes.  Uses the LLM to identify facts worth
    remembering (user preferences, decisions, context) and persists them to
    Chroma for future sessions.

    Also triggers context compression when the message list is long.
    """
    status_parts: list[str] = []
    try:
        from react_agent.memory import _ensure_memory_loaded, compress_context, extract_facts

        store = await _ensure_memory_loaded()
        if store is None:
            status_parts.append("⚠️ 记忆存储不可用")
            return {"memory_status": " | ".join(status_parts)}

        # Only extract from conversations with meaningful content
        total_content = sum(len(str(m.content)) for m in state.messages)
        if total_content < 200:
            _trace.info("📍 [5/5] Memory: conversation too short — skipping extraction")
            status_parts.append("⏭️ 对话太短，跳过事实提取")
            return {"memory_status": " | ".join(status_parts)}

        # Skip extraction for benchmark/eval conversations
        if state.benchmark_mode:
            _trace.info("📍 [5/5] Memory: benchmark_mode=True — skipping extraction")
            status_parts.append("⏭️ 基准测试模式，跳过事实提取")
            return {"memory_status": " | ".join(status_parts)}
        from react_agent.memory import _looks_like_benchmark

        if _looks_like_benchmark(state.user_query):
            _trace.info("📍 [5/5] Memory: benchmark query detected — skipping extraction")
            status_parts.append("⏭️ 检测到基准测试查询，跳过事实提取")
            return {"memory_status": " | ".join(status_parts)}

        # Use a lightweight model for extraction (same as router fallback)
        fallback_model = os.environ.get("MODEL", "openai/deepseek-v4-flash")
        model = load_chat_model(fallback_model)

        facts = await extract_facts(
            state.messages, model, max_facts=5,
            user_query=state.user_query,
        )
        if facts:
            ids = await store.store_many(facts)
            _trace.info("📍 [5/5] Memory: auto-stored %d facts (ids=%s)", len(facts), ids)
            # Show fact content, not internal Chroma IDs
            preview = " | ".join(f[:50] for f in facts)
            status_parts.append(f"✅ 存储 {len(facts)} 条: {preview}")
        else:
            status_parts.append("📝 未提取到值得存储的事实")

        # ── Context compression: when messages exceed 50, compress older ──
        # half into a summary paragraph via LLM and replace them with a
        # single SystemMessage.  We only RemoveMessage the *old* half
        # (not the recent messages that are being kept) so that
        # "Deleted Message" entries don't leak into the user-visible
        # output stream.  The summary + recent messages become the new
        # state via the add_messages reducer.
        compress_threshold = 50
        if len(state.messages) > compress_threshold:
            try:
                compressed = await compress_context(state.messages, model)
                if len(compressed) < len(state.messages):
                    # compress_context keeps the last 10 messages verbatim
                    # and returns [SystemMessage(summary)] + recent_10.
                    # Only emit RemoveMessage for the truly-old messages
                    # (index 0 .. split-1), and only add the new summary.
                    keep_last = 10
                    split = len(state.messages) - keep_last
                    old_ids = [
                        m.id for m in state.messages[:split]
                        if hasattr(m, "id") and m.id is not None
                    ]
                    removed = len(old_ids)
                    _trace.info(
                        "📍 [5/5] Memory: compressed %d messages → %d "
                        "(%d old removed, %d recent kept)",
                        len(state.messages), len(compressed),
                        removed, keep_last,
                    )
                    status_parts.append(f"🗜️ 压缩 {removed} 条旧消息")
                    return {
                        "conversation_summary": str(compressed[0].content),
                        "memory_status": " | ".join(status_parts),
                        "messages": [RemoveMessage(id=mid) for mid in old_ids]
                                    + [compressed[0]],
                    }
            except Exception as exc:
                _trace.info("📍 [5/5] Memory: compression skipped (%s)", exc)
                status_parts.append(f"⚠️ 压缩失败: {exc}")

        return {"memory_status": " | ".join(status_parts)}

    except ImportError:
        return {"memory_status": "⚠️ chromadb 未安装"}
    except Exception as exc:
        _trace.info("📍 [5/5] Memory: extraction failed (%s) — continuing", exc)
        return {"memory_status": f"❌ 记忆提取失败: {exc}"}


# ---------------------------------------------------------------------------
# Build the main orchestrator graph
# ---------------------------------------------------------------------------

builder = StateGraph(MainState, context_schema=Context)

# -- memory nodes --
builder.add_node("inject_memory", inject_memory)
builder.add_node("extract_memory", extract_memory)

# -- subgraph nodes --
builder.add_node("react_agent", build_react_subgraph())
builder.add_node("reflection_agent", build_reflection_subgraph())
builder.add_node("plan_solve_agent", build_plan_solve_subgraph())
builder.add_node("supervisor_agent", build_supervisor_subgraph())

# -- router --
builder.add_node("router", route_mode)

# -- wiring --
# __start__ → inject_memory → router → [subgraph] → extract_memory → __end__
builder.add_edge("__start__", "inject_memory")
builder.add_edge("inject_memory", "router")
builder.add_conditional_edges("router", _route_edge, {
    "react": "react_agent",
    "reflection": "reflection_agent",
    "plan_solve": "plan_solve_agent",
    "supervisor": "supervisor_agent",
})
builder.add_edge("react_agent", "extract_memory")
builder.add_edge("reflection_agent", "extract_memory")
builder.add_edge("plan_solve_agent", "extract_memory")
builder.add_edge("supervisor_agent", "extract_memory")
builder.add_edge("extract_memory", "__end__")

# LangGraph Platform (langgraph dev / LangSmith Deployment) provides its
# own persistence layer — we must NOT pass a custom checkpointer or the
# platform will refuse to load the graph.  Detect the platform by checking
# whether ``LANGSMITH_LANGGRAPH_API_VARIANT`` is set (the CLI always sets
# it to ``"local_dev"``; it is never set during standalone scripts/pytest).
import os as _os
from langgraph.checkpoint.memory import MemorySaver

_checkpointer = (
    None
    if _os.environ.get("LANGSMITH_LANGGRAPH_API_VARIANT", "")
    else MemorySaver()
)

graph = builder.compile(
    name="MultiMode Agent",
    checkpointer=_checkpointer,
)
