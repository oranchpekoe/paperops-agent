"""Day 1 trace test — run 3 different queries to see the lifecycle."""
import asyncio
import logging
import sys

from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Load .env before anything else
load_dotenv()

from react_agent.graph import graph

# 从本地的react_agent/graph导入graph类


# --- Configure logging to be readable ---
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(message)s",
    stream=sys.stdout,
)


async def run_query(label: str, query: str):
    print(f"\n{'='*70}")
    print(f"🧪 {label}")
    print(f"📝 Query: {query}")
    print(f"{'='*70}")

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": f"trace-{label}"}},
    )

    # Show final answer (last message)
    msgs = result.get("messages", [])
    if msgs:
        last = msgs[-1]
        content = last.content if hasattr(last, "content") else str(last)
        print(f"\n✅ 最终答案 (前200字): {str(content)[:200]}")
    print(f"📊 总消息数: {len(msgs)}")
    print(f"🔀 使用的模式: {result.get('mode', 'N/A')}")

    return result


async def main():
    # 1. Simple factual — expect react
    await run_query("测试 1 — 简单事实", "What is the capital of France?")

    # 2. Writing/Analysis — expect reflection
    await run_query("测试 2 — 写作/分析", "Write a short analysis: is Python good for AI development?")

    # 3. Planning — expect plan_solve
    await run_query("测试 3 — 规划任务", "Plan a 3-day trip to Tokyo for a first-time visitor")

    # 4. Multi-domain — expect supervisor
    await run_query("测试 4 — 多领域协同", "Research the GDP of Japan in 2024 and calculate what 5% of it is.")


if __name__ == "__main__":
    asyncio.run(main())
