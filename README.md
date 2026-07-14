# Multi-Mode Agent Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-green.svg)](https://github.com/langchain-ai/langgraph)

基于 LangGraph 的多模式 AI Agent，集成**模式路由**、**MCP 协议**、**RAG 文档检索**、**Supervisor 多 Agent 协同**及**三层记忆系统**。AI Agent 工程实习面试展示项目。

---

## 架构概览

```
用户输入
    │
    ▼
┌──────────────────┐
│  0. 记忆召回       │  ← 从 Chroma 向量库召回历史相关事实，注入上下文
│  (inject_memory) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  1. 模式路由       │  ← LLM 分析用户意图，四选一
│  (mode_router)   │
└────────┬─────────┘
         │
    ┌────┼────┬──────────┐
    │    │    │          │
    ▼    ▼    ▼          ▼
┌────┐ ┌───┐ ┌─────┐ ┌──────────┐
│ReAct││反思││规划求解││ Supervisor│
│简单 ││写作││多步骤 ││  多Agent  │
│问答 ││分析││任务   ││   协同    │
└──┬─┘ └─┬─┘ └──┬──┘ └────┬─────┘
    │    │    │          │
    └────┼────┴──────────┘
         │
         ▼
┌──────────────────┐
│  5. 记忆提取       │  ← LLM 自动提取关键事实 → 存 Chroma
│  (extract_memory)│     消息 > 50 条时自动压缩旧上下文
└────────┬─────────┘
         │
         ▼
      返回答案
```

**实际流程**：`注入记忆 → 路由分类 → 子图执行 → 提取记忆 → 返回`，共 5 步。记忆系统在每次对话前后自动运行。

### 四种模式对比

| 模式 | 适用场景 | 工作方式 | 工具 |
|------|----------|----------|------|
| **ReAct** | 简单问答、事实查询、搜索 | Reason → Act → Observe → 循环 | ✅ 全部 |
| **反思** | 写作、分析、代码审查 | 生成 → 自我批判 → 改进（最多 3 轮） | ❌ 纯推理 |
| **规划求解** | 多步骤任务、数学题、旅行规划 | 分解步骤 → 逐步执行 → 汇总 | ✅ 全部 |
| **Supervisor** | 跨领域任务（搜索+计算） | 主管决策 → 委派专家 → 审查 → 循环 | ✅ 按专家分配 |

---

## 核心特性

### 1. LLM 模式路由

在图的入口处，用 LLM 分析每条用户消息的语义，自动选择最合适的 Agent 架构。不是正则匹配——同一个 LLM 判断"这个问题的本质是什么"，然后路由到对应的子图。

### 2. 三层记忆系统

| 层级 | 存储位置 | 生命周期 | 做什么 |
|------|----------|----------|--------|
| **短期** | `MemorySaver` checkpointer | 同一 `thread_id` 内 | 当前会话的消息历史 |
| **长期** | Chroma 向量库（`.chroma_db/`） | 跨会话持久化 | 用户偏好、决策、背景等事实 |
| **摘要** | 压缩后替换旧消息 | 同 thread 内 | 消息超过 50 条时 LLM 压缩旧上下文 |

- **写入**：每次对话后，LLM 自动提取值得记住的事实 → embedding → 存入 Chroma → 去重检查
- **读取**：每次对话前，用用户当前问题去 Chroma 做语义搜索 → 注入 prompt
- **压缩**：消息过长时保留最近 10 条，其余压缩为一段摘要段落

### 3. MCP 协议集成

通过 `MCP_CONFIG` 环境变量动态加载外部工具服务器：

```json
{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    "transport": "stdio"
  }
}
```

- 懒加载：首次使用时连接，后续调用复用
- 单服务器故障不影响其他服务器（降级）
- 零配置：不设 `MCP_CONFIG` 时行为不变

### 4. RAG 文档检索

将 `.txt` / `.md` / `.pdf` 文件放入 `docs/` 目录，Agent 即可检索：

- **Chroma** 向量存储（本地持久化到 `.chroma_db/`）
- **OpenAI text-embedding-3-small** 做向量化
- **RecursiveCharacterTextSplitter** 分块（1000 字/块，200 字重叠）
- 懒加载单例模式：文档索引一次，所有查询共享

### 5. Supervisor 多 Agent 协同

```
Supervisor（主管 LLM）
    ├── Researcher（研究员）— 搜索收集信息
    ├── Analyst（分析师）— 纯推理，无工具
    └── Executor（执行者）— python_repl 计算
```

执行循环：**决策 → 委派专家 → 审查结果 → 再委派或结束**（最多 5 轮迭代）。

举例——"查日本 2024 年 GDP 再算它的 5%"：
```
Router → supervisor
  决策 → RESEARCH  → Researcher 搜索 GDP 数据
  审查 → EXECUTE   → Executor 计算 5%
  审查 → ANSWER    → Supervisor 汇总最终答案
```

### 6. 流式输出 + 基准测试

- **流式输出**：`stream.py` 提供 token 级别和事件级别的流式接口
- **双模型 Eval**：14 个场景 × 2 个 LLM，对比答案质量
- **基准测试**：100% 通过率（14/14），平均分 87.6%

---

## 项目结构

```
react-agent/
├── src/react_agent/
│   ├── graph.py              # 主编排器（路由 + 记忆 + 4 子图注册）
│   ├── state.py              # 统一 State Schema（MainState）
│   ├── tools.py              # 工具注册中心（search / python_repl / retrieve / MCP）
│   ├── memory.py             # 三层记忆：MemoryStore + extract_facts + compress_context
│   ├── mcp.py                # MCP 客户端封装
│   ├── stream.py             # 流式输出
│   ├── context.py            # 运行时配置
│   ├── prompts.py            # 各模式 System Prompt
│   ├── utils.py              # 模型加载辅助
│   └── modes/
│       ├── react.py          # ReAct 子图
│       ├── reflection.py     # 反思子图（生成→批判→改进）
│       ├── plan_solve.py     # 规划求解子图（规划→执行→汇总）
│       └── supervisor.py     # Supervisor 子图（多 Agent 协同）
├── docs/                     # RAG 文档 + 项目文档
├── tests/
│   ├── test_trace.py         # 端到端链路测试（4 种查询 × 2 模型）
│   ├── benchmarks.py         # 双模型 Eval 框架
│   ├── run_evals.py          # Eval 运行器
│   ├── unit_tests/           # 单元测试
│   └── integration_tests/    # 集成测试
├── scripts/                  # 面试文档构建脚本
├── mcp_demo_server.py        # MCP 演示服务器
├── pyproject.toml
├── .env.example
└── langgraph.json
```

---

## 快速开始

### 前置条件

- Python 3.11+
- [Tavily API Key](https://tavily.com)（网页搜索用）
- LLM API Key（支持 OpenAI 兼容接口）

### 安装配置

```bash
# 1. 进入项目
cd react-agent

# 2. 安装依赖（推荐 uv）
pip install uv
uv sync

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 4. （可选）安装 MCP 支持
pip install langchain-mcp-adapters

# 5. （可选）放入 RAG 文档
mkdir docs
echo "# 我的知识库" > docs/notes.md
```

### 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | ✅ | LLM API Key（aihubmix 或其他 OpenAI 兼容接口） |
| `OPENAI_BASE_URL` | ✅ | API 基础 URL，默认 `https://aihubmix.com/v1` |
| `MODEL` | ✅ | 模型名，如 `openai/deepseek-v4-flash` |
| `TAVILY_API_KEY` | ✅ | Tavily 搜索 API Key |
| `MCP_CONFIG` | ❌ | MCP 服务器配置（JSON 字符串或文件路径） |
| `EMBEDDING_MODEL` | ❌ | 向量化模型，默认 `text-embedding-3-small` |
| `CHROMA_PERSIST_DIR` | ❌ | Chroma 持久化目录，默认 `.chroma_db/` |

### 运行

**LangGraph Studio（开发推荐）：**
```bash
set PYTHONUTF8=1
langgraph dev --port 1024 --allow-blocking
```

**直接调用：**
```python
from react_agent.graph import graph

result = await graph.ainvoke({
    "messages": [{"role": "user", "content": "法国的首都是哪？"}]
})
print(result["messages"][-1].content)
```

---

## 测试

```bash
# 端到端链路测试
python tests/test_trace.py

# 单元测试
pytest tests/unit_tests/ -v

# 集成测试
pytest tests/integration_tests/ -v

# 双模型基准测试
python tests/run_evals.py
```

---

## 设计决策

### 为什么用 `--allow-blocking`？

`langgraph dev` 的 ASGI 服务器会检测同步阻塞调用以保护事件循环。`python_repl` 工具使用 `eval()`、Chroma 内部调用 `tiktoken` → `os.getcwd()`，都会触发阻塞检测。已将 Chroma 调用封装为 `asyncio.to_thread()`，但 `eval()` 是纯 CPU 计算，不适用线程化。`--allow-blocking` 是 LangGraph 为此场景设计的 escape hatch。生产环境用 `langgraph serve` + 独立 worker 部署不受此限制。

### 为什么所有模式共享同一套工具？

不按模式限制工具，而是通过 System Prompt 引导 LLM 判断该用哪些工具。Reflection 模式除外——它是纯推理循环，不调工具。这样做的好处是工具注册简单，LLM 可灵活判断（比如 Researcher 搜索过程中可能需要 `python_repl` 做简单计算）。

---

## License

MIT
