# Multi-Mode Agent Framework — 简历与面试准备

> 以下是一份可直接用于简历投递、GitHub Profile 和面试准备的完整项目描述。
> 各部分可独立使用：简历用 **§1**，GitHub 用 **§2**，面试口述用 **§3**，深挖准备用 **§4**，自评弱点用 **§7**。

---

## §1 简历条目（精简版，~10 行）

**Multi-Mode Agent Framework** | Python, LangGraph, LangChain, Chroma | 2026.05–2026.06

- 设计并实现 **4 模式 Agent 框架**（ReAct / Reflection / Plan-Solve / Supervisor-Worker），基于 **LangGraph 子图架构**，由 LLM Router 自动分类用户意图并路由至最优执行模式
- 集成 **MCP 协议**实现工具的动态发现与热插拔，多 MCP Server 独立连接，单点故障优雅降级（失败 → 空列表，不阻塞启动）
- 实现 **RAG 文档检索管道**：Chroma 向量存储 + OpenAI Embeddings + 递归分块 + 持久化（单例懒加载，零重启成本）
- 构建 **Supervisor-Worker 多 Agent 协同系统**：Supervisor LLM 将复杂任务分解委派给 Researcher / Analyst / Executor 三个专家代理迭代执行，决策层采用 **Pydantic Structured Output**（+ 文本 fallback 双路径）
- 实现 **三层记忆管理体系**：短期（`add_messages` reducer + `MemorySaver` checkpointer 跨调用持久化）、长期（Chroma `agent_memory` collection + 两段向量召回 0.6/0.75 阈值 + LLM 自动事实提取）、摘要（滑动窗口 + LLM 压缩，50 条阈值触发）；LangSmith 全链路可视化（`memory_status` 字段实时反馈每节点操作内容）
- 设计 **14 项 Benchmark 评估框架**，覆盖路由准确性、工具使用、回答质量和多步推理 5 个维度，加权多维评分 + JSON 报告输出，全部通过（均分 87.6%）
- 解决 **ASGI 事件循环同步阻塞**（tiktoken → os.getcwd 调用链），采用 `asyncio.to_thread()` 实现异步兼容

---

## §2 GitHub README（英文版）

```markdown
# Multi-Mode Agent Framework

A production-oriented AI Agent framework built on **LangGraph** with four
execution modes, LLM-based routing, and a full tool ecosystem.

## Architecture

User Query → [Inject Memory] → [Mode Router] → Agent Subgraph → [Extract Memory] → Response

## Modes

| Mode | Best For | Key Feature |
|------|----------|-------------|
| **ReAct** | Factual Q&A, simple lookups | Tool-augmented reasoning loop |
| **Reflection** | Writing, analysis, code review | Self-critique → refine cycle (max 3) |
| **Plan-Solve** | Multi-step problems, planning | Decompose → execute each step → aggregate |
| **Supervisor** | Multi-domain tasks (search + compute) | Supervisor delegates to Researcher / Analyst / Executor |

## Features

- **MCP Protocol** — Dynamic tool loading via `langchain-mcp-adapters`, hot-pluggable, fault-tolerant
- **RAG** — Chroma persistent vector store, `.txt/.md/.pdf` ingestion, lazy-loading singleton
- **Memory** — 3-layer: short-term (session), long-term (Chroma cross-session), summary (LLM compression)
- **Structured Output** — Pydantic schema for supervisor decisions, with text-parsing fallback
- **Streaming** — `astream_events()` full event stream + `stream_tokens()` token-level output
- **Eval Suite** — 14 benchmarks across 5 categories, weighted scoring, JSON report

## Quick Start

```bash
pip install -e .
export MODEL=openai/deepseek-v4-flash
export TAVILY_API_KEY=your-key
python tests/run_evals.py
```

## Project Structure

src/react_agent/
├── graph.py          # Main orchestrator + router
├── state.py          # Shared state schema
├── tools.py          # Tool registry (search, python_repl, retrieve, memory)
├── memory.py         # MemoryStore + context compression + auto-extraction
├── mcp.py            # MCP client loader
├── stream.py         # Streaming output (events + tokens)
├── context.py        # Runtime configuration
├── utils.py          # Model loader, prompt resolver
└── modes/
    ├── react.py          # ReAct subgraph
    ├── reflection.py     # Reflection subgraph
    ├── plan_solve.py     # Plan-Solve subgraph
    └── supervisor.py     # Supervisor-Worker subgraph
```
```

---

## §3 面试口述稿（2 分钟，中文）

> 我做了一个多模式 AI Agent 框架，基于 LangGraph。它最核心的特点是：不是只有一个 ReAct 模式，而是四种模式——ReAct 做简单问答、Reflection 做需要自批判的写作分析、Plan-Solve 做多步骤规划、Supervisor-Worker 做需要多专家协同的复杂任务。入口处有一个 LLM Router 自动分类用户意图，路由到对应的子图。
>
> 技术上做了五件事：
> - MCP 协议集成，动态加载外部工具，单个服务器挂了不影响整体
> - RAG 文档检索，Chroma 向量存储持久化，支持从本地文档检索上下文
> - Supervisor-Worker 多 Agent 协同，监督者把任务分给三个专家迭代执行
> - 三层记忆管理——短期存会话（checkpointer 持久化）、长期跨会话用 Chroma（两段召回，精确→宽松兜底）、超长时自动压缩摘要（50 条触发）
> - 14 项 benchmark 评估框架，5 个维度量化评分，全部通过
>
> 最让我有收获的三个问题：
> 1. Supervisor 决策从文本解析迁移到 Pydantic Structured Output，从协议层强约束输出格式
> 2. ASGI 事件循环同步阻塞，排查到 tiktoken → os.getcwd 的调用链，用 asyncio.to_thread 解决
> 3. 记忆系统设计——复用了 RAG 的 Chroma 基础设施做长期记忆，零额外依赖，两个 collection 隔离；发现元事实（"用户对X感兴趣"）与直接查询（"X有啥好玩"）的 embedding 语义鸿沟，设计了两段召回（0.6→0.75）解决问题

---

## §4 面试深挖 Q&A（8 题完整版）

### Q1: 为什么有四种模式，而不是一个通用 Prompt 搞定？
**答**：通用 Prompt 有两个硬伤——一是 token 成本高（每次把所有模式的规则塞进 context），二是 LLM 容易在模式间漂移（自批判任务被当简单问答处理）。显式路由 + 独立子图，每个模式有自己的 prompt 和控制流，可控、可调试、可单独优化。

### Q2: Supervisor 的 Structured Output 是怎么实现的？模型不支持怎么办？
**答**：`_get_decision()` 函数做了双路径。主路径用 `model.with_structured_output(SupervisorDecision)`——LangChain 内部把 Pydantic schema 转换成 function calling 的 JSON Schema，从协议层约束 LLM 输出合法 JSON。如果模型或 provider 不支持（比如某些开源模型），catch 异常后走 fallback：普通 `model.ainvoke()` + `_parse_text_decision()` 正则解析。两路径保证：好模型用强约束，差模型也能跑。

### Q3: 三层记忆各自解决什么问题？
**答**：短期记忆（`add_messages` reducer + `MemorySaver` checkpointer）解决会话内连贯——同 `thread_id` 内的消息跨 `ainvoke` 调用持久化，Agent 知道上一轮说了什么。长期记忆（Chroma `agent_memory` collection + 两段向量召回）解决跨会话失忆——用户上周说的偏好这周还能召回。摘要记忆（`compress_context()`，50 条触发）解决上下文溢出——旧消息被 LLM 压缩成一段摘要段落，通过 `RemoveMessage` reducer 替换旧消息。

三个层级互补：短期记当前、长期记重要、摘要防溢出。入库有两个路径：Agent 主动调用 `remember` 工具，以及对话结束后 LLM 自动 `extract_facts`（含用户原始 query 确保不被中间消息挤出窗口）。

长期记忆最棘手的问题：存储的事实是元陈述（"用户对三亚旅行感兴趣"），而用户搜索查询是直接问题（"三亚有啥好玩的"）——两种文本 embedding 天然有距离。解决方案是**两段召回**：第一轮 0.6 阈值高精度搜索 → 空则第二轮 0.75 兜底，覆盖两类文本的语义差距，同时仍然拦截 0.85+ 的完全无关噪音。所有 memory 节点通过 `memory_status` 字段在 LangSmith 中实时可视化操作内容。

### Q4: MCP 工具和内置工具有什么区别？
**答**：对 Agent 来说没区别——都是 `get_all_tools()` 返回的列表里的 callable。区别在加载路径：内置工具在 `tools.py` 硬编码，MCP 工具通过 `MultiServerMCPClient` 从外部 MCP Server 动态拉取。懒加载、幂等、失败返回空列表不阻塞。新增工具零代码改动——启动一个 MCP Server，改个环境变量就行。

### Q5: RAG 为什么用 Chroma 而不是 FAISS？
**答**：Windows 兼容性和持久化。FAISS 在 Windows 上需要 C++ 编译，Chroma 纯 Python 开箱即用。而且 Chroma 原生支持 `persist_directory`——一行参数从内存模式切磁盘模式。持久化后索引不会因重启丢失。另外复用了同一套 Chroma 基础设施做长期记忆（不同 collection），零额外依赖。

### Q6: --allow-blocking 是设计缺陷吗？
**答**：不是。LangGraph dev server 基于 ASGI，在事件循环里检测同步阻塞调用。我的 `python_repl` 工具里的 `eval()` 是同步 CPU 密集型操作，丢线程池没意义。开发环境用 `--allow-blocking` 放行，生产部署用 `langgraph serve` 时每个 run 跑在独立 worker 里，天然不会阻塞事件循环。

### Q7: Eval 框架是怎么设计的？为什么不直接用 LangSmith？
**答**：14 个 `BenchmarkCase` 分 5 个类别。每个 case 定义 `expected_mode`、`expected_tools`、`forbidden_tools`、`expected_keywords`。评分四维加权：Route 30% + Tools 25% + Quality 25% + Depth 20%。60% 为通过线。输出支持命令行报告和 `--json`（CI 友好）。不用 LangSmith 是因为它是 SaaS 平台要花钱——对于一个实习项目，14 个手写 benchmark 足够证明质量意识，而且面试官更看重"你会设计评估标准"而非"你会用某个工具"。

### Q8: 这个项目跟 Ragflow / Dify 比有什么不同？
**答**：赛道不同。Ragflow/Dify 是 RAG 平台——核心是文档问答，优势在混合检索和可视化管理。我们是 Agent 框架——核心是多策略调度，Router 自动选模式 + MCP 热插拔工具 + Supervisor 多专家 + 记忆跨会话。RAG 在我们系统里只是其中一个工具。如果团队已经有 Python 服务需要嵌入智能 Agent，我们零运维成本接入；如果需要一站式文档问答平台，选 Ragflow。

---

## §5 项目统计速览

| 指标 | 数值 |
|------|------|
| 总代码行数 | 2,847（src/）+ 1,602（tests/）= **4,539** |
| Agent 模式 | 4（ReAct, Reflection, Plan-Solve, Supervisor） |
| 内置工具 | 6（search, python_repl, retrieve, remember, recall, recall_all） |
| MCP 外部工具 | 可无限扩展（已接入 demo server: add, word_count） |
| 测试数量 | **73**（5 模块 × 单元测试 + 4 integration + 3 stream） |
| Benchmark 通过率 | 100%（14/14，均分 87.6%） |
| 支持 LLM | deepseek-v4-flash, gpt-4o-mini（已验证），任何 OpenAI-compatible 模型 |
| 单请求 tokens（react） | ~20K（"山西有啥好吃" 实测） |
| 单请求耗时（react） | ~21s（"山西有啥好吃" 实测） |
| 记忆系统 | 三层（短期 Checkpointer / 长期 Chroma 两段召回 / 摘要 LLM 压缩），LangSmith 全节点可视化 |
| 流式输出 | `astream_events()` 完整事件 + `stream_tokens()` token 级 |
| 已修复 bug | **23**（记忆污染、上下文爆炸、工具滥用、步骤过多、重复存储、工具绕过、代码重复、硬编码黑名单、无 streaming、测试不足、过度截断 ×2、DRY 违规、benchmark 列表不一致、Supervisor 三 specialist 重复、checkpointer 缺失、压缩未接线、recall 无相似度阈值、Plan-Solve 记忆工具副作用、压缩 DeleteMessages 泄露、extract 对简单查询返回 0、memory 节点 LangSmith 空白、inject 元事实↔查询 embedding 鸿沟） |
| Python 版本 | 3.11+ |
| 开发周期 | 2026.05–2026.06（~4 周） |

### 关键技术栈速查

| 类别 | 技术 |
|------|------|
| 框架 | LangGraph 1.0+, LangChain |
| LLM | deepseek-v4-flash (via aihubmix, OpenAI-compatible API) |
| 搜索工具 | Tavily Search API |
| 向量存储 | Chroma (persistent — RAG + 长期记忆双 collection) |
| Embedding | text-embedding-3-small (OpenAI-compatible) |
| MCP | langchain-mcp-adapters 0.2.2, MultiServerMCPClient |
| 结构化输出 | Pydantic + with_structured_output() |
| 记忆管理 | Chroma 长期记忆 + LLM 摘要记忆 + add_messages 短期记忆 |
| 评估框架 | 14 项 benchmark（routing/tool_use/quality/multi_step/memory），加权评分 + JSON 报告 |
| 流式输出 | LangGraph astream_events() |
| 开发工具 | LangGraph Studio, langgraph dev, pytest |
| Python | 3.11+, asyncio, typing |

---

## §6 核心卖点：你能讲的故事

面试官最想听的不是"我用了什么技术"，而是 "你遇到了什么问题，怎么解决的"。这个项目至少能讲 **8 个**这样的故事：

| # | 故事 | 技术深度 |
|---|------|---------|
| 1 | "Supervisor 一开始总是直接返回 FINISH" → prompt 工程治标不治本 → 迁移到 Pydantic Structured Output 从协议层强约束 | Structured Output, JSON Schema |
| 2 | "Chroma.from_documents 在 ASGI 下触发 BlockingError" → 排查依赖链（tiktoken → tempfile → os.getcwd）→ asyncio.to_thread() + --allow-blocking 双保险 | ASGI 事件循环, 异步编程 |
| 3 | "MCP 工具不能阻塞启动流程" → 设计懒加载单例 + 单服务器失败不影响全局的降级策略 | MCP 协议, 优雅降级 |
| 4 | "Router 有时候把多领域任务分错模式" → prompt 增加边界 case → 看 routing 日志调优 → eval 框架量化准确率 | 路由设计, 评估方法论 |
| 5 | "deepseek-v4-flash 不遵循复杂 prompt 格式" → 简化 prompt + 加 few-shot → 最终用 structured output 根本性解决 | 模型行为理解, 工程取舍 |
| 6 | "Agent 重启就失忆，上下文超长就崩溃" → 设计三层记忆（短期 add_messages / 长期 Chroma / 摘要 LLM 压缩）→ 复用了 RAG 的 Chroma 基础设施，零额外依赖 | 系统设计, 资源复用 |
| 7 | "单次请求消耗 2.34M tokens、跑了 20 分钟" → LangSmith trace 追查到记忆污染 + 上下文爆炸 + 工具滥用 + 步骤过多 → 1 天 Sprint 修复 6 个 bug，tokens ↓94%，耗时 ↓75% | 性能诊断, 全链路优化 |
| 8 | "代码有 3 处重复的 mini ReAct 循环，benchmark 检测靠硬编码字符串，测试只有 3 个" → 半天内提取公共函数、引入 State + ContextVar 三层防护替代黑名单、补齐 70 个单元测试、新增流式输出 → 代码质量从 demo 级提升到可面试展示级 | 代码重构, 测试工程, 系统设计 |

---

## §7 弱点与补救

对自己项目的诚实评估，展示技术判断力。

| 弱点 | 状态 | 补救措施 |
|------|------|---------|
| ~~MCP 是空的（没有真实 server）~~ | ✅ 已解决 | 创建了 `mcp_demo_server.py`（Python MCP SDK 1.27.2，含 add/word_count 工具） |
| ~~RAG 用内存 Chroma（重启就丢）~~ | ✅ 已解决 | 改为持久化 Chroma + 环境变量 `CHROMA_PERSIST_DIR` |
| ~~没有记忆管理~~ | ✅ 已解决 | 实现三层记忆：短期（add_messages）、长期（Chroma 跨会话）、摘要（LLM 上下文压缩） |
| ~~没有 eval 框架~~ | ✅ 已解决 | 14 个 benchmark case，5 个维度，加权评分 + JSON 报告，通过率 100% |
| ~~只测了 deepseek 一个模型~~ | ✅ 已解决 | deepseek-v4-flash (100%, 91.6%) vs gpt-4o-mini (92.9%, 90.1%)，跨模型验证通过 |
| ~~测试覆盖严重不足（仅 3 个）~~ | ✅ 已解决 | 扩展至 73 个测试，覆盖 state/tools/routing/memory/stream |
| ~~benchmark 检测硬编码黑名单~~ | ✅ 已解决 | State `benchmark_mode` + ContextVar + 硬编码兜底，三层防护 |
| ~~无流式输出~~ | ✅ 已解决 | `stream.py`：`astream_events()` 完整事件 + `stream_tokens()` token 级 |
| ~~supervisor ReAct 循环代码重复~~ | ✅ 已解决 | `run_mini_react_loop()` 公共函数，消除 3 处重复 |
| Human-in-the-Loop 缺失 | 📋 已记录 | 时间不足，`docs/gaps-and-improvements.md` 中已记录方案 |
| eval() sandbox 非真正安全 | 📋 已记录 | 白名单方案对内用可接受，公开部署前需升级为 subprocess |
| 无结构化日志/可观测性 | 📋 已记录 | 当前 trace log 够用，生产环境需引入 structlog |
| Plan-Solve 无并行执行 | 📋 已记录 | 独立步骤可用 `asyncio.gather` 并发，已记录方案 |

---

## §8 双模型 Eval 对比

| 指标 | deepseek-v4-flash | gpt-4o-mini |
|------|:-:|:-:|
| 通过率 | **100%** (14/14) | 92.9% (13/14) |
| 平均分 | **91.6%** | 90.1% |
| 总耗时 | 1467s | **577s** |
| 路由准确率 | 100% (5/5) | 80% (4/5) |
| 工具使用 | 100% (3/3) | 66.7% (2/3) |
| 质量评分 | 72.5% | 77.7% |
| 多步推理 | 96.9% | 100% |
| 记忆 | 93.8% | 87.5% |

**关键发现**：
- gpt-4o-mini 快 2.5×，但对复杂指令的跟随能力弱于 deepseek-v4-flash（`tool-calculation` 路由错误：复利计算 → plan_solve 而非 react + python_repl）
- 两模型在 quality 维度都偏低（72-78%），说明 keyword-matching 评分对开放式问题偏严格——这本身就是评估框架的有效信号
- Supervisor 模式在两个模型上都表现完美（100%），证明 structured output 路径模型无关

---

## §9 落地场景与竞争分析

### 定位：轻量级 Agent 框架，非 RAG 平台

Ragflow / Dify 是**以 RAG 为核心的一站式平台**（Elasticsearch 混合检索 + 可视化 Pipeline + 多租户权限），我们不是替代它们——我们是另一个赛道的产品。

### 差异化优势

| 维度 | Ragflow / Dify | Multi-Mode Agent Framework |
|------|---------------|---------------------------|
| 核心能力 | RAG 文档问答 | **多策略 Agent 调度**（ReAct / Reflection / Plan-Solve / Supervisor） |
| 路由 | 手动选择 Pipeline | **LLM 自动分类** → 最优执行模式 |
| 工具扩展 | 平台内置，需等更新 | **MCP 协议热插拔**，写个 MCP Server 就行，不碰框架代码 |
| 多 Agent 协同 | ❌ 无 | Supervisor-Worker 多专家协作 |
| 部署 | Docker / K8s 独立部署 | `pip install` → `from react_agent import graph`，可嵌入现有 Python 服务 |
| 记忆 | 无跨会话记忆 | 三层记忆（短期 + 长期 + 摘要），跨会话保持上下文 |
| 适用深度 | 浅（单轮 RAG 问答） | 深（多步推理 + 自我批判 + 多专家协作） |

### 劣势（诚实承认，展示技术判断力）

1. **搜索质量**：Chroma 纯向量检索 vs Elasticsearch BM25 + 向量混合。短期可引入 BM25 retriever 做混合排序（RRF 融合）。
2. **文档解析**：仅支持 txt/md/pdf vs 20+ 格式。可选集成 `unstructured` 库扩展。
3. **权限体系**：无多租户/RBAC。对内部工具场景影响小，对 SaaS 产品需补。
4. **生态**：没有可视化 UI 和拖拽 Pipeline 构建器。

### 面试总结句

> "这不是又一个 RAG 平台。Ragflow/Dify 解决的是'如何从文档里找答案'，我们解决的是'如何为不同类型的问题选择最合适的 Agent 策略'。RAG 在我们系统里只是一个工具——Agent 还会搜索网络、执行代码、自我反思、多专家协作。如果企业已经有一堆文档但问答质量不够，选 Ragflow；如果企业需要一个能嵌入现有系统的智能 Agent 框架来处理各种类型的任务，选我们。"

---

> **相关文档**：[开发日志与Bug修复记录](development-log.md) · [缺口分析与改进方案](gaps-and-improvements.md) · [优化Sprint总结](optimization-summary.md)
