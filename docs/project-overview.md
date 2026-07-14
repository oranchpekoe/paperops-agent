# Multi-Mode Agent Framework — 项目全景

> 最后一次更新：2026-06-08。覆盖 4 模式、MCP、RAG、三层记忆、流式输出、Eval 框架。

---

## 一分钟速览

你输入一句话 → 系统从长期记忆召回你的偏好 → LLM Router 判断问题类型 → 分派给 4 种子图之一 → Agent 执行推理/搜索/计算 → 自动提取值得记住的事实 → 返回答案。

```
User Query
    │
    ▼
[0] inject_memory      ← 从 Chroma 召回相关历史事实，注入上下文
    │
    ▼
[1] route_mode         ← LLM Router 分类：react / reflection / plan_solve / supervisor
    │
    ├─ react_agent          ← 简单问答、搜索、计算（ReAct 循环）
    ├─ reflection_agent     ← 写作、分析（Generate → Critique → Refine）
    ├─ plan_solve_agent     ← 多步骤规划（Plan → Execute → Aggregate）
    └─ supervisor_agent     ← 多专家协同（Supervisor → Researcher / Analyst / Executor）
    │
    ▼
[5] extract_memory     ← LLM 自动提取对话中的关键事实 → 存入 Chroma（跨会话持久）
    │
    ▼
Response
```

---

## 1. 总体架构

### 1.1 目录结构

```
src/react_agent/
├── graph.py           # 主编排器：Router + 记忆注入/提取 + 子图注册
├── state.py           # 统一 State schema（MainState，所有模式共享）
├── tools.py           # 工具注册中心（search / python_repl / retrieve / 记忆工具）
│                      #   + run_mini_react_loop() 公共 ReAct 循环
│                      #   + MCP 懒加载 + RAG Chroma 懒加载
├── memory.py          # 三层记忆：MemoryStore (Chroma) + extract_facts + compress_context
│                      #   + _BENCHMARK_SIGNALS + ContextVar benchmark_mode
├── mcp.py             # MCP 协议：解析配置 → 连接多 MCP Server → 返回 LangChain 工具
├── stream.py          # 流式输出：stream_events() / stream_tokens()
├── context.py         # 运行时配置（model, system_prompt, max_search_results）
├── utils.py           # load_chat_model(), resolve_model(), get_message_text()
├── prompts.py         # 默认 System Prompt
└── modes/
    ├── __init__.py
    ├── react.py       # ReAct 子图（最简单，2 节点 + 条件边）
    ├── reflection.py  # Reflection 子图（generate → reflect → refine 循环）
    ├── plan_solve.py  # Plan-Solve 子图（plan → execute_all → aggregate）
    └── supervisor.py  # Supervisor 子图（5 节点 + 2 组条件边，最复杂）
```

### 1.2 数据流

每一步的输入输出都通过 [MainState](src/react_agent/state.py) 传递。`messages` 字段使用 LangGraph 的 `add_messages` reducer（追加 + ID 去重），所有子图共享同一份消息历史。

```
__start__
  → inject_memory  (从 Chroma 召回历史事实，写 recalled_facts + SystemMessage)
  → router         (读 user_query，写 mode + route_reason)
  → [条件边]       (读 mode，路由到对应子图)
  → react_agent / reflection_agent / plan_solve_agent / supervisor_agent
  → extract_memory (读 messages + benchmark_mode，LLM 提取事实 → Chroma 存储)
  → __end__
```

---

## 2. Mode Router（入口分流器）

**位置**：[graph.py:60-107](src/react_agent/graph.py#L60-L107)

Router 是一个单次 LLM 调用，Prompt 定义了 4 种模式的适用场景和边界规则：

| 关键词 | 模式 | 典型场景 |
|--------|------|---------|
| "What is" / "Tell me" / "HOW to" | **react** | 事实查询、方法论咨询、闲聊 |
| "Write a" / "Analyze" / "Review" / "Is this correct" | **reflection** | 写作、代码审查、需要自批判的任务 |
| "Plan a trip" / "Solve this" / 明确步骤 | **plan_solve** | 旅行规划、数学题、多步骤执行 |
| "Research X and calculate Y" / 搜索+计算混合 | **supervisor** | 多领域复杂任务，需不同专家协作 |

**解析逻辑**：简单字符串匹配（`"supervisor" in raw` → supervisor, `"plan" in raw` → plan_solve, …），因为 Prompt 要求只输出一个单词。

---

## 3. 四种 Agent 模式

### 3.1 ReAct（Reasoning + Acting）

**位置**：[modes/react.py](src/react_agent/modes/react.py)

**图结构**（最简）：
```
__start__ → call_model ⇄ tools → __end__
```

**流程**：
1. `call_model`：LLM 绑定全部工具（`get_all_tools()`），决定调用哪个工具或直接回答
2. `tools`（ToolNode）：执行工具调用，结果返回
3. 循环直到 LLM 不再发出 tool_call → `__end__`

**适用**：~80% 的日常查询。单步搜索、简单计算、闲聊。

### 3.2 Reflection（生成 → 审视 → 修改）

**位置**：[modes/reflection.py](src/react_agent/modes/reflection.py)

**图结构**：
```
__start__ → generate → reflect ⇄ refine → __end__
```

**流程**：
1. `generate`：生成初稿（纯 LLM，无工具）
2. `reflect`：LLM 以严格评审者身份审视初稿，列出问题（或 PASS）
3. `refine`：根据评审意见重写
4. 循环回 `reflect`，直到 PASS 或达到 3 轮上限

**关键设计**：Reflection 模式**不绑定工具**——它做的是纯认知工作（批判、改进），而不是搜索或计算。

### 3.3 Plan-Solve（规划 → 执行 → 汇总）

**位置**：[modes/plan_solve.py](src/react_agent/modes/plan_solve.py)

**图结构**（线性）：
```
__start__ → plan → execute_all → aggregate → __end__
```

**流程**：
1. `plan`：LLM 分解用户问题为 3-5 个有序步骤
2. `execute_all`：顺序执行每个步骤。每步使用 `run_mini_react_loop()`（共享的 mini ReAct）进行工具调用（搜/算）。前序步骤的文本结果注入后续步骤的 prompt，供 LLM 参考
3. `aggregate`：所有步骤结果汇总为自然流畅的最终答案

**关键优化**（Bug 修复 #2, #4）：
- Plan Prompt 限制 3-5 步（防过度分解）
- 步骤间 `asyncio.sleep(3)`（防 API 限流）

### 3.4 Supervisor-Worker（多 Agent 协同）

**位置**：[modes/supervisor.py](src/react_agent/modes/supervisor.py)

**图结构**（最复杂，5 节点 + 2 组条件边）：
```
__start__ → supervisor_decide ──→ researcher ──→ supervisor_review
                │                → analyst    ──→      │
                │                → executor   ──→      │
                │                → __end__             │
                │                          ┌───────────┘
                │                          ▼
                │              supervisor_review ──→ (back to specialists or __end__)
                │
                └─────────────────────────────────────→ __end__
```

**3 个专家**：

| 专家 | 是否有工具 | 职责 |
|------|-----------|------|
| **Researcher** | ✅（偏 search） | 搜索互联网获取事实、数据 |
| **Analyst** | ❌（纯推理） | 分析、批判、评估已有信息 |
| **Executor** | ✅（偏 python_repl） | 执行计算、处理数据、代码运行 |

**决策层**（最核心的设计）：
- **主路径**：`model.with_structured_output(SupervisorDecision)` → Pydantic schema 从协议层约束 LLM 必须输出合法 JSON
- **Fallback 路径**：`model.ainvoke()` + `_parse_text_decision()` 正则解析（已修复 case-sensitivity bug）
- 每次 review 后决定：再委托专家 或 ANSWER（生成最终答案）
- 迭代上限 `MAX_SUPERVISOR_ITERATIONS = 5`

---

## 4. 工具生态系统

**位置**：[tools.py](src/react_agent/tools.py)

### 4.1 工具注册

`get_all_tools()` 是唯一的工具入口，所有模式都通过它获取工具列表：

```
get_all_tools()
  = TOOLS (内置 3 个)
  + _mcp_tools (MCP 协议懒加载，可 0-N 个)
  + memory tools (remember / recall / recall_all，可选)
```

### 4.2 内置工具

| 工具 | 类型 | 说明 |
|------|------|------|
| `search(query)` | async | Tavily 网络搜索 |
| `python_repl(code)` | sync | 沙箱化 eval()，仅安全内置函数 + math |
| `retrieve(query, k)` | async | Chroma 向量检索，从 docs/ 文档中召回 |

### 4.3 MCP 协议工具

**位置**：[mcp.py](src/react_agent/mcp.py)

- 配置源：`MCP_CONFIG` 环境变量（JSON 字符串 或 JSON 文件路径）
- 支持 stdio 和 HTTP transport
- 通过 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 连接
- **优雅降级**：单服务器失败 → 跳过该服务器（不阻塞启动）；整个包未安装 → 返回空列表
- 示例配置已提供 `mcp_demo_server.py`（含 add / word_count 两个工具）

### 4.4 RAG 文档检索

**位置**：[tools.py:166-300](src/react_agent/tools.py#L166-L300)

- 从 `docs/` 目录加载 `.txt` / `.md` / `.pdf`
- `RecursiveCharacterTextSplitter` 分块（chunk_size=1000, overlap=200）
- `OpenAIEmbeddings`（text-embedding-3-small）→ Chroma 向量存储，持久化至 `.chroma_db/`
- **懒加载单例**：首次 `retrieve()` 触发，后续调用零开销
- 复用同一 Chroma 基础设施做长期记忆（不同 collection：`rag_docs` vs `agent_memory`）

### 4.5 共享 Mini ReAct 循环

**位置**：[tools.py:307-354](src/react_agent/tools.py#L307-L354)

```python
async def run_mini_react_loop(model, tools, messages, *, max_rounds=3) -> list
```

被以下节点使用：
- `_supervisor_researcher`（supervisor.py）
- `_supervisor_executor`（supervisor.py）
- `_execute_all`（plan_solve.py）

消除了原本 3 处 ~50 行的重复代码。

---

## 5. 三层记忆管理

**位置**：[memory.py](src/react_agent/memory.py)

### 5.1 短期记忆（会话内）

LangGraph 内置的 `add_messages` reducer。
- 自动追加新消息
- 基于消息 ID 去重（同一 ID 只会保留一份）
- 所有模式共享

### 5.2 长期记忆（跨会话）

基于 Chroma `agent_memory` collection（与 RAG 的 `rag_docs` collection 物理隔离）。

**两个入库路径**：
1. **Agent 主动调用**：`remember(fact)` 工具
2. **自动提取**：对话结束后 `extract_facts()` 让 LLM 从对话中提取值得记住的事实（用户偏好、决策、个人背景）

**去重**：`store()` 方法默认 `dedup=True`，存储前用 `asimilarity_search_with_score` 检查 cosine distance < 0.05 的已有事实。

**两个召回路径**：
1. `recall(query)`：语义搜索
2. `recall_all()`：列出全部（按时间倒序）

### 5.3 摘要记忆（防上下文溢出）

`compress_context(messages, model, keep_last=10)`：
- 当 `len(messages) > keep_last * 2` 时触发
- 旧消息 → LLM 压缩为一个段落
- 最近 `keep_last` 条消息保持原文

### 5.4 Benchmark 隔离（三层防护）

防止 eval/benchmark 的合成数据污染长期记忆：

| 层级 | 位置 | 机制 |
|------|------|------|
| **Layer 1（主）** | `MainState.benchmark_mode` | Eval runner 设置 `benchmark_mode=True` → `extract_memory` 直接跳过 |
| **Layer 2** | `memory.py` ContextVar | `set_benchmark_mode(True)` → `remember()` 和 `extract_facts()` 拒绝操作 |
| **Layer 3（兜底）** | 硬编码 `_BENCHMARK_SIGNALS` | 14 个已知 benchmark 模式字符串匹配 |

---

## 6. 流式输出

**位置**：[stream.py](src/react_agent/stream.py)

两种接口，均基于 `graph.astream_events()`：

| 函数 | 返回 | 用途 |
|------|------|------|
| `stream_events(query)` | `AsyncIterator[dict]` | 完整事件流（LLM token / tool start-end / node transition）→ 调试、可观测性 |
| `stream_tokens(query)` | `AsyncIterator[str]` | 仅 LLM 文本 token → 前端打字机效果 |

---

## 7. Eval 评估框架

**位置**：`tests/benchmarks.py` + `tests/run_evals.py`

### 7.1 结构

- **14 个 BenchmarkCase**，分 5 个类别：
  - `routing` × 5：路由准确性
  - `tool_use` × 3：工具使用正确性
  - `quality` × 3：回答质量
  - `multi_step` × 2：多步推理深度
  - `memory` × 1：记忆功能
- 每个 case 定义 `expected_mode`、`expected_tools`、`forbidden_tools`、`expected_keywords`

### 7.2 评分（加权）

| 维度 | 权重 | 评估方式 |
|------|:----:|---------|
| Route | 30% | 实际路由 vs 期望路由 |
| Tools | 25% | 实际工具 vs 期望/禁用工具 |
| Quality | 25% | 回答中的期望关键词命中率 |
| Depth | 20% | 多步推理的步数 ≥ 期望 |

- 单 case 及格线：60%
- 输出：命令行报告（默认）+ `--json`（CI 友好）
- Eval runner 设置 `benchmark_mode=True`，确保不污染记忆

### 7.3 当前成绩

| 模型 | 通过率 | 均分 | 总耗时 |
|------|:---:|:---:|-----:|
| deepseek-v4-flash | 100% (14/14) | 87.6% | 912s |
| gpt-4o-mini | 92.9% (13/14) | 90.1% | 577s |

---

## 8. 关键设计决策

### 8.1 为什么是子图（Subgraph）而不是一个大 Prompt？

显式路由 + 独立子图：
- 每个模式有自己的 Prompt 和控制流，互不干扰
- Token 经济（不需要把 4 种模式的规则全塞进 context）
- 可单独调试、优化、扩展某个模式

### 8.2 为什么 Structured Output 还保留文本 Fallback？

`_get_decision()` 双路径保证框架的**模型无关性**：
- 好模型（支持 function calling）→ Pydantic Schema 强约束
- 差模型（不支持或 provider 不支持）→ 正则解析降级
- 实际效果：deepseek-v4-flash 和 gpt-4o-mini 在 Supervisor 模式都 100% 通过

### 8.3 为什么 Chroma 而不是 FAISS？

- Windows 兼容性：Chroma 纯 Python，FAISS 需 C++ 编译
- 持久化原生支持：`persist_directory` 一行参数
- 基础设施复用：RAG 和长期记忆共用同一 Chroma 实例（不同 collection）

### 8.4 为什么 Mini ReAct 循环要提取为公共函数？

原本 3 处代码完全相同（supervisor researcher、supervisor executor、plan_solve executor），修改一处容易遗漏另外两处。提取后：
- 单点维护（bug 修复只需改一处）
- 统一的 `max_rounds` 和错误处理
- 可独立 mock 测试

### 8.5 Benchmark 隔离为什么是三层？

单靠硬编码黑名单（Layer 3）不够：
- 新增 benchmark 需手动同步两处（graph.py + memory.py）
- `remember` 工具可以被 Agent 直接调用，绕过 graph 层的检测

三层防护：
- Layer 1（State）→ graph 层跳过，eval runner 明确标记
- Layer 2（ContextVar）→ 工具层独立检测，不依赖 State
- Layer 3（硬编码）→ 兜底，即使前两层失效也能拦截已知模式

---

## 9. 项目统计

| 指标 | 数值 |
|------|------|
| 源码行数 | 2,847（src/） |
| 测试行数 | 1,602（tests/） |
| Agent 模式 | 4 |
| 内置工具 | 6（search, python_repl, retrieve, remember, recall, recall_all） |
| MCP 支持 | 无限（热插拔协议） |
| 单元测试 | 73 |
| Benchmark | 14 case，5 维度，加权评分 |
| 已修复 Bug | 10 |
| 开发周期 | 2026.05–2026.06（~4 周） |

---

## 10. 速查：各文件改了会影响到谁

| 文件 | 影响的模块 |
|------|-----------|
| `state.py` | 所有（共享 State） |
| `tools.py` | 所有模式（统一工具入口）+ supervisor / plan_solve（mini ReAct） |
| `memory.py` | graph.py（inject/extract）+ tools.py（remember/recall 工具） |
| `mcp.py` | tools.py（懒加载 MCP tools） |
| `graph.py` | 入口编排，引用所有 modes |
| `stream.py` | 独立模块，引用 graph.py |
| `utils.py` | 所有模式（模型加载） |
| `context.py` | LangGraph Server 运行时配置 |

---

> **相关文档**：[简历与面试准备](resume.md) · [开发日志与Bug修复](development-log.md) · [缺口与改进](gaps-and-improvements.md) · [优化Sprint总结](optimization-summary.md)
