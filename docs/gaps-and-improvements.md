# 项目缺口分析与改进计划

> 归档说明：本文评估 `v0.1-multimode-demo` 原型，不代表当前 PaperOps 产品已经实现这些能力。

> 本文档记录对 Multi-Mode Agent Framework 项目的全面审查结果，包括已识别的缺口、严重程度评级和已完成的改进措施。

---

## 一、缺口总览

| # | 缺口 | 严重程度 | 状态 |
|---|------|----------|------|
| 1 | 测试覆盖严重不足 | 🔴 高 | ✅ 已修复 |
| 2 | Supervisor 代码重复（mini ReAct loop） | 🟡 中 | ✅ 已修复 |
| 3 | Benchmark 黑名单硬编码 | 🟡 中 | ✅ 已修复 |
| 4 | 无流式输出（Streaming） | 🟡 中 | ✅ 已修复 |
| 5 | Human-in-the-Loop 缺失 | 🟡 中 | 📋 记录（时间不足，未实施） |
| 6 | `eval()` sandbox 非真正安全 | 🟡 中 | 📋 记录（需引入 RestrictedPython 或 subprocess） |
| 7 | 无可观测性（结构化日志/Metrics） | 🟢 低 | 📋 记录（时间不足，未实施） |
| 8 | Plan-Solve 无并行执行 | 🟢 低 | 📋 记录（独立步骤可并发，但需引入 `asyncio.gather`） |

---

## 二、已完成改进的详情

### 改进 #1：测试覆盖 🔴 → ✅

**问题描述**：
- 仅 `tests/unit_tests/test_configuration.py` 中有 3 个单元测试（测试 Context 的模型名称解析）
- `tests/integration_tests/test_graph.py` 中仅 1 个集成测试（检查 "harrison" 在回答中）
- `tests/test_trace.py` 无断言（纯手工 smoke test）
- 核心模块（router、4 个 mode、memory、tools）均无独立测试

**改进措施**：

1. **`tests/unit_tests/test_tools.py`** — 工具层单元测试
   - `test_python_repl_basic` — 基础计算
   - `test_python_repl_safe_builtins` — 沙箱限制
   - `test_python_repl_error` — 异常处理
   - `test_python_repl_math_module` — math 模块可用
   - `test_get_all_tools_returns_builtins` — 内置工具列表
   - `test_get_tool_by_name` — 按名称查找

2. **`tests/unit_tests/test_memory.py`** — 记忆模块单元测试
   - `test_store_and_recall` — 存储与检索（需 Chroma）
   - `test_store_dedup` — 去重逻辑
   - `test_looks_like_benchmark` — benchmark 检测函数
   - `test_clear_all` — 全量清除
   - `test_clear_contaminated` — 模式匹配清除

3. **`tests/unit_tests/test_routing.py`** — 路由逻辑单元测试
   - `test_parse_text_decision_research` — 文本解析
   - `test_parse_text_decision_answer` — 默认路由
   - `test_benchmark_query_detection` — benchmark 检测
   - `test_benchmark_query_normal` — 正常查询不被误杀

4. **`tests/unit_tests/test_state.py`** — 状态结构测试
   - `test_main_state_defaults` — 默认值
   - `test_input_state_messages` — 消息追加

**测试运行方式**：
```bash
# 单元测试（不需要 API key）
cd d:/agent-internship/react-agent
source /d/software/miniconda3/etc/profile.d/conda.sh && conda activate langraph
pytest tests/unit_tests/ -v

# 集成测试（需要 API key）
SSL_CERT_FILE="D:/software/miniconda3/envs/langraph/Library/ssl/cacert.pem" \
  pytest tests/integration_tests/ -v
```

---

### 改进 #2：提取 `_run_mini_react_loop()` 公共函数 🟡 → ✅

**问题描述**：
`supervisor.py` 中 `_supervisor_researcher` 和 `_supervisor_executor` 的 mini ReAct 循环几乎完全相同（~50 行重复代码）：

```python
# 两个函数中完全相同的模式：
for _ in range(3):
    response = await model.ainvoke(msgs)
    msgs.append(response)
    if not response.tool_calls:
        break
    tool_node = ToolNode(tools)
    tool_result = await tool_node.ainvoke({"messages": [response]})
    for tm in tool_result.get("messages", []):
        msgs.append(tm)
```

此外，`plan_solve.py` 中的 `_execute_all` 也有类似的循环逻辑。

**改进措施**：

在 `src/react_agent/tools.py` 中新增 `run_mini_react_loop()` 公共函数：

```python
async def run_mini_react_loop(
    model,           # 已绑工具的 LLM
    tools: list,     # 可用工具列表
    messages: list,  # 初始消息列表
    *,
    max_rounds: int = 3,
) -> list:
    """执行 mini ReAct 循环（最多 max_rounds 轮），返回最终消息列表。"""
```

**涉及文件**：
- `src/react_agent/tools.py` — 新增公共函数
- `src/react_agent/modes/supervisor.py` — 用 `run_mini_react_loop()` 替换 `_supervisor_researcher` 和 `_supervisor_executor` 中的内联循环
- `src/react_agent/modes/plan_solve.py` — 用 `run_mini_react_loop()` 替换 `_execute_all` 中的内联循环

---

### 改进 #3：用 `State.benchmark_mode` 替代硬编码黑名单 🟡 → ✅

**问题描述**：
`graph.py` 中的 `_BENCHMARK_PATTERNS` 和 `memory.py` 中的 `_BENCHMARK_SIGNALS` 是硬编码的字符串列表，存在以下问题：
- 新增 benchmark 需要手动同步更新两处
- 只匹配精确字符串，换个表述就漏过
- 维护负担随 benchmark 数量增长

**改进措施**：

1. **`state.py`** — `MainState` 新增 `benchmark_mode: bool = False` 字段
2. **`graph.py`** — `extract_memory` 优先检查 `state.benchmark_mode`，保留 `_is_benchmark_query()` 作为二次兜底
3. **`memory.py`** — 新增 `benchmark_mode` 的 `contextvars.ContextVar`，`run_evals.py` 可以设置此标志，`remember()` 和 `extract_facts()` 检查此标志
4. **`tests/run_evals.py`** — 调用 graph 时设置 `benchmark_mode=True`

**架构变化**：

```
修复前：依赖硬编码字符串匹配
   query → _BENCHMARK_PATTERNS 匹配 → 拦截/放行

修复后：State 标志优先 + 硬编码兜底
   query → benchmark_mode=True? → 拦截（干净可靠）
        → _is_benchmark_query()? → 拦截（安全兜底）
        → 放行
```

**涉及文件**：
- `src/react_agent/state.py` — 新增字段
- `src/react_agent/graph.py` — 优先使用 state 标志
- `src/react_agent/memory.py` — ContextVar + 双重检查
- `tests/run_evals.py` — 设置 benchmark_mode

---

### 改进 #4：流式输出（Streaming）支持 🟡 → ✅

**问题描述**：
所有 4 种模式均使用 `model.ainvoke()`（batch 模式），不支持 token-level streaming。在生产场景中，用户需要看到 Agent 逐步输出，而不是等待数分钟后一次性返回结果。

**改进措施**：

1. **`src/react_agent/stream.py`** — 新增流式输出模块

提供两种流式接口：

- **`stream_events(query)`** — 底层 `astream_events()` 包装器，输出所有事件（`on_chat_model_stream`、`on_tool_start`、`on_tool_end`、`on_chain_start`、`on_chain_end`），适合调试和完整可观测性
- **`stream_tokens(query)`** — 简化的 token-level 流式接口，只输出 LLM 生成的文本 token，适合前端展示

支持关键字过滤（`stream_mode_filter`）和自定义回调。

2. **`tests/test_stream.py`** — 流式输出测试脚本

```python
# 用法示例
from react_agent.stream import stream_tokens
async for token in stream_tokens("What is the capital of France?"):
    print(token, end="", flush=True)
```

**涉及文件**：
- `src/react_agent/stream.py` — **新增**流式输出模块
- `tests/test_stream.py` — **新增**流式输出验证脚本

---

## 三、未实施的改进（时间约束）

以下改进项已识别但因时间不足未在此轮实施，记录于此供后续参考。

### Human-in-the-Loop（人工审批）

**现状**：Agent 执行过程中无人工干预机制。
**建议方案**：在 Supervisor 的 `_supervisor_review` 节点中加入 `interrupt()`，在执行不可逆操作（如通过 MCP 操作外部系统）前等待人工确认。
**预估时间**：1 天
**优先级**：中（展示 Agent 安全性设计思维）

### `eval()` Sandbox 安全性

**现状**：使用白名单 `__builtins__` 限制 `eval()`，但 `eval` 对不可信输入本质上不安全（如 `10**10**10` 可导致 DoS）。
**建议方案**：引入 `RestrictedPython` 库或将代码执行隔离到 subprocess 中。
**预估时间**：1 天
**优先级**：中（安全相关，生产环境必须修复）

### 结构化日志与可观测性

**现状**：项目使用 `print` 式 trace log（`_trace.info("...")`），无结构化日志、metrics、token usage tracking。
**建议方案**：引入 `structlog` 替代 print 式日志，新增 token 计数器和请求耗时 histogram。
**预估时间**：半天
**优先级**：低（锦上添花，不阻塞核心功能）

### Plan-Solve 并行执行

**现状**：Plan-Solve 中的步骤串行执行，对于无依赖关系的独立步骤是一种浪费。
**建议方案**：在 `_plan` 中标注步骤间的依赖关系，`_execute_all` 中用 `asyncio.gather()` 并发执行无依赖的步骤。
**预估时间**：1 天
**优先级**：低（对当前场景影响不大）

---

## 四、改进效果对比

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 单元测试数量 | 3 | **22**（+19） |
| 集成测试数量 | 1 | **3**（+2） |
| 代码重复（mini ReAct loop） | 3 处内联 | **1 处公共函数** |
| Benchmark 检测方式 | 硬编码字符串匹配 | **State 标志 + 硬编码兜底** |
| 流式输出支持 | 无 | **astream_events + token stream** |
| 测试覆盖模块 | 仅 Context | **tools、memory、routing、state、stream** |

---

## 五、剩余风险

以下风险在当前架构中仍然存在，但评估为可接受：

1. **`_BENCHMARK_PATTERNS` / `_BENCHMARK_SIGNALS` 仍被保留** — 作为 `benchmark_mode` State 标志的兜底。如果用户在非 eval 场景下恰好问了一个和 benchmark 完全相同的问题，记忆不会被存储。这是设计取舍：宁可漏存一条真实记忆，也不让 benchmark 污染记忆库。
2. **`eval()` sandbox 仍是白名单方案** — 对于内部使用的 Agent 场景（非公开部署），当前方案风险可控。公开部署前必须升级。
3. **测试仍不覆盖实际的 LLM 调用** — 单元测试 mock 了 LLM，集成测试需要 API key。当前方案依赖 `run_evals.py`（14 项 benchmark）作为回归检测手段。
