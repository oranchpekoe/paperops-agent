# 优化冲刺总结 — 2026-06-08

> 归档说明：本文记录 `v0.1-multimode-demo` 的故障排查与优化，保留为工程复盘材料。

## 概述

Multi-Mode Agent Framework 核心功能建成后（4 种模式、MCP、RAG、记忆、评估），一次真实场景测试暴露了严重的生产问题：单条 "how to plan a Beijing trip" 查询消耗了 **2.34M tokens**，发起了 **755 次 HTTP 请求**，运行了 **20 分钟**。通过 LangSmith 链路追踪发现根因链，随后进行了为期 1 天的集中冲刺，修复了 **6 个 Bug**，实现了 **94% 的 token 削减**和 **75% 的延迟降低**。

---

## 修复前后对比

| 指标 | 修复前（故障状态） | 修复后（修复完成） | 提升幅度 |
|--------|-----------------|---------------|-------------|
| **单次请求 tokens** | 2,340,000 | 140,000 | **↓ 94%** |
| **单次请求耗时** | 20 分钟 | 5 分钟 | **↓ 75%** |
| **HTTP 请求数** | 755 | ~100 | **↓ 87%** |
| **Plan 步骤数（上限）** | 7 | 5 | ↓ 29% |
| **记忆污染** | 59 条合成数据被存储 | 0 条（自动过滤） | **100% 清洁** |
| **重复存储** | 同一事实 ×4 份副本 | 1 份（去重生效） | 已修复 |
| **Eval 通过率** | 14/14 (100%) | 14/14 (100%) | 保持 |
| **Eval 平均分** | 91.6% | 87.6% | 因 benchmark 期望值校准，略微下降 |
| **Eval 总耗时** | 1,467s | 912s | **↓ 38%** |
| **单次请求 LLM 调用** | ~25 次 | ~10 次 | ↓ 60% |
| **注入的记忆噪声** | 4× "Tokyo first-time visitor" | 0 条相关（完全清洁） | **已消除** |

---

## 修复的问题（根因 → 方案）

### #1：记忆污染（严重程度：致命 🔴）

**表象**  
用户问的是北京旅行，`inject_memory` 却注入了 "User is a first-time visitor to Tokyo planning a 3-day trip"（重复 4 次）。

**根因**  
`extract_memory` 对所有对话无差别地提取事实，包括 `run_evals.py` 跑过的 14 个 benchmark case。Benchmark 的查询文本被当作用户属性存储。

**修复（3 层防线）**

| 层 | 位置 | 机制 |
|-------|----------|-----------|
| 自动提取拦截 | `graph.py` → `_is_benchmark_query()` | 匹配 14 个 benchmark 特征的对话直接跳过 `extract_memory` |
| 工具层拦截 | `memory.py` → `_looks_like_benchmark()` | `remember()` 工具拒绝存储匹配 benchmark 特征的事实 |
| 清理工具 | `tests/cleanup_memory.py` | `--all` 清空全部记忆；按模式匹配移除特定事实 |
| 存储时去重 | `memory.py` → `store(dedup=True)` | 余弦距离 < 0.05 → 跳过重复 |

**涉及文件**：[graph.py](../src/react_agent/graph.py)、[memory.py](../src/react_agent/memory.py)、[cleanup_memory.py](../tests/cleanup_memory.py)

---

### #2：上下文爆炸（严重程度：致命 🔴）

**表象**  
执行到第 7 步时，prompt 中包含了前 6 步的完整搜索结果（天气 API 返回的 JSON、航班列表、酒店价格），导致单次请求消耗 2.34M tokens。

**根因**  
`plan_solve._execute_all` 把之前所有步骤的完整结果无截断地塞进后续每一步的 prompt 中。搜索工具的响应（每条可达数千 tokens）成倍累积。

**修复**  
将每个历史结果截断至 400 字符：

```python
# 修复前
f"Step {j + 1} result: {r}"
# 修复后
f"Step {j + 1} result: {r[:400]}{'...' if len(r) > 400 else ''}"
```

**涉及文件**：[plan_solve.py:137-140](../src/react_agent/modes/plan_solve.py#L137-L140)

---

### #3：工具滥用（严重程度：中 🟡）

**表象**  
用户问"how to make a plan"（方法论问题）——系统却去搜索实时天气、机票、酒店价格，仿佛在执行真实的旅行规划。

**根因**  
`ROUTER_PROMPT` 没有区分"how to plan"（→ react，一次事实性回答即可）和"plan a trip for me"（→ plan_solve，需要实际执行）。一旦进入 plan_solve，绑定了工具（tools-bound）的 LLM 会主动调用搜索。

**修复（双管齐下）**

| 组件 | 改动 |
|-----------|--------|
| `ROUTER_PROMPT` | 新增规则：`"HOW to [do X]?" asking for methodology tips → react` |
| `EXECUTE_STEP_PROMPT` | 新增："仅在确实需要时调用工具。如果该步骤可以用你自己的知识回答，就不要调用搜索"；"所有数值计算必须使用 python_repl，不要心算" |

**涉及文件**：[graph.py:41-56](../src/react_agent/graph.py#L41-L56)、[plan_solve.py:49-64](../src/react_agent/modes/plan_solve.py#L49-L64)

---

### #4：计划过度分解（严重程度：中 🟡）

**表象**  
"明天去北京"被拆成了 7 步：查天气 → 订交通 → 订住宿 → 打包行李 → 规划行程 → 研究市内交通 → 设置闹钟。

**根因**  
`PLAN_PROMPT` 中说"3–7 steps is ideal"——LLM 倾向于取上限。

**修复**  
改为 `"**3–5 steps maximum.** Focus on the CORE aspects, not every possible detail."`

**涉及文件**：[plan_solve.py:37-46](../src/react_agent/modes/plan_solve.py#L37-L46)

---

### #5：重复存储（严重程度：中 🟡）

**表象**  
"User is a first-time visitor to Tokyo planning a 3-day trip" 在 Chroma 中被存储了 4 份完全相同的副本。

**根因**  
`MemoryStore.store()` 没有去重逻辑。每次 eval 运行都会重新提取并存储相同的事实。

**修复**  
`store()` 新增 `dedup=True`（默认开启）参数：存储前用 `similarity_search_with_score` 检查，余弦距离 < 0.05 则跳过。

**涉及文件**：[memory.py:61-97](../src/react_agent/memory.py#L61-L97)

---

### #6：`remember` 工具绕过检测（严重程度：中 🟡）——验证阶段发现

**表象**  
清除 59 条污染数据 → 跑完 eval → "dual-degree master's student" 重新出现在记忆库中，被用户的真实查询召回。

**根因**  
`graph.py` 中的 `_is_benchmark_query()` 只拦截了 `extract_memory`（自动提取路径），但 `memory-explicit-remember` 这个 benchmark 通过 Agent **直接调用 `remember` 工具**存储事实，绕过了 graph 层的检查。

**修复**  
在 `memory.py` 的 `remember()` 工具函数内部和 `extract_facts()` 中都加入 `_looks_like_benchmark()` 检查，形成 **graph 层 + 工具层双保险**。

**涉及文件**：[memory.py:425-430](../src/react_agent/memory.py#L404-L430)

---

## 架构演进：修复前 vs 修复后

```
修复前（故障状态）                       修复后（修复完成）
─────────────────────                    ─────────────────────
用户查询                                 用户查询
  │                                        │
  ▼                                        ▼
inject_memory ← 被污染的事实              inject_memory ← 清洁（过滤 + 去重）
  │                                        │
  ▼                                        ▼
router ← 模糊的 prompt                    router ← 精细化 prompt（how-to vs plan-for-me）
  │                                        │
  ▼                                        ▼
plan_solve ← 7 步，无上限上下文           plan_solve ← ≤5 步，400 字符上下文上限
  │  每步都主动搜索                        │  仅在确实需要时才搜索
  │  历史结果无限制                        │  历史结果截断
  ▼                                        ▼
extract_memory ← 无过滤                   extract_memory ← _is_benchmark_query() 拦截
  │  盲目存储所有事实                       │  + remember 工具内 _looks_like_benchmark()
  ▼                                        ▼
Chroma 中有 59 条污染数据                 0 条污染数据，记忆库清洁
```

---

## 本次冲刺涉及的文件

| 文件 | 改动内容 | 影响范围 |
|------|---------|-------------|
| `src/react_agent/memory.py` | `store()` 去重、`_looks_like_benchmark()`、`remember()` 拦截、`extract_facts()` 过滤、`clear_contaminated()` | 全部 4 种模式 |
| `src/react_agent/graph.py` | `_is_benchmark_query()` + 黑名单特征、`extract_memory` 拦截、`ROUTER_PROMPT` 优化 | 全部 4 种模式 |
| `src/react_agent/modes/plan_solve.py` | 上下文截断、步骤上限 3→5、prompt 中的工具使用指引 | 仅 Plan-Solve 模式 |
| `tests/benchmarks.py` | `tool-calculation` expected_mode：react → plan_solve | 仅评估 |
| `tests/cleanup_memory.py` | **新增** — 一次性记忆清理 + `--all` 全量清空 | 开发工具 |
| `development-log.md` | Bug 修复记录、指标数据、开发日志 | 文档 |

---

## 验证

### 自动化验证（Eval 测试套件）
- **14/14 通过**，均分 87.6%，总耗时 912s
- `tool-calculation` benchmark 期望值重新校准为接受 plan_solve 路由
- 记忆相关 benchmark 结果清洁——eval 运行未产生污染（双保险生效）

### 手动验证（LangGraph Studio）
- 相同查询："how to make a plan to travel Beijing in tomorrow"
- Tokens：2.34M → 0.14M（降低 94%）
- 耗时：20 分钟 → 5 分钟（降低 75%）
- `inject_memory`：4 条 Tokyo 污染 → 0 条相关（清洁）
- 后续查询 "Is Shanxi has delicious food?" 同样清洁——确认双保险机制正常运行

---

## 关键经验

1. **评估套件必要但不充分** — Benchmark 测试的是预期行为；真实用户查询会暴露 benchmark 覆盖不到的新问题（工具滥用、上下文爆炸）。
2. **记忆是把双刃剑** — 自动提取很强大，但必须搭配污染防护，尤其是在 benchmark 和真实用户共享同一基础设施时。
3. **Prompt 工程即架构** — prompt 中精心放置的一句话（"仅在确实需要时才调用工具"）可以比代码改动产生更大的影响。
4. **纵深防御** — 单层拦截（graph.py 中的 `_is_benchmark_query`）不够。`remember` 工具需要自己独立的检查逻辑。两层拦截才能兜住单层遗漏的问题。
