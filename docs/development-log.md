# 开发日志与Bug修复记录

> 归档说明：本文记录 [`v0.1-multimode-demo`](https://github.com/oranchpekoe/paperops-agent/tree/v0.1-multimode-demo) 原型的开发过程，文中的代码路径均属于该标签。当前 PaperOps 产品范围以 [product-spec.md](product-spec.md) 为准。

## 开发日志

| 日期 | 进展 |
|------|------|
| 2026-05 | 项目初始化：四种模式子图（ReAct, Reflection, Plan-Solve, Supervisor） + LLM Router |
| 2026-06-01 | MCP 协议集成：`mcp.py` + `MultiServerMCPClient` + 懒加载降级；当时创建的演示 Server 保留在 `v0.1-multimode-demo` 标签中 |
| 2026-06-02 | RAG 文档检索：Chroma 向量存储 + `RecursiveCharacterTextSplitter` → 持久化 `.chroma_db/` |
| 2026-06-03 | Supervisor Structured Output 迁移：文本解析 → Pydantic `SupervisorDecision` + `with_structured_output()` |
| 2026-06-05 | 三层记忆管理：`memory.py`（`MemoryStore` + `remember/recall` 工具 + `extract_facts` + `compress_context`）→ 跨会话生效 |
| 2026-06-06 | Eval 框架：`benchmarks.py`（14 cases × 5 维度）+ `run_evals.py`（加权评分 + JSON 报告）→ 100% 通过率，均分 91.6% |
| 2026-06-08 | 跨模型验证：deepseek-v4-flash (100%, 91.6%) vs gpt-4o-mini (92.9%, 90.1%)，确认框架模型无关性 |
| 2026-06-08 | **Bug 修复 Sprint**：5 个生产问题修复（记忆污染、上下文爆炸、工具滥用、步骤过多、重复存储），见下方 Bug 修复记录 |
| 2026-06-08 | 修复验证：全量 eval 14/14 通过（100%），均分 87.6%，总耗时 912s（↓38%），清除 59 条记忆污染，dedup 生效 |
| 2026-06-08 | 用户实体验证：同 query tokens 2.34M→0.14M（↓94%），耗时 20min→5min（↓75%），请求 755→~100（↓87%），记忆注入干净，项目收工 |
| 2026-06-08 | **工程质量 Sprint — 代码重构**：提取 `run_mini_react_loop()` 公共函数消除 supervisor.py 和 plan_solve.py 中 3 处重复 mini ReAct 循环（各减 ~15 行）；新增 `benchmark_mode: bool` State 字段 + `ContextVar` 双层隔离（graph 层 + tool 层），替代 graph.py 和 memory.py 中两套硬编码 benchmark 黑名单；修复 `_parse_text_decision` 中关键字匹配 case-sensitivity bug（小写关键字对大写字串 → 改用 `raw_lower`） |
| 2026-06-08 | **工程质量 Sprint — 功能补全**：新增 `stream.py` 流式输出模块（`stream_events()` 完整事件流 + `stream_tokens()` token 级流式）；新增 `extract_facts` benchmark_mode 提前返回（省 LLM 调用）；`pyproject.toml` 注册 `integration` pytest mark |
| 2026-06-08 | **测试覆盖补全**：从 3 个单元测试扩展到 **73 个测试**（+70），覆盖 state(13) / tools(17) / routing(21) / memory(15, 含 4 integration) / stream(3) / configuration(3) / graph(1)；创建 `docs/gaps-and-improvements.md` 缺口分析文档；`docs/optimization-summary.md` 翻译为中文版 |
| 2026-06-08 | 用户 LangSmith 验证："山西有啥好吃的东西" → react 路由 ✓，1 次 search ✓，0 记忆污染 ✓，21s ✓，框架全链路正常；确认 Router prompt 优化、工具克制提示、dedup、双保险过滤全部生效 |
| 2026-06-10 | **文档补齐**：创建 `docs/trace-examples.md`（4 种模式完整数据流追踪，含 messages 逐步变化）；重写 `docs/project-overview.md`（从 47 行扩至 ~280 行）；审查数据流时发现并修复 2 处过度截断问题，见下方 Bug 修复 #11-#12 |
| 2026-06-10 | **代码质量审计**：全项目扫描发现 6 处问题——2 处过度截断（memory.py compress_context/extract_facts）、2 处 DRY 违规（重复的"找最后 HumanMessage"逻辑 + 重复的 benchmark 模式列表）、`get_message_text()` 定义但未使用。全部修复，见 Bug 修复 #13-#14 + 附带发现 |
| 2026-06-10 | **Supervisor 三 specialist DRY 重构**：`_supervisor_researcher` / `_analyst` / `_executor` 三个节点 79 行高度重复（仅 prompt 和是否绑工具有区别）→ 提取 `_run_specialist()` 公共函数（58 行），三个节点变为 1-3 行 thin wrapper。附带修正 researcher 不收集上下文的不一致问题。见 Bug 修复 #15 |
| 2026-06-11 | **短期记忆修复 — 添加 LangGraph Checkpointer**：发现 `builder.compile()` 未配置 checkpointer，每次 `ainvoke()` `state.messages` 从零开始——短期记忆名存实亡。添加 `MemorySaver` checkpoint 后同 `thread_id` 内 messages 跨调用累积。见 Bug 修复 #16 |
| 2026-06-11 | **上下文压缩接线 — `compress_context` 从半成品到真正生效**：发现 `compress_context()`（memory.py）和 `conversation_summary`（state.py）均已定义但从未被调用——摘要记忆层是写好了电路但没接电源。在 `extract_memory` 中接入：messages > 20 条时自动调用 LLM 压缩旧消息为摘要段落，通过 `RemoveMessage` + `add_messages` reducer 替换旧消息。详情见 Bug 修复 #17 |
| 2026-06-11 | **记忆系统全线完工**：两段召回（0.6→0.75）、`memory_status` 可视化、extract_facts 覆盖简单查询、压缩 20→50 阈值。三层记忆全部验证通过。详情见 Bug 修复 #20-#23 |

---

## Bug 修复记录

*触发背景*：用户测试 "how to make a plan to travel Beijing in tomorrow" → 单次请求消耗 2.34M tokens、755 次请求、耗时 20 分钟。LangSmith trace 显示 `inject_memory` 注入了 4 条 benchmark 污染数据（"User is a first-time visitor to Tokyo planning a 3-day trip"）。

### 修复 #1：记忆污染 —— benchmark 数据被当成用户事实存储 🔴

| 项目 | 内容 |
|------|------|
| **现象** | `inject_memory` 注入的记忆上下文包含 "User is a first-time visitor to Tokyo planning a 3-day trip"（重复 4 次），这是 eval benchmark 的测试 query |
| **根因** | `extract_memory` 对所有对话无差别提取事实，包括 `run_evals.py` 跑的 14 个 benchmark case |
| **修复** | (a) 新增 `_is_benchmark_query()` + `_BENCHMARK_PATTERNS` 黑名单，在 `extract_memory` 中自动跳过 benchmark 对话；(b) 新增 `MemoryStore.clear_contaminated()` 方法，支持按模式批量清除污染数据；(c) 创建 `tests/cleanup_memory.py` 一次性清理脚本 |
| **文件** | [graph.py](../src/react_agent/graph.py), [memory.py](../src/react_agent/memory.py), [cleanup_memory.py](../tests/cleanup_memory.py) |

### 修复 #2：上下文爆炸 —— plan_solve 每步把前面所有结果完整塞入 prompt 🔴

| 项目 | 内容 |
|------|------|
| **现象** | 到 Step 7 时，prompt 包含前 6 步的完整搜索结果（天气 API JSON、航班列表、酒店价格），单次请求消耗 2.34M tokens |
| **根因** | `_execute_all` 中 `previous_results` 无截断，search 工具返回的 JSON blob（每条可达数千 tokens）逐步累积 |
| **修复** | 在 `_execute_all` 中对每个 previous result 截断至 400 字符 |
| **文件** | [plan_solve.py](../src/react_agent/modes/plan_solve.py) |

### 修复 #3：工具滥用 —— "how to make a plan" 被当成 "execute the plan" 🟡

| 项目 | 内容 |
|------|------|
| **现象** | 用户问 "how to make a plan"，但系统搜索了实时天气、机票价格、酒店房价——用户只需要方法论，不需要实时数据 |
| **根因** | `ROUTER_PROMPT` 未区分 "how to plan"（方法论，应 → react）和 "plan a trip for me"（执行规划，应 → plan_solve）；plan_solve 绑了 search 工具后 LLM 会主动搜索 |
| **修复** | `ROUTER_PROMPT` 增加规则：'"HOW to [do X]?" asking for methodology tips → react'；`EXECUTE_STEP_PROMPT` 增加："Use tools ONLY when genuinely needed" |
| **文件** | [graph.py](../src/react_agent/graph.py), [plan_solve.py](../src/react_agent/modes/plan_solve.py) |

### 修复 #4：步骤过多 —— 简单规划拆了 7 步 🟡

| 项目 | 内容 |
|------|------|
| **现象** | "明天去北京" 被拆成 7 步：查天气、订交通、订住宿、打包行李、规划行程、研究交通、设置闹钟 |
| **根因** | `PLAN_PROMPT` 说 "3-7 steps is ideal"，LLM 倾向于取上限 |
| **修复** | `PLAN_PROMPT` 改为 "**3–5 steps maximum.**" + "Focus on the CORE aspects" |
| **文件** | [plan_solve.py](../src/react_agent/modes/plan_solve.py) |

### 修复 #5：重复存储 —— 同一个 fact 存了 4 次 🟡

| 项目 | 内容 |
|------|------|
| **现象** | "User is a first-time visitor to Tokyo planning a 3-day trip" 在 Chroma 中存储了 4 份完全相同的副本 |
| **根因** | `MemoryStore.store()` 无去重逻辑，每次 benchmark 运行都重新提取并存储相同事实 |
| **修复** | `store()` 新增 `dedup` 参数（默认 True）：存储前用 `asimilarity_search_with_score` 检查，cosine distance < 0.05 则跳过 |
| **文件** | [memory.py](../src/react_agent/memory.py) |

### 修复 #6：`remember` 工具绕过 benchmark 过滤 —— 验证阶段发现的漏网之鱼 🟡

| 项目 | 内容 |
|------|------|
| **现象** | 清除 59 条污染数据 → 跑完 eval → "dual-degree master's student" 重新出现在记忆库，被用户真实查询召回 |
| **根因** | `_is_benchmark_query()` 只拦截了 `extract_memory`（自动提取），但 `memory-explicit-remember` benchmark 通过 Agent **直接调用 `remember` 工具**存储，绕过了 graph 层的检查 |
| **修复** | 在 `memory.py` 的 `remember()` 工具函数和 `extract_facts()` 中都加入 `_looks_like_benchmark()` 检查，形成 **graph 层 + 工具层双保险** |
| **文件** | [memory.py](../src/react_agent/memory.py) |

### 修复 #7：代码重复 —— supervisor / plan_solve 中 mini ReAct 循环重复 3 次 🟡

| 项目 | 内容 |
|------|------|
| **现象** | `_supervisor_researcher`、`_supervisor_executor`、`_execute_all` 中的 mini ReAct 循环几乎完全相同（~50 行 × 3） |
| **根因** | 各节点独立实现了相同的 tool-calling 循环逻辑 |
| **修复** | 在 `tools.py` 中新增 `run_mini_react_loop()` 公共函数，3 处调用点改用统一实现 |
| **文件** | [tools.py](../src/react_agent/tools.py), [supervisor.py](../src/react_agent/modes/supervisor.py), [plan_solve.py](../src/react_agent/modes/plan_solve.py) |

### 修复 #8：benchmark 黑名单硬编码 —— 新增 benchmark 需手动同步两处 🟡

| 项目 | 内容 |
|------|------|
| **现象** | `graph.py` 的 `_BENCHMARK_PATTERNS` 和 `memory.py` 的 `_BENCHMARK_SIGNALS` 是两套独立的硬编码字符串列表，新增 benchmark 需要手动同步 |
| **根因** | benchmark 检测完全依赖字符串匹配，无通用隔离机制 |
| **修复** | (a) `MainState` 新增 `benchmark_mode: bool` 字段，eval runner 设置后 graph 节点直接跳过记忆提取；(b) `memory.py` 新增 `ContextVar` + `set_benchmark_mode()` / `_in_benchmark_mode()`，tool 函数独立检测；(c) 硬编码黑名单保留作为兜底，形成 **State 标志优先 + ContextVar 工具层 + 硬编码兜底** 三层防护 |
| **文件** | [state.py](../src/react_agent/state.py), [graph.py](../src/react_agent/graph.py), [memory.py](../src/react_agent/memory.py), [run_evals.py](../tests/run_evals.py) |

### 修复 #9：无流式输出 🟡

| 项目 | 内容 |
|------|------|
| **现象** | 所有 4 种模式均使用 `model.ainvoke()`（batch），不支持 token-level streaming |
| **根因** | 未利用 LangGraph 的 `astream_events()` 能力 |
| **修复** | 新增 `stream.py`：`stream_events()` 完整事件流（调试/可观测性）+ `stream_tokens()` token 级流式（前端展示） |
| **文件** | [stream.py](../src/react_agent/stream.py), [test_stream.py](../tests/test_stream.py) |

### 修复 #10：测试覆盖严重不足 🔴

| 项目 | 内容 |
|------|------|
| **现象** | 仅 3 个单元测试（Context 初始化），核心模块（router / 4 个 mode / memory / tools）均无独立测试 |
| **根因** | 开发节奏快，测试滞后于功能开发 |
| **修复** | 新增 70 个测试：`test_state.py`(13) / `test_tools.py`(17) / `test_routing.py`(21) / `test_memory.py`(15, 含 4 integration) / `test_stream.py`(3) / `test_configuration.py`(3, 原有)；`run_mini_react_loop` 和 memory ContextVar 逻辑均有 mock 验证 |
| **文件** | `tests/unit_tests/*`, `tests/test_stream.py`, `pyproject.toml`（注册 `integration` mark） |

### 修复 #11：`_execute_all` 前序结果过度截断 —— 截的是 LLM 总结而非原始搜索结果 🟡

| 项目 | 内容 |
|------|------|
| **现象** | `_execute_all` 中对每个 previous result 执行 `r[:400]`，但 `results` 里存的已经是 LLM 在 mini ReAct 结束后输出的总结文本（如"高铁 4.5 小时最快，票价约 ¥550"），天然不超 400 字符，截断只会砍掉关键数字或结论的后半段 |
| **根因** | Sprint #2 修复上下文爆炸时一刀切加了截断，但 `results.append(str(last.content))` 存的是压缩后的文本，不是原始 ToolMessage JSON。真正的上下文爆炸已在步数限制（3-5）+ max_rounds=3 处解决 |
| **修复** | 移除 `r[:400]` 截断，完整保留每步 LLM 总结文本 |
| **文件** | [plan_solve.py](../src/react_agent/modes/plan_solve.py) |
| **发现者** | 审查数据流 trace 文档时发现 |

### 修复 #12：`_gather_context` 一刀切截断 —— 所有消息无差别截断，丢失关键信息 🟡

| 项目 | 内容 |
|------|------|
| **现象** | `_gather_context` 对所有消息执行 `str(msg.content)[:500]`。第一次改为"只截 ToolMessage"，但进一步审查发现搜索 ToolMessage 里的 Tavily JSON 包含事实数据和关键数字，截断同样危险——supervisor 决策或 `_synthesise_final_answer` 可能因为丢失一个数字而给出错误答案 |
| **根因** | Sprint 阶段为了防上下文爆炸加的防御性截断，但当前 Supervisor 已有结构性限制（max 5 迭代 + max_rounds=3 + asyncio.sleep(3)），实际消息量不会爆炸 |
| **修复** | 移除所有截断逻辑，完整保留每条消息。上下文安全由结构性的迭代上限保证，而非事后截断 |
| **文件** | [supervisor.py](../src/react_agent/modes/supervisor.py) |
| **发现者** | 审查数据流时发现过度截断，经讨论确认 ToolMessage 也不应截 |

### 修复 #13：`memory.py` `compress_context` 和 `extract_facts` 过度截断 🟡

| 项目 | 内容 |
|------|------|
| **现象** | `compress_context` 对每条消息执行 `str(m.content)[:400]`，`extract_facts` 执行 `[:500]`。被截断的消息文本直接送给 LLM 做摘要/事实提取。如果关键信息落在截断点之后，摘要会丢失重要上下文，事实提取会漏掉用户偏好或决策 |
| **根因** | 与 #11/#12 同源——Sprint 阶段一刀切加截断，未考虑截断对象（LLM 的输入文本）对完整性的要求。compress_context 和 extract_facts 本身已有其他结构限制（messages[-20:]、max_facts=5），不需要额外截断 |
| **修复** | 移除两处截断，完整保留消息内容 |
| **文件** | [memory.py](../src/react_agent/memory.py)（`compress_context` L298 + `extract_facts` L358） |
| **发现者** | 代码审计时发现 |

### 修复 #14：重复代码消除 —— 重复的 HumanMessage 提取 + 重复的 benchmark 列表 🟡

| 项目 | 内容 |
|------|------|
| **现象** | (a) "找最后一条 HumanMessage" 逻辑在 `route_mode` 和 `inject_memory` 中各有一份完全相同的实现（~7 行）；(b) benchmark 模式列表 `_BENCHMARK_PATTERNS`（graph.py 14 项）和 `_BENCHMARK_SIGNALS`（memory.py 13 项）不一致——前者有 `"remember this:"` 后者没有，且各自维护检测函数 `_is_benchmark_query` / `_looks_like_benchmark` |
| **根因** | 功能迭代过程中各自添加，未及时提取共享模块 |
| **修复** | (a) 提取 `_get_last_user_message(state)` 公共 helper；(b) 删除 graph.py 的 `_BENCHMARK_PATTERNS` 和 `_is_benchmark_query`，统一使用 memory.py 的 `_BENCHMARK_SIGNALS`（补充缺失的 `"remember this:"`）+ `_looks_like_benchmark`。修改后 benchmark 模式列表为单一数据源，新增 benchmark 只需改一处 |
| **文件** | [graph.py](../src/react_agent/graph.py), [memory.py](../src/react_agent/memory.py), [test_routing.py](../tests/unit_tests/test_routing.py), [test_memory.py](../tests/unit_tests/test_memory.py) |
| **发现者** | 代码审计时发现 |

### 修复 #15：Supervisor 三 specialist 重复代码 —— 79 行高度重复的三个节点 🟡

| 项目 | 内容 |
|------|------|
| **现象** | `_supervisor_researcher`（28 行）、`_supervisor_analyst`（21 行）、`_supervisor_executor`（30 行）共享相同结构：提取 subtask → 收集上下文 → 格式化 prompt → 构建 messages → 调模型 → 返回结果。researcher 和 executor 额外多一层 MCP 加载 + `bind_tools` + `run_mini_react_loop` + 消息过滤，但二者之间完全一致。三个函数仅在三个参数上不同：`name`（决定 trace label）、`prompt_template`、`use_tools` |
| **根因** | 最初按专家类型逐个实现，各自复制粘贴后微调，未及时识别共有模式。此外 researcher 是唯一不收集上下文的 specialist——在首次调用时无伤大雅，但被 `_supervisor_review` 二次委派时不知道已搜到的事实，可能重复搜索 |
| **修复** | 提取 `_run_specialist(state, runtime, *, name, prompt_template, use_tools)` 公共函数（~58 行含文档），三个 specialist 节点变为 1-3 行 thin wrapper。`name` 参数驱动 trace label、HumanMessage 前缀、是否收集上下文。三个 specialist 统一收集上下文，消除 researcher 的不一致 |
| **文件** | [supervisor.py](../src/react_agent/modes/supervisor.py) |
| **发现者** | 用户审查代码发现 |

### 修复 #16：短期记忆形同虚设 —— graph 编译时缺少 checkpointer 🔴

| 项目 | 内容 |
|------|------|
| **现象** | 用户在 LangSmith 上做多轮测试时，每一轮新的 input JSON 都从 `__start__` 进入，`state.messages` 被清空。例如第一轮问"北京天气怎么样"，第二轮问"那后天呢"——Agent 不知道"那后天呢"指北京，因为上一轮对话已经丢失 |
| **根因** | `graph.py` 中 `builder.compile(name="MultiMode Agent")` 未配置 checkpointer。LangGraph 的 `add_messages` reducer 只在**同一次 ainvoke 调用内**累积消息——跨调用时没有 checkpointer 则每次创建全新 state。流式模块 `stream.py` 虽用了 `thread_id`，但那只是 streaming 的配置标记，不代表持久化 |
| **修复** | `builder.compile()` 添加 `checkpointer=MemorySaver()`（内存级，适合开发/演示）。同 `thread_id` 的多轮 `ainvoke()` 现在共享 state，`add_messages` reducer 自动追加新消息而非覆盖。调用方式：`graph.ainvoke({"messages": [...]}, config={"configurable": {"thread_id": "session-1"}})` |
| **文件** | [graph.py](../src/react_agent/graph.py)（新增 `from langgraph.checkpoint.memory import MemorySaver`，compile 添加 checkpointer 参数） |
| **发现者** | 用户讨论记忆系统时发现 |

### 修复 #17：上下文压缩半成品 —— `compress_context` 定义但从未被调用 🟡

| 项目 | 内容 |
|------|------|
| **现象** | `compress_context()` 函数（memory.py）实现了消息过长时 LLM 压缩旧消息的功能，`MainState.conversation_summary` 字段（state.py）预留了存储压缩摘要的位置，`extract_memory` 的 docstring 明确写了 "Also triggers context compression when the message list is long"——但实际代码中 `compress_context` 只被 import，从未被调用。加上 checkpointer 后多轮对话消息无限累积，上下文必然爆炸 |
| **根因** | 摘要记忆作为三层记忆之一的设计稿已写好、代码已写好，但接入 graph 的最后一步没做——典型的"电路板画好了但电源线没焊" |
| **修复** | 在 `extract_memory` 中添加压缩逻辑：(1) messages > 20 条触发；(2) 调用 `compress_context` 压缩旧消息为摘要段落；(3) 通过 `RemoveMessage` + `add_messages` reducer 移除旧消息、写入压缩后的消息列表；(4) 同时写入 `conversation_summary` 字段。阈值 20 条在实践中约等于 5-10 轮对话 |
| **文件** | [graph.py](../src/react_agent/graph.py)（新增 `RemoveMessage` import，`extract_memory` 中添加压缩逻辑 ~15 行） |
| **发现者** | 用户质疑"上下文岂不是很容易爆炸"时发现 |

### 修复 #18：记忆召回的语义不相关 —— `MemoryStore.recall()` 无相似度阈值 🔴

| 项目 | 内容 |
|------|------|
| **现象** | 用户问"我想去中国有海的地方旅游"，`inject_memory` 召回了"LangChain was founded by Harrison Chase and Ankush Gola"作为相关记忆注入上下文。这个事实在语义上和旅游查询完全无关，属于向量搜索的"噪音匹配" |
| **根因** | `MemoryStore.recall()` 使用 `asimilarity_search()` 获取 top-k 结果，无条件返回——即使 cosine distance 接近 1.0（完全不相似）也会被当作"相关记忆"注入。Chroma 总是返回 k 个最近结果，不保证它们真的相似 |
| **修复** | 改用 `asimilarity_search_with_score()`，过滤 cosine distance < 0.5 的结果（0=完全相同，2=完全相反）。阈值 0.5 对应约 60° 的夹角，保证注入的记忆至少与查询有中等以上的语义相关性 |
| **文件** | [memory.py](../src/react_agent/memory.py)（`MemoryStore.recall()` 方法） |
| **发现者** | 用户 LangSmith 测试发现 |

### 修复 #19：Plan-Solve 执行时记忆工具副作用 —— `remember` 在步骤执行中被调用 🟡

| 项目 | 内容 |
|------|------|
| **现象** | Plan-Solve 执行"制定旅行日程"步骤时，mini ReAct 循环调用了 `remember` 工具，存储了"用户想去中国有海的地方旅游"——纯执行逻辑产生了持久化副作用。此外 `recall` 在每个步骤的 mini ReAct 循环中也可能被调用，浪费 tokens 且拖慢执行（本案例 4 个步骤耗时 3.5 分钟） |
| **根因** | `_execute_all` 使用 `get_all_tools()` 获取工具集，该函数自动追加了 memory tools（`remember` / `recall` / `recall_all`）。mini ReAct 循环中的 LLM 看到这些工具后可能在执行步骤时"顺便"调用 |
| **修复** | `_execute_all` 改为直接使用 `list(TOOLS) + list(_mcp_tools)`，排除 memory tools。memory tools 的设计意图是全局可用（通过 `get_all_tools()`），但 Plan-Solve 的执行阶段不应有存储/召回记忆的能力——记忆读取应在 `inject_memory` 完成，记忆写入应在 `extract_memory` 完成 |
| **文件** | [plan_solve.py](../src/react_agent/modes/plan_solve.py)（`_execute_all` 函数） |
| **发现者** | 用户 LangSmith trace 分析发现 |

### 修复 #20：上下文压缩 DeleteMessages 泄露到输出 + 阈值过低 🟡

| 项目 | 内容 |
|------|------|
| **现象** | 用户在 LangSmith 第 4 轮对话后看到 9 条 "Deleted Message" 刷屏在最终输出中，且仅 4 轮对话就触发了压缩（21 条消息 > 20 阈值） |
| **根因** | (a) `compress_threshold = 20` 太低——4 个问题的 ReAct loop + tool calls 轻松超 20；(b) `extract_memory` 返回 `[RemoveMessage(id=mid) for mid in old_ids] + compressed`，其中 `old_ids` 取了**全部**消息 ID（包括最近 10 条要保留的），导致保留的消息也被 Remove→re-add，每个 RemoveMessage 作为调试事件流到客户端显示为 "Deleted Message" |
| **修复** | (a) 阈值 20 → **50**（约 10+ 轮对话）；(b) 只对真正被压缩的旧消息（`messages[:split]`，split = len - 10）发送 RemoveMessage，只添加新摘要 SystemMessage（`compressed[0]`），保留的最近 10 条消息不动——消除了多余的 RemoveMessage |
| **文件** | [graph.py](../src/react_agent/graph.py)（`extract_memory` 压缩逻辑） |
| **发现者** | 用户 LangSmith 多轮测试发现 |

### 修复 #21：`extract_facts` 对简单查询提取 0 条事实 🔴

| 项目 | 内容 |
|------|------|
| **现象** | 用户问"怎么去三亚"→ 系统回答了详细的交通攻略 → `extract_memory` 返回 `extracted 0 facts`。连续 3 个三亚相关查询都无法提取事实，导致 `inject_memory` 后续查询全部 "no relevant facts found"——整个长期记忆链断裂 |
| **根因** | (a) `extract_facts` 只看最后 20 条消息——多步模式下原始用户问题可能被 tool 输出挤出窗口；(b) JSON 解析失败被 `except Exception: _logger.debug()` 静默吞掉，终端不可见；(c) prompt 示例全是英文场景，对中文简单查询（"怎么去三亚"）引导不足——LLM 判定为 "transient detail" |
| **修复** | (a) 新增 `user_query` 参数，始终放在 transcript 第一行；(b) `except json.JSONDecodeError` → `_logger.warning()` 打印 LLM 原始返回前 200 字符；(c) prompt 增加 CRITICAL 规则——"即使是简单查询，至少提取一条用户兴趣事实"，附带中文示例（"怎么去三亚" → "用户对三亚旅行感兴趣，询问交通方式"），加 "when in doubt, err on the side of extracting" |
| **文件** | [memory.py](../src/react_agent/memory.py)（`extract_facts` 函数）, [graph.py](../src/react_agent/graph.py)（传入 `user_query`） |
| **发现者** | 用户 LangSmith 测试发现 |

### 修复 #22：LangSmith 中 inject/extract 节点显示为空 🟡

| 项目 | 内容 |
|------|------|
| **现象** | 用户在 LangSmith Studio 中查看 `inject_memory` 和 `extract_memory` 节点时，节点存在但内容为空白——无法判断节点是否工作、做了什么 |
| **根因** | LangSmith 按 state 字段变化量渲染节点内容。`inject_memory` 有 5 个 `return {}` 路径（无用户消息、store 不可用、未找到事实、ImportError、Exception），`extract_memory` 有 6 个——只有成功召回/成功压缩时才有 state 变化。节点跑了但返回空 dict，LangSmith 看起来就是"空的" |
| **修复** | (a) `MainState` 新增 `memory_status: str` 字段；(b) `inject_memory` 所有 return 路径带上 `memory_status`（如 `🔍 未找到相关记忆`、`✅ 召回 3 条: 用户对三亚感兴趣...`、`⚠️ 存储不可用`）；(c) `extract_memory` 所有 return 路径同理（如 `✅ 存储 4 条: 用户对三亚... | 用户查询了景点...`、`📝 未提取到值得存储的事实`），且显示事实内容而非 Chroma 内部 ID；(d) `memory_status` 允许中文字符——LangSmith Studio 正确渲染 Unicode |
| **文件** | [state.py](../src/react_agent/state.py), [graph.py](../src/react_agent/graph.py) |
| **发现者** | 用户 LangSmith UI 审查发现 |

### 修复 #23：`inject_memory` 对后续查询始终召回 0 条——元事实 vs 直接问题的 embedding 鸿沟 🔴

| 项目 | 内容 |
|------|------|
| **现象** | 用户在同一 thread 内先后问"怎么去三亚"→"三亚有机场吗"→"三亚有啥好玩的"。每个查询 `extract_memory` 都成功存储了事实（"用户对三亚旅行感兴趣，正在了解交通方式"、"用户确认了三亚凤凰国际机场"等），但每个后续查询的 `inject_memory` 都返回 "no relevant facts found"——尽管存储和查询明显都是关于三亚旅行 |
| **根因** | 存储的事实是**元陈述**（"用户对三亚旅行感兴趣，正在了解交通方式"），而搜索查询是**直接问题**（"三亚有啥好玩的"）。两种语言的 embedding 向量天然有差距——元陈述聚焦于"用户行为描述"，直接问题聚焦于"目的地信息"。text-embedding-3-small 对这两类文本的 cosine distance 通常在 0.6–0.75 之间，而默认阈值 0.6（修复 #18 设为 0.5，本 Sprint 放宽至 0.6）刚好卡在边缘——部分查询能过，部分过不了 |
| **修复** | **两段召回**：(1) 第一轮 `score_threshold=0.6`（高精度）；(2) 若第一轮为空，第二轮 `score_threshold=0.75`（覆盖元事实↔直接问题的 embedding 差距），结果标记为"部分匹配度较低，仅供参考"。0.75 仍能拦截 0.85+ 的完全无关事实（如"LangChain was founded by..."）。同时存储端 prompt 引导 LLM 生成更贴近自然查询的事实（面向用户的兴趣描述而非纯元分析） |
| **文件** | [graph.py](../src/react_agent/graph.py)（`inject_memory`）, [memory.py](../src/react_agent/memory.py)（`recall` score_threshold 默认值 0.5→0.6, `extract_facts` prompt） |
| **发现者** | 用户连续多轮同主题测试发现 |

---

## 附带发现并修复的 Bug

- **两个"过度截断"问题**（#11, #12）：Bug 修复 Sprint 中为了快速止血加的截断逻辑，未区分消息类型的语义——`results` 存的是 LLM 总结（已压缩），AIMessage 含关键结论（不应截）。修复：`_execute_all` 去掉 `r[:400]`，`_gather_context` 去掉全部截断。
- **`get_message_text()` 定义但从未使用**：`utils.py` 提供了能正确处理 `str` / `dict` / `list[content_block]` 三种内容格式的工具函数，但全项目 15+ 处都用 `str(msg.content)` 直接转字符串。已在 memory.py（`compress_context` + `extract_facts`）和 graph.py（`_get_last_user_message`）等 LLM 输入关键路径上替换为 `get_message_text()`。

---

## 修复效果（实测）

| 指标 | 修复前 | 修复后（用户实测） |
|------|--------|-------------------|
| 单次请求 tokens | 2.34M | **0.14M**（↓94%） |
| 单次请求耗时 | 20 min | **5 min**（↓75%） |
| 单次 HTTP 请求数 | 755 | **~100**（↓87%） |
| plan 步骤数 | 7 | ≤5（prompt 约束） |
| eval 通过率 | 100% (14/14) | 100% (14/14) |
| eval 均分 | 91.6% | 87.6%（含 benchmark 期望值校准） |
| eval 总耗时 | 1467s | **912s**（↓38%） |
| 记忆污染 | 59 条（100% benchmark 合成） | **0 条**（全量清除 + 双保险过滤） |
| 重复存储 | 同一 fact ×4 副本 | 1 副本（dedup 生效） |
| `tool-calculation` 路由 | react | react（plan_solve prompt 优化后恢复正确路由） |

**验证通过日期**：2026-06-08

### 2026-06-11 记忆系统完工验证

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| extract 提取率（简单查询） | 0%（"怎么去三亚"→0 facts） | **100%**（每次至少 1 条） |
| extract 提取率（复杂查询） | 0–1 条 | **4–5 条**（用户偏好、约束、决策全覆盖） |
| inject 召回率（同主题连续查询） | 0%（元事实 vs 问题鸿沟） | **>0%（两段召回兜底）** |
| DeleteMessages 输出 | 9 条刷屏 | **0**（阈值 50 + 精准 Remove） |
| LangSmith 节点可见性 | 空白（return {}） | **每节点显示状态描述 + 事实预览** |
| 压缩触发轮数 | 4 轮 | **10+ 轮**（阈值 50） |

---

## 三层记忆系统架构

```
┌──────────────────────────────────────────────────┐
│                  三层记忆系统                      │
├───────────────┬────────────────┬─────────────────┤
│  短期记忆      │  摘要记忆       │  长期记忆        │
│  (Short-term) │  (Summary)     │  (Long-term)    │
├───────────────┼────────────────┼─────────────────┤
│ 消息历史累积   │ 上下文压缩      │ Chroma 向量存储  │
│ add_messages  │ compress_      │ MemoryStore     │
│ +             │ context()      │                 │
│ MemorySaver   │                │                 │
├───────────────┼────────────────┼─────────────────┤
│ 同 thread 内   │ >50 条消息触发  │ 跨 session 持久  │
│ 多轮可见       │ 旧→摘要段落     │ embedding 语义搜 │
├───────────────┼────────────────┼─────────────────┤
│ 实现位置:      │ 实现位置:       │ 实现位置:        │
│ graph.py      │ graph.py       │ memory.py       │
│ checkpointer  │ extract_memory │ extract_memory  │
│               │                │ + inject_memory │
├───────────────┼────────────────┼─────────────────┤
│ 修复编号:      │ 修复编号:       │ 修复编号:        │
│ #16           │ #17, #20       │ #18, #21, #22, │
│               │                │ #23             │
└───────────────┴────────────────┴─────────────────┘
```

**数据流**: `__start__` → `inject_memory`（长短期记忆注入）→ `router` → `[subgraph]` → `extract_memory`（事实提取 + 上下文压缩）→ `__end__`
