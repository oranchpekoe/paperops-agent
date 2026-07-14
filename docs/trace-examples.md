# 数据流追踪：从用户输入到最终回答

> 本文用 4 个具体 query 完整追踪 `state.messages` 在每个节点的变化。
> 理解 messages 的流转是理解整个框架的关键。

---

## 前置知识：state.messages 如何增长

`MainState.messages` 使用了 LangGraph 的 `add_messages` reducer：

```python
messages: Annotated[Sequence[AnyMessage], add_messages]
```

**两条规则**：
1. **追加**（append）—— 每个节点返回的 `{"messages": [...]}` 会接到已有列表后面
2. **去重**（dedup by ID）—— 如果两条消息有相同 ID，只保留一条

所以整个请求过程中 `state.messages` 是**单调增长**的——节点只管往里面加，不管删。最终返回给用户的 messages 包含从输入到输出的完整对话历史。

**阅读下文的约定**：
- 每步用 `📦 state.messages` 展示**该步执行后**的完整 messages 列表
- `➕ 节点返回` 展示该节点新增了什么
- `[0]` `[1]` 等是列表索引，方便追踪哪条消息是哪步加的

---

## 通用流程（所有模式共享）

无论路由到哪个子图，以下 3 个节点始终执行：

```
__start__ → inject_memory → router → [子图] → extract_memory → __end__
```

---

# 示例 1：ReAct 模式

**Query**：`"What's the weather like in Tokyo right now?"`

**Router 判断**：`react`（简单事实查询，需搜索）

---

### Step 0: 用户输入 → graph.ainvoke()

```python
# 外部调用
result = await graph.ainvoke({
    "messages": [HumanMessage(content="What's the weather like in Tokyo right now?")]
})
```

```
📦 state.messages = [
    [0] HumanMessage(content="What's the weather like in Tokyo right now?")
]
```

---

### Step 1: inject_memory 节点

节点从 Chroma 召回与 query 相关的历史事实，作为 SystemMessage 注入。

```
📍 [0/5] Memory: recalled 1 relevant facts
   → 注入记忆上下文 (1 条)
```

```python
# 节点返回
{"recalled_facts": "[Long-term memory — relevant facts...]\n1. User prefers Celsius for weather reports",
 "messages": [SystemMessage(content="[Long-term memory — relevant facts from previous conversations]\n1. User prefers Celsius for weather reports\n2. User lives in Beijing")]}
```

```
📦 state.messages = [
    [0] HumanMessage(content="What's the weather like in Tokyo right now?"),
    [1] SystemMessage(content="[Long-term memory — relevant facts...]\n1. User prefers Celsius..."),  ← 新增
]
```

> **关键点**：记忆是以 SystemMessage 形式注入的——Agent 在后续推理中会看到它，就像系统提示的一部分。

---

### Step 2: route_mode 节点（Router）

LLM 分析用户 query，返回一个单词的路由决策。

```
📍 [1/5] ROUTER 节点被调用
   → 当前 messages 数量: 2
   → 提取到的用户问题: What's the weather like in Tokyo right now?
   → LLM 路由决策: react → mode='react'
```

```python
# 节点返回（不追加 messages！注意：Router 的返回里没有 "messages" 键）
{"mode": "react", "route_reason": "react", "user_query": "What's the weather like in Tokyo right now?"}
```

```
📦 state.messages = [
    [0] HumanMessage(content="What's the weather like in Tokyo right now?"),
    [1] SystemMessage(content="[Long-term memory — relevant facts...]\n1. User prefers Celsius..."),
]
# ↑ 没有变化！Router 只写 mode/route_reason/user_query，不操作 messages
```

> **关键点**：Router **不修改 messages**。它只设置 `mode` 字段，条件边根据 `mode` 选择子图。

---

### Step 3: 条件边路由 → react_agent 子图

```
📍 [2/5] 条件边路由 → 'react' 子图
```

ReAct 子图内部结构：
```
__start__ → call_model ⇄ tools → __end__
```

---

### Step 3a: ReAct._call_model

LLM 绑定全部工具（search, python_repl, retrieve, remember, recall, recall_all），决定调用 search。

```
📍 [3/5] ReAct._call_model — 当前 messages 数: 2
   → LLM 决定调用工具: ['search']
```

```python
# 节点返回 — AIMessage 带有 tool_calls
{"messages": [AIMessage(
    content="",
    tool_calls=[{"name": "search", "args": {"query": "Tokyo weather right now Celsius"}, "id": "call_001"}]
)]}
```

```
📦 state.messages = [
    [0] HumanMessage(content="What's the weather like in Tokyo right now?"),
    [1] SystemMessage(content="[Long-term memory...]"),
    [2] AIMessage(content="", tool_calls=[{"name": "search", "args": {...}, "id": "call_001"}]),  ← 新增
]
```

---

### Step 3b: ReAct._execute_tools（ToolNode）

ToolNode 执行 `search` 工具，返回搜索结果。

```python
# Tavily 搜索结果
{"messages": [ToolMessage(
    content='{"results": [{"title": "Tokyo Weather Today", "content": "Tokyo: 22°C, partly cloudy, humidity 65%..."}]}',
    tool_call_id="call_001"
)]}
```

```
📦 state.messages = [
    [0] HumanMessage(content="What's the weather like in Tokyo right now?"),
    [1] SystemMessage(content="[Long-term memory...]"),
    [2] AIMessage(content="", tool_calls=[{"id": "call_001", ...}]),    ← LLM 的工具调用决定
    [3] ToolMessage(content='{"results": [...]}', tool_call_id="call_001"), ← 工具执行结果
]
```

---

### Step 3c: ReAct._call_model（第二次）

LLM 收到搜索结果，判断信息足够，输出最终答案（无 tool_calls）。

```
📍 [3/5] ReAct._call_model — 当前 messages 数: 4
   → LLM 输出最终答案 (无工具调用) — 前60字: Tokyo is currently experiencing 22°C with partly cloudy skies...
```

```python
{"messages": [AIMessage(content="Tokyo is currently experiencing 22°C (72°F) with partly cloudy skies. Humidity is at 65%...")]}
```

```
📦 state.messages = [
    [0] HumanMessage(content="What's the weather like in Tokyo right now?"),
    [1] SystemMessage(content="[Long-term memory...]"),
    [2] AIMessage(content="", tool_calls=[{"id": "call_001", ...}]),
    [3] ToolMessage(content='{"results": [...]}', tool_call_id="call_001"),
    [4] AIMessage(content="Tokyo is currently experiencing 22°C..."),  ← 最终答案
]
```

---

### Step 3d: ReAct._route → __end__

最后一个 AIMessage 没有 `tool_calls` → 路由到 `__end__`，子图结束。

```
📍 [4/5] 无工具调用 → __end__ (流回主图 → 结束)
```

---

### Step 4: extract_memory 节点

对话完成，LLM 从对话中提取值得记住的事实。

```
📍 [5/5] Memory: auto-stored 2 facts (ids=['a1b2c3d4', 'e5f6g7h8'])
```

LLM 提取到：
- `"User asked about Tokyo weather on 2026-06-09"` → 过滤掉（临时信息，不值得记）
- `"User prefers Celsius for temperature"` → 存入 Chroma（用户偏好）
- `"User is interested in international weather"` → 存入 Chroma

```python
# 节点返回（不追加 messages）
{}
```

`extract_memory` 不向 messages 追加任何内容，它只是副作用（写 Chroma）。

---

### Step 5: 返回给用户

LangGraph 返回最终的 `MainState`。用户收到的 `result["messages"]` 就是上面的完整列表 `[0]~[4]`。前端通常取最后一条 `AIMessage.content` 作为回答。

```
最终 state.messages 长度: 5
   [0] HumanMessage      ← 用户输入
   [1] SystemMessage     ← 长期记忆注入
   [2] AIMessage         ← LLM 决定调 search
   [3] ToolMessage       ← search 返回结果
   [4] AIMessage         ← 最终答案 ✅ 展示给用户
```

---

# 示例 2：Reflection 模式

**Query**：`"Write a short paragraph analyzing why Python became so popular for data science"`

**Router 判断**：`reflection`（"Write a... analyze..." → 写作+分析，需自批判）

---

### Step 0-2：inject_memory → router

与示例 1 相同。假设无相关记忆。

```
📦 state.messages = [
    [0] HumanMessage(content="Write a short paragraph analyzing why Python became so popular for data science"),
]
```

Router 返回：
```python
{"mode": "reflection", "route_reason": "reflection", "user_query": "Write a short paragraph analyzing why Python became so popular for data science"}
```

---

### Step 3: 进入 reflection_agent 子图

```
__start__ → generate → reflect ⇄ refine → __end__
```

---

### Step 3a: _generate — 生成初稿

```
📍 [3/5] Reflection._generate — 生成初始回答
   → 生成完成，回复长度: 520 字符
```

```python
{"messages": [AIMessage(content="Python's dominance in data science stems from three key factors...")],
 "reflection_iteration": 0}
```

```
📦 state.messages = [
    [0] HumanMessage(content="Write a short paragraph analyzing why Python..."),
    [1] AIMessage(content="Python's dominance in data science stems from three key factors..."),  ← 初稿
]
```

`reflection_iteration` 设为 0。

---

### Step 3b: _reflect — 审视初稿（第 1 轮）

```
   → Reflection._reflect — 审视第 1 轮
```

LLM 以苛刻评审者身份审视初稿：

```python
{"messages": [AIMessage(content="ISSUES_FOUND\n1. Missing mention of Jupyter Notebooks...\n2. No discussion of the role of pandas...\n3. Could mention the 2008-2012 timeframe...")]}
```

```
📦 state.messages = [
    [0] HumanMessage(content="Write a short paragraph analyzing why Python..."),
    [1] AIMessage(content="Python's dominance in data science stems from three key factors..."),  ← 初稿
    [2] AIMessage(content="ISSUES_FOUND\n1. Missing mention of Jupyter..."),  ← 审视意见
]
```

---

### Step 3c: _route_reflect → "refine"

审视结果不是 PASS（以 `ISSUES_FOUND` 开头），且未达到 3 轮上限 → 路由到 refine。

```
   → 有问题需要修改 → 路由到 _refine
```

---

### Step 3d: _refine — 根据意见修改

```
   → Reflection._refine — 根据审视意见修改 (迭代 #1)
```

```python
{"messages": [AIMessage(content="Python rose to dominance in data science through a confluence of factors: the 2008 release of pandas...")],
 "reflection_iteration": 1}
```

```
📦 state.messages = [
    [0] HumanMessage(content="Write a short paragraph analyzing why Python..."),
    [1] AIMessage(content="Python's dominance in data science stems from three key factors..."),  ← 初稿
    [2] AIMessage(content="ISSUES_FOUND\n1. Missing mention of Jupyter..."),  ← 审视意见
    [3] AIMessage(content="Python rose to dominance in data science through a confluence..."),  ← 修改稿
]
```

`reflection_iteration` 增加到 1。

---

### Step 3e: _reflect — 审视修改稿（第 2 轮）

```
   → Reflection._reflect — 审视第 2 轮
```

```python
{"messages": [AIMessage(content="PASS — the response now covers all major points clearly and concisely.")]}
```

```
📦 state.messages = [
    [0] HumanMessage(content="Write a short paragraph analyzing why Python..."),
    [1] AIMessage(content="Python's dominance in data science stems from three key factors..."),
    [2] AIMessage(content="ISSUES_FOUND\n1. Missing mention of Jupyter..."),
    [3] AIMessage(content="Python rose to dominance in data science through a confluence..."),
    [4] AIMessage(content="PASS — the response now covers all major points..."),  ← 通过
]
```

---

### Step 3f: _route_reflect → __end__

```
📍 [4/5] Reflection 审视通过（PASS）→ __end__
```

---

### Step 4-5：extract_memory → 返回

同示例 1。

```
最终 state.messages 长度: 5
   [0] HumanMessage   ← 用户输入
   [1] AIMessage      ← 初稿
   [2] AIMessage      ← 审视意见
   [3] AIMessage      ← 修改稿
   [4] AIMessage      ← PASS（最终审视通过）
```

> **注意**：messages 中包含了初稿和修改稿。用户看到的"最终答案"是 `[3]`（最后一次 refine 的输出）。在返回给前端时，通常过滤掉中间的审视消息，只展示最后一个实质性回答。

---

# 示例 3：Plan-Solve 模式

**Query**：`"I'm going to Shanghai this weekend, help me make a travel plan"`

**Router 判断**：`plan_solve`（"Make a plan for..." → 多步骤规划）

---

### Step 0-2：inject_memory → router

同前。假设记忆中有"User lives in Beijing"。

```
📦 state.messages = [
    [0] HumanMessage(content="I'm going to Shanghai this weekend, help me make a travel plan"),
    [1] SystemMessage(content="[Long-term memory...]\n1. User lives in Beijing"),
]
```

Router：`mode = "plan_solve"`

---

### Step 3: 进入 plan_solve_agent 子图

```
__start__ → plan → execute_all → aggregate → __end__
```

---

### Step 3a: _plan — 分解任务

```
📍 [3/5] PlanSolve._plan — 分解任务为步骤
   → 分解出 4 个步骤
```

LLM 输出：
```
1. Determine transportation from Beijing to Shanghai (flight vs high-speed rail)
2. Find 2-3 must-visit attractions and neighborhoods in Shanghai
3. Plan a rough 2-day itinerary with timing
4. Suggest local food and restaurants to try
```

```python
{"plan_steps": [
    "Determine transportation from Beijing to Shanghai (flight vs high-speed rail)",
    "Find 2-3 must-visit attractions and neighborhoods in Shanghai",
    "Plan a rough 2-day itinerary with timing",
    "Suggest local food and restaurants to try"
],
 "current_step": 0,
 "step_results": []}
```

```
📦 state.messages = [
    [0] HumanMessage(content="I'm going to Shanghai this weekend..."),
    [1] SystemMessage(content="[Long-term memory...]\n1. User lives in Beijing"),
]
# ↑ messages 没变！_plan 只设置 plan_steps / current_step / step_results，不操作 messages
```

> **关键点**：`_plan` 返回的 `plan_steps` 是 State 的独立字段，不是 messages。后续 `execute_all` 从 `state.plan_steps` 读取步骤列表，而不是从 messages 中解析。

---

### Step 3b: _execute_all — 顺序执行 4 个步骤

每步使用 `run_mini_react_loop()` 进行工具调用。这是理解 Plan-Solve 最容易出错的地方，需要仔细看代码：

```python
# plan_solve.py _execute_all 内部：
for i, step_desc in enumerate(state.plan_steps):
    ...
    msgs = [                                          # ← 每步新建一个 LOCAL 列表
        SystemMessage(content=prompt),
        HumanMessage(content=f"Execute step {i + 1}..."),
    ]
    msgs = await run_mini_react_loop(model, tools, msgs, max_rounds=3)
    # ↑ run_mini_react_loop 往这个 local msgs 里 append AIMessage / ToolMessage
    last = msgs[-1]
    results.append(str(last.content))                 # ← 只提取最后一条 AIMessage 的文字

return {
    "step_results": results,                          # ← 只有文本结果进 State
    "current_step": len(state.plan_steps),
}                                                    # ← 注意：没有 "messages" 键！
```

`msgs` 是 `_execute_all` 内部的局部变量。`run_mini_react_loop` 确实在里面 append 了 AIMessage(tool_calls) 和 ToolMessage，但这些消息**从未被返回给 LangGraph**——节点返回的字典里没有 `"messages"` 键。函数结束后 `msgs` 被垃圾回收。

而每步的最终文本被提取出来，存入了 `state.step_results`（一个 `list[str]`）。

```
   → PlanSolve._execute_all — 开始执行 4 个步骤
      → 执行步骤 1/4: ...（local msgs 内部：AIMessage(tool_calls) → ToolMessage → AIMessage 结果）
      → 执行步骤 2/4: ...（同上，另一个全新的 local msgs）
      → 执行步骤 3/4: ...（同上）
      → 执行步骤 4/4: ...（同上）
```

```
📦 state.messages = [
    [0] HumanMessage(content="I'm going to Shanghai this weekend..."),
    [1] SystemMessage(content="[Long-term memory...]\n1. User lives in Beijing"),
]
# ↑ 完全没变！_execute_all 不返回 messages，4 个步骤的 mini ReAct 过程是瞬时的
```

```
📦 state（非 messages 字段）:
    plan_steps = ["Determine transportation...", "Find 2-3 must-visit...", ...]
    current_step = 4
    step_results = [
        "The fastest option is the high-speed rail at 4.5 hours...",
        "Key attractions: The Bund, Yu Garden, French Concession...",
        "Day 1: Morning - Arrive Shanghai...",
        "Must-try: xiaolongbao at Jia Jia Tang Bao...",
    ]
```

> **关键点**：Plan-Solve 的 `_execute_all` 和 Supervisor 的 `_supervisor_researcher` / `_supervisor_executor` 都调用了同一个 `run_mini_react_loop()`，但前者把结果留在局部变量里只取文本，后者把结果通过 `{"messages": result_msgs}` 写入 State。这是**有意为之**——Plan-Solve 的中间搜索过程如果全进 messages，4 步 × 3 轮 = 12 条消息，messages 会迅速膨胀。只保留文本结果，大幅节省上下文。

---

### Step 3c: _aggregate — 汇总

`_aggregate` 从 `state.plan_steps` 和 `state.step_results` 读取数据构造 prompt，生成最终答案。**它是 Plan-Solve 子图中唯一向 state.messages 追加内容的节点。**

```
📍 [4/5] PlanSolve._aggregate — 汇总所有步骤为最终答案
```

```python
# _aggregate 内部逻辑（伪代码）：
plan_and_results = ""
for step, result in zip(state.plan_steps, state.step_results):
    plan_and_results += f"Step: {step}\nResult: {result}\n\n"

# 注意：是从 state.step_results 读，不是从 state.messages 读！
response = await model.ainvoke([SystemMessage(content=AGGREGATE_PROMPT), ...])

return {"messages": [response]}
```

```python
# 节点返回
{"messages": [AIMessage(content="# Shanghai Weekend Travel Plan\n\n## Transportation\nTake the high-speed rail from Beijing (4.5h, ~¥550)...\n\n## Day 1\n- Morning: Arrive, check in near Nanjing Road\n- Afternoon: The Bund, Yu Garden...\n\n## Day 2\n...")]}
```

```
📦 state.messages = [
    [0] HumanMessage(content="I'm going to Shanghai this weekend..."),
    [1] SystemMessage(content="[Long-term memory...]\n1. User lives in Beijing"),
    [2] AIMessage(content="# Shanghai Weekend Travel Plan\n\n## Transportation\n..."),  ← 唯一新增！aggregate 的汇总答案
]
```

---

### Step 4-5：extract_memory → 返回

```
最终 state.messages 长度: 3
   [0] HumanMessage      ← 用户输入
   [1] SystemMessage     ← 记忆注入（如果有）
   [2] AIMessage         ← aggregate 汇总的最终答案 ✅
```

> **对比**：Plan-Solve 的 messages 只有 2-3 条（不含中间工具调用），而 Supervisor 的 messages 包含全部专家的工具调用过程（10+ 条）。这是因为 `_execute_all` 没有把 mini ReAct 的消息写入 State——只提取了文本结果到 `step_results`。

---

# 示例 4：Supervisor 模式

**Query**：`"Find the latest population of Tokyo and calculate its population density (area: 2,194 km²)"`

**Router 判断**：`supervisor`（"Find X and calculate Y" → 搜索 + 计算）

---

### Step 0-2：inject_memory → router

同前。假设无相关记忆。

```
📦 state.messages = [
    [0] HumanMessage(content="Find the latest population of Tokyo and calculate its population density (area: 2,194 km²)"),
]
```

Router：`mode = "supervisor"`。

---

### Step 3: 进入 supervisor_agent 子图

```
__start__ → supervisor_decide → researcher → supervisor_review → executor → supervisor_review → __end__
```

---

### Step 3a: supervisor_decide — 首次决策

```
📍 [3/5] Supervisor._decide — 分析任务，选择第一个专家
   → 结构化输出: action=RESEARCH
   → 监督者决策: researcher
```

```python
{"messages": [AIMessage(content="[Supervisor → researcher] Search for the latest population of Tokyo (the city or metropolitan area) as of 2025-2026")],
 "supervisor_next_specialist": "researcher",
 "supervisor_iteration": 0}
```

```
📦 state.messages = [
    [0] HumanMessage(content="Find the latest population of Tokyo..."),
    [1] AIMessage(content="[Supervisor → researcher] Search for the latest population of Tokyo..."),  ← 监督者指令
]
```

`supervisor_next_specialist = "researcher"` → 条件边路由到 researcher。

---

### Step 3b: supervisor_researcher — 搜索

```
   → Supervisor._researcher — 执行搜索/信息收集
```

Researcher 使用 `run_mini_react_loop()` 搜索：

```python
# 返回后，messages 中追加了 search 的工具调用和结果
{"messages": [
    AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "Tokyo population 2025 2026 latest estimate"}, "id": "call_r1"}]),
    ToolMessage(content='{"results": [{"title": "Tokyo Population 2025", "content": "As of 2025, the Tokyo metropolitan area has an estimated population of 37.1 million..."}]}', tool_call_id="call_r1"),
    AIMessage(content="Latest data: Tokyo metropolitan population is approximately 37.1 million as of 2025. The 23 special wards (core Tokyo) have about 9.7 million."),
]}
```

```
📦 state.messages = [
    [0] HumanMessage(content="Find the latest population of Tokyo..."),
    [1] AIMessage(content="[Supervisor → researcher] Search for the latest..."),
    [2] AIMessage(content="", tool_calls=[{"name": "search", ...}]),           ← researcher 决定搜索
    [3] ToolMessage(content='{"results": [...]}', tool_call_id="call_r1"),      ← 搜索结果
    [4] AIMessage(content="Latest data: Tokyo metropolitan population is approximately 37.1 million..."),  ← researcher 总结
]
```

> **注意**：researcher 返回的消息过滤掉了内部的 HumanMessage prompt（见 supervisor.py `_supervisor_researcher` 的 `result_msgs` 过滤逻辑）。

---

### Step 3c: supervisor_review — 第 1 次评估

```
   → Supervisor._review — 评估进度 (迭代 1)
   → 结构化输出: action=EXECUTE
   → 监督者决策: EXECUTE (迭代 1/5)
```

监督者看到 researcher 拿到了人口数据，但还需要计算 → 委派 executor：

```python
{"supervisor_next_specialist": "executor",
 "supervisor_iteration": 1,
 "messages": [AIMessage(content="[Supervisor → executor] Calculate population density: 37.1 million / 2,194 km². Use python_repl.")]}
```

```
📦 state.messages = [
    [0] HumanMessage(content="Find the latest population of Tokyo..."),
    [1] AIMessage(content="[Supervisor → researcher] Search..."),
    [2] AIMessage(content="", tool_calls=[...]),
    [3] ToolMessage(content='{"results": [...]}'),
    [4] AIMessage(content="Latest data: Tokyo metropolitan population..."),
    [5] AIMessage(content="[Supervisor → executor] Calculate population density..."),  ← 监督者新指令
]
```

---

### Step 3d: supervisor_executor — 计算

```
   → Supervisor._executor — 执行计算/代码
```

Executor 使用 `run_mini_react_loop()` 调用 python_repl：

```python
{"messages": [
    AIMessage(content="", tool_calls=[{"name": "python_repl", "args": {"code": "37100000 / 2194"}, "id": "call_e1"}]),
    ToolMessage(content="16909.754", tool_call_id="call_e1"),
    AIMessage(content="The population density of Tokyo is approximately 16,910 people per square kilometer."),
]}
```

```
📦 state.messages = [
    [0]~[5] ... (之前的内容),
    [6] AIMessage(content="", tool_calls=[{"name": "python_repl", ...}]),   ← executor 决定计算
    [7] ToolMessage(content="16909.754", tool_call_id="call_e1"),            ← 计算结果
    [8] AIMessage(content="The population density of Tokyo is approximately 16,910 people/km²..."),
]
```

---

### Step 3e: supervisor_review — 第 2 次评估

```
   → Supervisor._review — 评估进度 (迭代 2)
   → 结构化输出: action=ANSWER
   → 监督者决策: ANSWER (迭代 2/5)
```

信息足够 → 监督者决定输出最终答案：

```python
{"supervisor_next_specialist": "FINISH",
 "supervisor_iteration": 2,
 "messages": [
     AIMessage(content="[Supervisor] Tokyo's metropolitan population is 37.1M..."),  ← 监督者口述
     AIMessage(content="# Tokyo Population Density Analysis\n\n..."),                  ← _synthesise_final_answer 生成的最终答案
 ]}
```

```
📦 state.messages = [
    [0] HumanMessage(content="Find the latest population of Tokyo..."),
    ... (中间的研究和计算过程),
    [9] AIMessage(content="[Supervisor] Tokyo's metropolitan population..."),   ← 监督者总结
    [10] AIMessage(content="# Tokyo Population Density Analysis\n\nTokyo has a metropolitan population of approximately 37.1 million...\n\nPopulation density: **16,910 people/km²**..."),  ← 最终答案 ✅
]
```

---

### Step 3f: _route_supervisor → __end__

```
📍 [4/5] Supervisor 完成 → __end__
```

---

### Step 4-5：extract_memory → 返回

```
最终 state.messages 长度: 11
   [0]  HumanMessage    ← 用户输入
   [1]  AIMessage       ← 监督者指令（→ researcher）
   [2]~[4]              ← researcher 的搜索过程
   [5]  AIMessage       ← 监督者指令（→ executor）
   [6]~[8]              ← executor 的计算过程
   [9]  AIMessage       ← 监督者口述
   [10] AIMessage       ← 最终综合答案 ✅
```

---

# 四种模式的 messages 特征对比

| 特征 | ReAct | Reflection | Plan-Solve | Supervisor |
|------|-------|-----------|------------|------------|
| messages 最终长度 | ~5 | ~5 | **~3**（极短） | ~11 |
| 包含 ToolMessage | ✅ | ❌（不绑工具） | **❌**（中间工具调用是瞬时的） | ✅ |
| 中间过程可见 | ✅（ToolMessage 在 messages 里） | ✅（审视意见在 messages 里） | **❌**（只在 local msgs + step_results 里） | ✅（专家输出在 messages 里） |
| 最终答案位置 | `[-1]` | `[-2]` 或 `[-1]` | `[-1]` | `[-1]` |
| 非 messages 的 State 字段 | 无 | `reflection_iteration` | `plan_steps` `step_results` `current_step` | `supervisor_next_specialist` `supervisor_iteration` |
| mini ReAct 消息去向 | 直接写入 messages | 不适用（无工具） | **局部变量，用完即弃** | 写入 messages |

---

# 核心要点

1. **messages 只增不减**。每个节点往里面追加，`add_messages` reducer 负责去重。整个对话历史完整保留在 messages 中。

2. **Router 不碰 messages**。它只设置 `mode` / `user_query` / `route_reason` 三个字段。

3. **子图的消息策略各不相同**。ReAct 直接把工具调用过程写入 messages；Reflection 把草稿和审视意见写入 messages；**Plan-Solve 的中间步骤是瞬时的**——`_execute_all` 在局部变量里跑 mini ReAct，只提取文本结果到 `step_results`，不碰 messages；Supervisor 把监督者指令和专家输出都写入 messages。

4. **非 messages 字段用于控制流**。`plan_steps`、`supervisor_next_specialist`、`reflection_iteration` 这些字段驱动子图内部的条件边，但它们不暴露给用户。

5. **extract_memory 是纯副作用**。它读 messages、写 Chroma，但不向 messages 追加任何内容。

6. **记忆注入是 SystemMessage**。`inject_memory` 把召回的历史事实包装成 SystemMessage 放到 messages[1]，这样 Agent 在后续推理中始终能看到它。
