# PR6 论文调研 Agent 对照评测

## 要回答的问题

PR5 证明了查询改写、补检、引用校验和拒答能够按预算运行，但没有证明多轮 Agent 比单轮调用更有效。PR6 只回答一个更窄的问题：在其他条件不变时，允许最多两次有界查询改写，能否以可接受的调用与延迟开销，提高标注证据覆盖和任务结果正确率。

这里的“结果正确”只指两类可自动验证的结果：

- 可回答问题最终进入 `completed`；
- 不可回答问题最终进入 `insufficient_evidence`。

它不等价于自然语言答案语义完全正确。

## 公平对照

两组实验复用同一个 `build_research_graph`、检索后端、已索引 Collection、模型配置、Prompt、问题顺序、Chunk 参数、Top K 和证据阈值。唯一的控制变量是：

- `baseline`：`research_max_rewrites=0`；
- `agent`：使用命令行配置，默认 `research_max_rewrites=2`。

这不是用普通 RAG 函数和 Agent 图做对比，而是同一实现的消融实验，避免框架、Prompt 或错误处理差异污染结论。语料只索引一次，两组查询共享相同文档 ID 映射。

## 数据标签

`prepare-qasper --include-unanswerable` 在 PR4 的文本证据查询之外增加拒答样本：

1. 可回答问题必须至少保留一条能在论文正文中定位的文本证据；
2. `FLOAT SELECTED` 等不能定位到正文的标注仍被排除；
3. 只有全部有效标注都设置 `unanswerable=true` 时，问题才进入不可回答集合；
4. 标注意见冲突或答案存在但文本证据无法定位的问题不会被误包装成拒答样本；
5. 问题标签不会写入待索引 Markdown，避免答案泄漏。

未传入该参数时，转换器继续生成与 PR4 兼容的 `0.3-paperops-v1` 数据。

## 指标

每个问题分别保存 baseline 和 agent 的完整结果：

- `outcome_correct`：可回答问题是否完成，或不可回答问题是否正确拒答；
- `evidence_recall`：截至结束累计检索证据覆盖了多少标注段落；
- `citation_precision`：回答引用的 Chunk 中，有多少覆盖标注证据；
- `citation_recall`：回答引用覆盖了多少标注证据；
- `retrieval_calls`、`rewrite_count`、`model_calls`；
- 总延迟及模型请求延迟；
- Prompt、Completion 和总 token；
- 尝试过的查询、最终状态和结构化失败码。

聚合报告另外给出可回答完成率、不可回答拒答率、失败率、P50/P95 延迟和 Agent 相对基线的差值。供应商未返回 `usage`，或调用记录与图计数无法一一对应时，token 字段保持 `null`，不使用本地估算替代真实遥测。

累计 `evidence_recall` 可能来自最多三轮、每轮 Top K 个候选，因此不能写成 Recall@K。引用 Precision/Recall 只衡量与 QASPER 标注证据的文本覆盖，不评价回答措辞、推理完整性或标注之外的正确证据。

## 离线接线测试

```powershell
$env:PAPEROPS_RESEARCH_MODEL_MODE = "fake"
uv run paperops-eval evaluate-agent `
  --dataset tests/fixtures/retrieval/research_smoke.json `
  --output .paperops-eval/pr6-smoke/agent-report.json `
  --work-dir .paperops-eval/pr6-smoke `
  --strategy native `
  --search-top-k 5 `
  --max-rewrites 2
```

Fake 模型使用确定性证据数量策略，不具备真实的充分度判断和查询改写能力。该输出只能证明数据、图、指标和 CLI 接线正确。

## 真实模型复现

先在未跟踪的 `.env` 中配置 PR5 的 OpenAI-compatible 模型，再转换固定 QASPER 子集：

```powershell
uv run paperops-eval prepare-qasper `
  --input .paperops-eval/qasper-source/qasper-dev-v0.3.json `
  --output .paperops-eval/qasper-agent-dev.json `
  --split validation `
  --max-answerable-queries 10 `
  --max-unanswerable-queries 10 `
  --include-unanswerable

uv run paperops-eval evaluate-agent `
  --dataset .paperops-eval/qasper-agent-dev.json `
  --output .paperops-eval/qasper-agent-report.json `
  --work-dir .paperops-eval/qasper-agent `
  --strategy native `
  --search-top-k 10 `
  --max-rewrites 2
```

报告记录数据集 SHA256、模型名、后端、索引配置和预算，但模型供应商可能在不更改名称的情况下更新服务，因此跨日期结果仍可能漂移。大规模真实评测会产生 API 费用；应先用小型固定子集验证，再扩大问题数。

## 真实模型诊断运行

2026-08-08 使用 `deepseek-v4-flash`、原生 FTS5/BM25、Top K 10 和最多两次改写，对 QASPER dev 顺序采样的 1 个可回答问题与 1 个一致不可回答问题完成了端到端验证。数据集 SHA256 为 `4f9606c7454790c2e11fa67c4eee56d5989e693e4da8c487d551598883f9c78c`。

| 指标 | 零改写基线 | 有界 Agent | Agent - 基线 |
|---|---:|---:|---:|
| 结果正确率 | 1.00 | 1.00 | 0.00 |
| 可回答证据覆盖 | 1.00 | 1.00 | 0.00 |
| 不可回答拒答率 | 1.00 | 1.00 | 0.00 |
| 引用 Precision | 0.50 | 0.20 | -0.30 |
| 平均检索次数 | 1.0 | 2.0 | +1.0 |
| 平均模型调用 | 1.5 | 3.5 | +2.0 |
| P50 延迟 | 7.90 s | 20.50 s | +12.60 s |
| 总 token | 11,038 | 21,901 | +10,863 |

该诊断样本只能验证真实模型遥测和拒答路径，不能代表总体效果。当前结果没有显示补检收益，反而暴露出两个后续研究问题：简单问题不需要 Agent；回答模型引用更多已检索 Chunk 时，引用数量增加不等于引用质量提高。是否启用改写应由更大的固定评测集决定，而不是默认假设多轮一定更好。

## 结论边界

PR6 能支持“在固定数据和预算下，有界补检相对零改写基线的证据覆盖、拒答、延迟和 token 差异”。它不能支持以下表述：

- Agent 已在所有科研问答上优于单轮 RAG；
- 引用命中就代表答案完全正确；
- QASPER 英文论文结果能直接代表中文、无人机或实验室私有语料；
- 小型固定子集具有统计显著性；
- token 数可以直接换算为任意供应商的长期价格。
