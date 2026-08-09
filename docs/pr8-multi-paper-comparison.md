# PR8 多论文证据矩阵与缺口补检

## 1. 落地场景

实验室学生阅读同一主题的多篇论文时，常需要按固定字段整理方法差异，例如无人机多智能体强化学习论文的训练架构、状态/动作空间、奖励设计、实验环境和局限性。普通问答一次只返回一段自然语言，难以区分“论文没有报告”与“检索没有找到”，也不便检查不同论文的证据是否串用。

PR8 将任务限定为：对 2–8 篇已经通过 PaperOps 入库的论文，按 1–6 个用户明确给出的维度生成带 Chunk 引用的证据矩阵。它不自动发明比较维度，也不进行开放式网络调研。

## 2. 为什么需要执行图

单次模型调用可以抽取表格，但不能可靠承担以下控制职责：

- 每个论文—维度查询必须限制在所属文档；
- 每个矩阵单元格必须完整返回 `supported` 或 `missing`；
- `supported` 必须引用本论文已检索到的 Chunk；
- 低置信度结果降级为缺失，未知或跨论文引用失败关闭；
- 只有缺失单元格允许补检，已完成单元格不能重复消耗；
- 重复查询、无新增证据和轮数耗尽必须有不同停止原因；
- 初始矩阵、最终矩阵和补检轨迹需要进入 checkpoint，支持 API 查询与失败恢复。

LangGraph 在这里负责可恢复状态、条件路由和有界循环；语义模型只负责“从给定证据抽取指定字段并提出一个缺口查询”。

## 3. 状态机

```text
PENDING
  -> RETRIEVING_INITIAL   # 每篇论文 × 每个维度，文档内检索
  -> EXTRACTING           # 每篇论文一次结构化抽取
       -> COMPLETED       # 所有单元格已有证据
       -> RETRIEVING_GAPS # 只搜索 missing 单元格
            -> COMPLETED  # 无新证据、无新查询或预算耗尽
            -> EXTRACTING # 只重新抽取此前 missing 的维度
  -> FAILED
```

默认只允许一轮缺口补检。证据按 `document_id + chunk_id` 去重，并受单 Chunk 字符数和比较级总字符预算限制。

## 4. 类型化输出与引用约束

每个单元格包含：

- `document_id` 与 `dimension_id`；
- `supported` 或 `missing`；
- 支持时的 claim、置信度和至多 5 个引用 ID；
- 缺失时的原因与一个聚焦查询。

模型必须为每个请求维度返回且只返回一个单元格。服务校验引用是否属于当前论文的检索证据，并要求 claim 内出现引用标记；只有“单个合法引用仅漏写行内标记”这一无歧义情况允许确定性修复。

## 5. API

`POST /comparisons` 接收已入库文档 ID 和维度，异步执行后返回状态地址。`GET /comparisons/{thread_id}` 同时暴露：

- `initial_cells`：首次检索后的零补检基线；
- `cells`：有界补检后的最终矩阵；
- `attempted_searches` 与累计证据；
- 检索/模型调用数、缺口轮数和恢复单元格数；
- `all_cells_supported`、`stagnant_retrieval` 或 `gap_budget_exhausted`；
- 结构化失败与运行器错误。

可重试外部检索或模型异常可通过 `POST /comparisons/{thread_id}/resume` 从持久化节点边界继续。当前后台运行器仍是单 FastAPI 进程能力，不宣称多实例任务调度。

## 6. 配对评测协议

`paperops-eval evaluate-comparison-retrieval` 先将论文 × 维度标签投影为文档内检索任务，不调用 LLM，分别报告 Recall@K、MRR、NDCG 与延迟。`paperops-eval evaluate-comparison` 再使用同一矩阵评测结构化抽取和有界补检。每个 `supported` 标签必须提供独立标注的证据段落，`missing` 标签不得带证据。

为避免两次模型采样制造伪增益，两个变体共享：

1. 同一语料和索引；
2. 同一批文档级初始检索；
3. 同一次首次结构化抽取；
4. `extract_matrix` 后的同一个 checkpoint。

零补检基线在该 checkpoint 直接结算；Agent 分支只从此处继续检索缺失单元格。报告分别给出：

- 状态准确率与 `annotation_grounded_accuracy`；
- 可支持单元格完成率与应缺失单元格拒绝率；
- 证据覆盖率、引用 Precision/Recall；
- 失败率、停滞停止率；
- 检索、模型调用、延迟和供应商 token；
- 初始漏掉的可支持单元格、恢复数、恢复率和每次恢复的增量 token。

## 7. 固定 QASPER 诊断集

仓库不复制 QASPER 全量语料，只提交固定的论文、问题与维度映射。生成器从官方 QASPER JSON 保留原始论文正文、问题 ID、问题文本和人工证据段落；`source_query_id` 与 `source_question` 使每个比较标签可追溯。development profile 包含 6 篇论文、2 个任务和 15 个可支持单元格，用于选择检索后端；heldout profile 在选择前冻结，包含 5 篇不同论文和 10 个可支持单元格。

```bash
uv run paperops-eval prepare-qasper-comparison \
  --input .paperops-eval/qasper-source/qasper-dev-v0.3.json \
  --output .paperops-eval/qasper-comparison-development.json \
  --split validation \
  --profile development

uv run paperops-eval prepare-qasper-comparison \
  --input .paperops-eval/qasper-source/qasper-dev-v0.3.json \
  --output .paperops-eval/qasper-comparison-heldout.json \
  --split validation \
  --profile heldout
```

这里的 `annotation_grounded_accuracy` 比普通状态准确率更严格：`supported` 不仅要判断正确，引用还要覆盖对应 QASPER 问题的人工证据段落。它不是完整语义准确率，因为 QASPER 的问题证据并不穷举论文内所有可能支持同一比较维度的段落。

固定 profile 不再把 QASPER 的 `unanswerable` 直接映射为比较矩阵的 `missing`。问题级不可回答不能证明更宽泛的比较维度在整篇论文中不存在；强检索曾从这样的论文中召回有效模型对比，说明原负标签会制造伪错误。真实缺失项需要全文级人工审查，本 PR 只在可控测试中验证拒答与停止逻辑。

## 8. 当前验证结果

确定性测试构造了 2 篇论文 × 2 个维度的完整矩阵，其中一个可支持单元格必须经过聚焦查询才能命中。共享初始矩阵的 `annotation_grounded_accuracy` 为 `0.75`，一次补检后为 `1.00`；恢复 `1/1` 个初始漏检单元格，同时保留两个应缺失单元格。该测试验证控制流，不作为真实效果证据。

### 8.1 先选择检索底座

固定 `chunk=1200/overlap=160`，对 development 的 15 个文档内维度查询执行模型无关评测：

| 后端 | Recall@1 | Recall@3 | Recall@5 | MRR | P50 |
|---|---:|---:|---:|---:|---:|
| BM25 | 31.11% | 61.11% | 68.52% | 0.545 | 3.0ms |
| Dense | 33.70% | 59.26% | 90.00% | 0.600 | 7.9ms |
| Hybrid | 41.11% | 62.59% | 82.59% | 0.628 | 9.1ms |
| Hybrid+Rerank | 41.11% | **87.78%** | **95.19%** | **0.683** | 293.8ms |

Dense 并没有在 `top_k=3` 稳定超过 BM25，说明“换成向量库”本身不是解决方案。development 选择 Hybrid+Rerank 后，只在冻结 heldout 与 BM25 对照：

| heldout 后端 | Recall@1 | Recall@3 | Recall@5 | MRR | P50 |
|---|---:|---:|---:|---:|---:|
| BM25 | 5.00% | 40.00% | 58.00% | 0.327 | 2.2ms |
| Hybrid+Rerank | **35.00%** | **78.00%** | **78.00%** | **0.600** | 322.6ms |

### 8.2 再评价 Agent 增量

2026-08-08 的 BM25 配对诊断中，heldout 首次矩阵状态准确率为 `90%`，一次补检恢复唯一初始漏检后达到 `100%`，但 `annotation_grounded_accuracy` 只从 `50%` 增至 `60%`，额外消耗 `2,489` tokens 和约 `3.18s`。这证明补检循环能修复这个弱底座的一个已观察漏检，但不能证明弱底座是合理产品配置。

2026-08-09 使用 development 选定的 Hybrid+Rerank 重跑冻结 heldout：首次矩阵已经达到 `100%` 状态正确、`80%` 标注证据准确率和 `78%` 证据召回；5 次模型调用共消耗 `13,671` tokens，端到端约 `29.37s`。没有 `missing` 单元格，因此图没有补检，Agent 相对首次矩阵的检索、模型、token 和质量增量均为 `0`。

最终结论是：当前小样本中的主要效果提升来自检索底座，而不是 LangGraph 或多跑一轮模型。执行图仍负责文档隔离、结构化校验、持久化恢复、按缺口路由和安全停止；其补检分支是有界兜底，不作为默认必然提升点。该结论仍不具有统计显著性，也不代表业务收益。

## 9. 已知限制

- 当前输入必须使用已入库文档 ID，不负责自动选论文；
- 维度由用户明确提供，不做无约束的计划生成；
- QASPER 原始问题被人工映射为共享比较维度；这是可追溯诊断集，不是官方多论文比较 benchmark；
- 证据匹配衡量标注段落覆盖，可能漏计替代有效段落，也不代替 claim 的完整语义正确性；
- 一次抽取按论文聚合多个维度，极大矩阵不在本版本范围内；
- 当前固定真实 profile 只有可支持标签，尚未建立经全文审查的真实 `missing` 集合；
- 检索与全链路诊断样本都很小，因此不宣称普遍效果或业务收益；
- 不包含 Web 搜索、引用网络扩展、任务队列、多租户或 Agentic RL。
