# PaperOps MVP 产品说明

## 1. 问题定义

实验室成员整理科研 PDF 时，接口返回“解析成功”只能说明调用完成，无法证明标题、正文、公式和表格被可靠保留。即使文档已经写入知识库，不透明的切片、召回和排序策略也可能让关键证据无法命中。

PaperOps 将这两个问题放进同一条质量闭环：MinerU 负责专业 PDF 解析；确定性规则负责发现明确缺陷；不确定样本暂停等待人工确认；通过审核的 Markdown 由项目自身完成结构化切片、索引和文档级检索探测。PR5 另设调研查询链路，LLM 只负责无法由规则完成的证据判断、查询改写与答案综合，不参与文档入库质量门。

## 2. 第一版用户故事

> 作为一名需要整理无人机强化学习论文的实验室学生，我希望提交一篇 PDF 后，系统能够解析、检查、必要时重试或请求确认，并告诉我生成了哪些 Chunk、保存了哪些章节信息，以及索引是否能够返回该文档的证据。

## 3. 输入与输出

### 输入

- 一篇本地 PDF；
- 一个逻辑 Collection ID；
- 可选的人工审核意见。

### 输出

- 原文件哈希和任务 ID；
- MinerU 解析产物路径；
- 规则质量决策；
- 重试、异常和人工审批记录；
- 索引文档 ID和 Chunk 数量；
- 文档级索引探测问题和证据片段；
- 结构化验收报告。

## 4. 后端边界

默认真实装配为：

```text
MinerUClient -> heading-aware chunking -> NativeRetrievalBackend
                                      -> SQLite FTS5 / BM25
```

- MinerU 是外部专业解析基础设施；
- 标题边界、Chunk 大小、重叠和元数据由 PaperOps 控制；
- SQLite FTS5/BM25 是 PR3 的透明关键词检索基线；
- RAGFlow 仅作为可选 `RetrievalBackend` 适配器和后续对比对象，不是默认实现。

## 5. PR3 单文档状态机

```text
PENDING
  -> PARSING
  -> QUALITY_CHECK
       -> RETRY -> PARSING
       -> WAITING_APPROVAL -> INDEXING
       -> INDEXING
  -> RETRIEVAL_EVAL
  -> COMPLETED | FAILED
```

`INDEXING` 在原生后端中表示“切片并写入索引”，在可选外部后端中表示“提交并等待远端索引”。FastAPI 用 `thread_id` 标识执行线程，SQLite Checkpointer 保存节点级状态；大段正文仍然只保存在 artifact 文件中。

## 6. 验收条件

- 正常文档能完成完整状态流转；
- 低质量解析最多自动重试两次；
- 无法可靠判断时进入人工审核状态；
- 外部服务和索引异常被记录为结构化错误；
- 大段 Markdown 不写入 LangGraph State；
- 同一个文件哈希和 Collection 不重复创建索引文档；
- MinerU 任务提交后持久化 task manifest，重放时继续轮询已有任务；
- 切片不跨越 Markdown 标题定义的章节边界；
- Chunk 保存稳定 ID、顺序和标题路径；
- Collection 与文档过滤在检索层生效；
- 进程重启后能够查询等待审批的 checkpoint 并继续执行；
- 上传大小、PDF 签名及 MinerU ZIP 路径/解压大小均经过校验；
- 每条条件分支均有自动化测试。

## 7. PR3 探测结果的含义

当前文档级问题使用同一篇文档的标题生成，并限定目标文档进行检索。它只能证明：

- Markdown 被切成了可检索 Chunk；
- 索引写入和过滤条件有效；
- 检索调用可以返回目标文档证据。

它不能证明跨文档语义检索质量，也不能替代独立标注的 Recall@K、MRR 或 nDCG 评测。因此 API 和报告把它称为 `index_probe`，不把它包装成最终 RAG 效果。

## 8. 暂不包含

- 批量并发调度；
- Redis、Kubernetes 和独立任务队列；
- FastAPI 多进程/多实例调度与启动时自动恢复；
- 多租户、RBAC 和按 Collection 授权；
- 大规模 ANN 向量索引与分布式检索；
- LLM 语义质量审核；
- Agentic RL 训练；
- 通用聊天、长期偏好记忆和四模式自动路由。

## 9. PR4 检索评测

使用公开 QASPER 论文问答数据建立查询—相关证据标注，比较：

1. FTS5/BM25 关键词基线；
2. 稠密向量召回；
3. BM25 + 稠密召回 + RRF；
4. 融合召回 + Cross-Encoder 重排。

主要指标为 Recall@K、MRR、nDCG、查询延迟和索引耗时。人工介入率与故障恢复率属于工作流评测，不与检索排序指标混合。当前诊断结果证明融合和重排有质量收益，但重排 CPU 延迟明显，因此默认链路继续使用 BM25，复杂策略通过配置显式启用。数据规则、结果与限制见 [PR4 检索评测说明](pr4-retrieval-evaluation.md)。

## 10. PR5 调研查询状态机

```text
PENDING -> RETRIEVING -> ASSESSING
                         -> ANSWERING -> COMPLETED | FAILED
                         -> REWRITING -> RETRIEVING
                         -> INSUFFICIENT_EVIDENCE
```

一次查询最多执行初始检索加两轮补检。每轮证据按文档与 Chunk 去重，并受单 Chunk 和总字符预算约束；模型判定充分但置信度低于阈值时仍视为证据不足。回答节点只接收充分度判断选出的相关证据；未知引用、重复引用或位置不明确的多个缺失标记都会失败关闭，只有单个有效引用漏标记时允许确定性补全。证据预算耗尽或改写后没有新增证据时不生成猜测答案。

## 11. PR6 Agent 对照评测

在同一份带证据与拒答标签的数据上，将 PR5 Query Graph 分别固定为零次改写和最多两次改写。两组共享模型、Prompt、检索后端、索引、Chunk 参数和 Top K，只比较有界补检带来的证据覆盖与拒答收益，以及检索轮数、模型调用、延迟和 token 增量。

该评测不使用自定义综合分数。可回答完成率与不可回答拒答率分别报告；证据覆盖、引用 Precision/Recall、失败率和资源开销保留为独立指标。详细协议和结论边界见 [PR6 Agent 评测说明](pr6-agent-evaluation.md)。

## 12. PR7 自适应停止与论文级配对评测

充分度判断同时选择最小相关证据集，回答节点不再接收全部累计候选。改写后若没有新增去重证据，图以 `stagnant_retrieval` 提前拒答；重复查询和预算耗尽使用不同停止原因，均通过 API 与评测报告暴露。

QASPER 查询按所属论文限制检索。基线与 Agent 共享初始检索和充分度判断，只在初始证据不足时让 Agent 从 checkpoint 继续补检，以减少独立模型采样造成的伪差异。当前真实 3+3 诊断没有观察到改写收益，因此默认配置保留有界能力，但不宣称其已优于零改写基线。实现、指标和负结果见 [PR7 自适应停止与配对评测](pr7-adaptive-research.md)。
