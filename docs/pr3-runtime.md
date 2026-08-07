# PR3 运行时与恢复边界

## 1. 组件职责

```text
HTTP Client
  -> FastAPI（上传限制、作业/审批接口）
  -> JobRunner（同一 thread_id 单飞）
  -> LangGraph（状态迁移、interrupt、结构化失败）
  -> SQLite Checkpointer（节点级状态）
  -> MinerU Client（外部解析任务与幂等恢复）
  -> RetrievalBackend
       -> Native：标题感知切片 + SQLite FTS5/BM25
       -> RAGFlow：可选外部适配器
```

- FastAPI 只接收 PDF 文件和逻辑 Collection ID，不接受任意服务器文件路径；
- `JobRunner` 防止同一进程内两个请求并发推进同一 LangGraph thread；
- Checkpoint 数据库保存小型状态和 artifact 路径，不保存 PDF、Markdown 正文或密钥；
- MinerU task manifest 保存外部任务 ID；
- 原生检索数据库保存文档元数据、Chunk 正文、稳定 Chunk ID和标题路径。

## 2. MinerU 接口契约

MinerU 使用当前官方自托管异步接口：

1. `POST /tasks` 上传单文件并要求 Markdown ZIP；
2. `GET /tasks/{task_id}` 轮询 `pending/processing/completed`；
3. `GET /tasks/{task_id}/result` 下载 ZIP；
4. 解压前检查目录穿越、符号链接、文件数量和总大小。

当首轮 `auto` 或 `txt` 质量检查明确失败时，第二轮切换为 `ocr`，避免用完全相同的解析参数重复请求。参考：[MinerU 官方用法](https://github.com/opendatalab/MinerU/blob/master/docs/en/usage/quick_usage.md)。

## 3. 原生检索契约

`NativeRetrievalBackend` 是真实模式的默认后端：

1. 按 Markdown 标题维护章节路径；
2. 每个 Chunk 只属于一个标题章节，章节内按字符上限和 overlap 切分；
3. 由 `idempotency_key` 派生稳定文档 ID，由正文和序号派生稳定 Chunk ID；
4. 文档记录与所有 Chunk 在同一 SQLite 事务中写入；
5. FTS5 使用 trigram tokenizer 兼容中英文子串，并通过 BM25 排序；
6. 查询始终应用 Collection 过滤，可选应用目标文档过滤。

PR3 只将它定位为关键词检索基线。稠密召回、融合和重排必须在 PR4 有独立标注评测后才能进入默认链路。

## 4. 可选 RAGFlow 适配器

设置 `PAPEROPS_RETRIEVAL_BACKEND=ragflow` 后，工作流可切换到 RAGFlow v1 文档上传、索引轮询和 `/retrieval` 接口。该适配器用于系统集成演示和后续同口径对比，不承担默认检索实现。

HTTP 契约单测通过 `httpx.MockTransport` 验证请求形状，不等同于已经在任意 RAGFlow 版本的真实部署上完成兼容性验收。

## 5. 中断恢复语义

| 中断位置 | 已持久化内容 | 恢复方式 |
|---|---|---|
| LangGraph 节点之间 | 最近 SQLite checkpoint | `POST /jobs/{thread_id}/resume` |
| 人工审核 | interrupt 与质量报告 | `POST /jobs/{thread_id}/approval` |
| MinerU 提交之后 | `mineru-task.json` 中的 task/status/result URL | 重放解析节点时继续轮询 |
| 原生索引写入之后 | 文档表与 FTS5 Chunk 的原子事务 | 重放时按 idempotency key 直接复用 |
| RAGFlow 上传之后 | 确定性文件名已存在于 Dataset | 重放时先查询再决定是否触发索引 |

节点内部异常会转换为 `WorkflowFailure`；框架级意外异常通过 API 的 `runtime_error` 暴露，同时保留上一个 checkpoint。

## 6. 当前不保证的能力

- SQLite 与内存 `JobRunner` 只支持单应用实例，不是分布式任务队列；
- 应用启动时不会自动扫描并执行所有未完成线程，恢复由调用方显式触发；
- 进程终止时正在进行的网络请求可能已被上游接收，因此恢复依赖幂等查询，而不是假设请求未发生；
- 当前文档级探测仅验收索引链路，不代表跨文档语义检索质量；
- API 尚未实现用户认证、Collection 级授权、配额和审计日志外发。
