# PR5 可恢复论文调研 Agent

## 目标与非目标

PR5 把 PR4 的可评测检索能力用于一个具体任务：研究者针对已入库论文提出问题，系统在有限预算内寻找足够证据，并返回可解析到具体 Chunk 的回答。它不是通用聊天机器人，也不把“调用过 LLM”视为答案可信。

本阶段不包含多租户权限、分布式任务队列、自动访问互联网、训练或微调模型，也不宣称有界补检已经优于 PR4 的单轮检索基线。后续的同图消融协议与诊断结果见 [PR6 Agent 评测说明](pr6-agent-evaluation.md)。

## 两张图的职责

- Ingestion Graph：PDF 校验、MinerU 解析、规则质检、人工审批、切片入库、索引探测；
- Research Query Graph：检索、证据充分度判断、有界查询改写、补检、回答生成、引用校验或拒答。

两张图复用同一个 `RetrievalBackend` 协议与 SQLite checkpointer，但使用不同 State 和运行器。这样查询的模型异常不会改变已经入库的文档状态，摄取重试也不会重复执行某个用户问题。

## 控制权边界

LLM 通过 `ResearchModel` 协议只返回三类 Pydantic 结果：

1. `EvidenceAssessment`：充分度、置信度、理由和缺失信息；
2. `QueryRewrite`：下一条检索查询和改写理由；
3. `ResearchAnswer`：回答、引用 ID 和限制。

LangGraph 与普通代码保留控制权：

- 查询改写次数由配置硬限制，不能由模型自行增加；PR7 评测后生产默认值降为 `0`，实验消融显式使用 `2`；
- 没达到最小证据数时不浪费模型调用做充分度判断；
- 低于置信度阈值的“充分”判断被降级为继续补检；
- 重复查询立即停止，防止无效循环；
- Chunk 内容和累计证据都有字符上限，防止 checkpoint 和 Prompt 无界增长；
- 回答中的引用必须属于充分度判断选出的证据集合，正文含对应 `[E<n>]` 标记；仅在单个有效引用漏标记时做确定性补全，其他引用错误仍失败关闭；
- 预算耗尽返回 `insufficient_evidence`，不生成无依据答案。
- 检索或模型服务的可重试失败可通过 `/queries/{thread_id}/resume`
  精确返回失败节点，不重复已经完成的检索轮次。

## 本地验收

离线 Fake 模型用于验证图控制流，不代表真实问答质量：

```powershell
uv run pytest tests/unit_tests/test_research_graph.py -q
uv run pytest tests/unit_tests/test_research_model_client.py -q
uv run pytest tests/unit_tests/test_paperops_api.py -q
```

真实模型模式使用 JSON-mode chat completions：

```dotenv
PAPEROPS_RESEARCH_MODEL_MODE=openai_compatible
PAPEROPS_RESEARCH_MODEL_BASE_URL=https://api.openai.com/v1
PAPEROPS_RESEARCH_MODEL_API_KEY=replace-in-local-env-only
PAPEROPS_RESEARCH_MODEL_NAME=gpt-4o-mini
# 如果模型网关不能直连，可只为模型客户端指定代理：
# PAPEROPS_RESEARCH_MODEL_PROXY_URL=http://127.0.0.1:7890
```

启动 `uv run paperops-api`，完成论文入库后调用 `POST /queries`，再轮询返回的 `status_url`。如果服务在节点间中断，可调用 `POST /queries/{thread_id}/resume` 从 SQLite checkpoint 继续。

## 已覆盖的失败场景

- 初始检索无命中后改写并恢复；
- 两次改写后仍无证据并拒答；
- 模型充分度置信度低于阈值后继续补检；
- 模型生成不存在的引用后失败关闭；
- checkpoint 暂停恢复后不重复已完成检索；
- HTTP 模型响应不是 JSON 或不符合 Pydantic schema；
- API 完成查询后跨进程生命周期读取持久化结果。
- 真实模型返回临时错误后，从失败节点显式恢复并保留原审计轨迹。
