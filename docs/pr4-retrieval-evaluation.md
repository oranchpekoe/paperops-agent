# PR4 检索评测与策略选择

> 历史说明：本页的 50 篇/148 问题结果来自 PR4 的 `0.3-paperops-v1` Collection 级数据。PR7 发现 QASPER 问题实际绑定单篇论文，新转换器已生成带 `document_id` 的 v3 数据并限制到所属论文检索。旧报告仍可用于记录当时的策略探索，但不能与 v3 论文级指标直接比较；迁移原因见 [PR7 自适应停止与配对评测](pr7-adaptive-research.md)。

PR4 的目标不是把向量数据库接到工作流后就宣称“支持 RAG”，而是让同一份论文语料、问题和人工证据标签在不同检索策略下可重复比较。默认运行链路仍使用 SQLite FTS5/BM25；稠密检索、RRF 融合和交叉编码器重排均为显式可选项。

## 实现边界

- Sparse：结构感知切片 + SQLite FTS5/BM25；
- Dense：相同 Chunk ID 和切片参数 + FastEmbed bi-encoder + `sqlite-vec` 精确 KNN；
- Hybrid：分别取 Sparse/Dense 候选，以 Reciprocal Rank Fusion 合并；
- Hybrid + Reranker：只对前 20 个融合候选运行 cross-encoder，限制推理开销；
- Evaluation：独立的数据转换、证据匹配、Recall@K、MRR、nDCG@K、p50/p95 延迟和逐查询命中轨迹。

`sqlite-vec` 当前在这里执行的是过滤后的暴力精确 KNN，而不是 ANN。它适合本项目的单机可复现实验，不应据此宣称已经解决百万级索引扩展问题。

## 数据集与标签

评测使用 [QASPER v0.3](https://huggingface.co/datasets/allenai/qasper) 的 dev split。QASPER 面向科研论文问答，并提供由标注者选出的证据段落；数据集论文见 [NAACL 2021](https://aclanthology.org/2021.naacl-main.365/)，许可为 CC BY 4.0。

转换器执行以下确定性规则：

1. 跳过不可回答问题和 `FLOAT SELECTED` 图表占位证据；
2. 只保留能在论文正文中精确定位的文本证据；
3. 多位标注者选择同一证据时，将同意数作为相关性等级，最高为 3；
4. 评测标签不写入待索引 Markdown，避免答案泄漏；
5. Chunk 覆盖一个证据段落至少 60% 的 token，才判定该证据被召回。

当前转换器还会保存问题所属的论文 ID，评测请求只在该论文内召回；无作用域的 v1 历史数据仍可读取，但不会被静默改写。

仓库只提交三篇自编内容构成的 `smoke_fixture`，用于验证评测接线。它不能作为简历或报告中的效果数据。QASPER 原始数据和生成的报告均写入被忽略的 `.paperops-eval/`。

## 可复现命令

安装本地检索模型依赖：

```bash
uv sync --extra retrieval-models
```

将已下载的官方 QASPER JSON 转换为 PaperOps 格式：

```bash
uv run paperops-eval prepare-qasper \
  --input .paperops-eval/qasper-source/qasper-dev-v0.3.json \
  --output .paperops-eval/qasper-dev-50.json \
  --split dev \
  --max-documents 50
```

运行一种策略；`STRATEGY` 可取 `native`、`dense`、`hybrid`、`hybrid-reranked`：

```bash
uv run paperops-eval evaluate \
  --dataset .paperops-eval/qasper-dev-50.json \
  --output .paperops-eval/qasper-dev-50-STRATEGY.json \
  --work-dir .paperops-eval/qasper-dev-50-models \
  --strategy STRATEGY \
  --top-k 1,3,5,10 \
  --candidate-k 20
```

报告记录数据集 SHA256、后端名称、模型索引配置、切片参数和逐查询结果。索引幂等键也包含数据集、切片和模型指纹，防止复用旧索引污染结果。BM25、Dense、RRF 与重排均为同分候选设置稳定的 Chunk ID 决胜规则；连续两次完整运行的 Recall、MRR 和 nDCG 逐项一致。

## 当前诊断结果

固定配置：QASPER dev 前 50 篇论文、148 个有文本证据的问题、Chunk 1200 字符、Overlap 160 字符、候选数 20。数据集 SHA256 为 `1d526c0b2374bfdafe7f47d97c454b8766f48cc83983ac4a550dac3a3c26b5df`。

| 策略 | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 延迟 |
|---|---:|---:|---:|---:|---:|---:|
| FTS5/BM25 | 0.112 | 0.267 | 0.382 | 0.270 | 0.232 | 12.9 ms |
| BGE-small Dense | 0.084 | 0.210 | 0.371 | 0.224 | 0.209 | 9.3 ms |
| BM25 + Dense RRF | 0.109 | 0.314 | 0.429 | 0.278 | 0.253 | 17.0 ms |
| Hybrid + MiniLM rerank | 0.143 | 0.386 | 0.479 | 0.333 | 0.301 | 374.7 ms |

这里的模型是 CPU 友好的英文诊断配置：`BAAI/bge-small-en-v1.5` 和 `Xenova/ms-marco-MiniLM-L-6-v2`。延迟为单台 Windows 开发机单次运行结果，只用于量级判断。

## 结论与限制

Dense 单路没有超过 BM25，因此项目不把“向量化”本身当作优化成果。Sparse 与 Dense 的互补候选使 Hybrid 的 Recall@10 相对 BM25 提升约 12.3%；重排继续提高排序质量，但 p50 延迟约为 Hybrid 的 22 倍。因此：

- 默认使用 `native`，保证最小依赖和可解释基线；
- 对召回率更敏感时显式选择 `hybrid`；
- 仅在允许数百毫秒 CPU 推理预算时选择 `hybrid_reranked`；
- QASPER 子集是英文科研问答诊断集，不代表中文、无人机领域或全量数据表现；
- 下一轮应扩展到完整 dev split，并补充目标领域论文问题集、Chunk 参数消融和批量并发压测。

真实服务配置示例：

```dotenv
PAPEROPS_CLIENT_MODE=real
PAPEROPS_RETRIEVAL_BACKEND=hybrid
PAPEROPS_RETRIEVAL_CANDIDATE_K=20
PAPEROPS_RETRIEVAL_RRF_K=60
```

首次选择模型后端时 FastEmbed 会下载模型。生产部署应在构建阶段预热缓存，而不是让服务首次请求承担下载成本。
