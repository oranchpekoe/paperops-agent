# 上游来源与实现边界

## 上游

- 仓库：[langchain-ai/react-agent](https://github.com/langchain-ai/react-agent)
- 许可证：MIT，原版权声明保留在根目录 `LICENSE`
- 本仓库保留完整上游 Git 历史

## 已核对的开发基线

多模式功能进入前的基线提交为：

```text
d89d249c2a768e8b166c7c4e3bfeff2c73acb00f
```

该版本包含：

- `src/react_agent` Python包结构；
- Context、State、工具和模型加载辅助代码；
- `call_model -> tools -> call_model` 的基础 ReAct 图；
- LangGraph Studio配置及基础测试骨架。

## v0.1 多模式原型

提交 `9f6b752` 首次加入：

- `modes/react.py`：将基础 ReAct 流程抽取为子图；
- `modes/plan_solve.py`；
- `modes/reflection.py`；
- `modes/supervisor.py`；
- Mode Router与主图编排；
- MCP、RAG、三层记忆和流式输出；
- Benchmark、回归测试和故障复盘文档。

该提交相对基线增加约 8,860 行、删除约 182 行。后续修复继续保留在 Git 历史中。归档标签为：

```text
v0.1-multimode-demo
```

## PaperOps 产品化

PaperOps 不声称发明 ReAct、Plan-and-Solve、Reflection 或 Supervisor 等已有范式。产品化工作的贡献边界是：将通用范式展示项目重构为科研论文解析、质量审核、人工确认、知识库入库和检索验收的领域工作流，并通过可复现评测验证其效果。
