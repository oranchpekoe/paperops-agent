# Multi-Mode Agent Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-green.svg)](https://github.com/langchain-ai/langgraph)

A production-oriented multi-mode AI Agent built on LangGraph, featuring **mode routing**, **MCP protocol integration**, **RAG document retrieval**, and **Supervisor-Worker multi-agent coordination**. Designed as a showcase project for AI Agent engineering internships.

---

## Architecture Overview

```
                        ┌─────────────────────┐
                        │     User Message     │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │    Mode Router       │
                        │  (LLM classifier)    │
                        └──────────┬──────────┘
                                   │
          ┌────────────┬───────────┼───────────┬──────────────┐
          │            │           │           │              │
   ┌──────▼─────┐ ┌───▼────┐ ┌───▼─────┐ ┌───▼───────┐
   │   ReAct    │ │Reflect │ │Plan-    │ │Supervisor │
   │  (简单问答) │ │(写作分析)│ │Solve    │ │(多Agent) │
   │            │ │        │ │(多步骤)  │ │           │
   └────────────┘ └────────┘ └─────────┘ └───────────┘
```

Four agent architectures behind a single unified entry point. An LLM-based Mode Router analyses each incoming query and delegates to the most appropriate mode.

### Mode Comparison

| Mode | Best For | Pattern | Tool Access |
|------|----------|---------|-------------|
| **ReAct** | Simple Q&A, factual lookups, single-step tasks | Reason → Act → Observe → Repeat | ✅ Full |
| **Reflection** | Writing, analysis, code review, complex reasoning | Generate → Critique → Refine (×3) | ❌ Pure reasoning |
| **Plan-Solve** | Multi-step problems, math, travel planning | Plan → Execute each step → Aggregate | ✅ Full |
| **Supervisor** | Multi-domain tasks mixing search + computation | Decide → Delegate to specialists → Review → Repeat | ✅ Per-specialist |

---

## Key Features

### 1. Mode Router — Intelligent Query Classification

An LLM-powered router at the graph entry point analyses every user message and selects the best agent architecture. Not regex-based — it uses the same LLM to understand query semantics.

### 2. MCP Protocol Integration

Supports the **Model Context Protocol** for dynamic tool loading from external servers. Configure MCP servers via the `MCP_CONFIG` environment variable and tools are loaded lazily at runtime — no restart required.

```json
{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    "transport": "stdio"
  }
}
```

- Eager loading on first tool use, idempotent thereafter
- Individual server failures don't crash the agent (graceful degradation)
- Zero-config: no `MCP_CONFIG` → agent works normally with built-in tools

### 3. RAG — Local Document Retrieval

Drop `.txt`, `.md`, or `.pdf` files into the `docs/` directory and the agent can search them. Powered by:

- **Chroma** vector store (in-memory, no external DB)
- **OpenAI text-embedding-3-small** (or any OpenAI-compatible embedding model)
- **RecursiveCharacterTextSplitter** (1000-char chunks, 200-char overlap)
- Lazy loading with singleton pattern — documents indexed once, reused across all queries

### 4. Supervisor-Worker Multi-Agent Coordination

A supervisor LLM orchestrates three specialist agents:

```
Supervisor
   ├── Researcher  — search + web tools (gathers facts)
   ├── Analyst     — pure reasoning, no tools (critiques, evaluates)
   └── Executor    — python_repl + computation (runs calculations)
```

**Execution loop:**
1. **Decide** — Supervisor analyses task, picks first specialist
2. **Delegate** — Specialist runs (with up to 3 rounds of ReAct if it has tools)
3. **Review** — Supervisor evaluates all output, decides: `RESEARCH | EXECUTE | ANALYSE | ANSWER`
4. **Repeat** (up to 5 iterations) or **Finish** with a synthesised final answer

Example for *"Research Japan's 2024 GDP and calculate 5% of it"*:
```
Router → supervisor
  Decide → "RESEARCH"  → Researcher searches GDP data
  Review → "EXECUTE"   → Executor calculates 5%
  Review → "ANSWER"    → Supervisor synthesises final answer
```

---

## Project Structure

```
react-agent/
├── src/react_agent/
│   ├── graph.py              # Main orchestrator (router + 4 subgraphs)
│   ├── state.py              # Shared state definitions
│   ├── tools.py              # Tool registry (search, python_repl, retrieve + MCP)
│   ├── mcp.py                # MCP client wrapper
│   ├── context.py            # Runtime context / configuration
│   ├── prompts.py            # Default system prompts
│   ├── utils.py              # Model loading helpers
│   └── modes/
│       ├── react.py          # ReAct subgraph
│       ├── reflection.py     # Reflection subgraph (Generate → Critique → Refine)
│       ├── plan_solve.py     # Plan-Solve subgraph (Plan → Execute → Aggregate)
│       └── supervisor.py     # Supervisor-Worker subgraph (multi-agent)
├── docs/
│   └── project-overview.md   # Sample document for RAG testing
├── tests/
│   ├── test_trace.py         # End-to-end trace test (4 query types)
│   ├── unit_tests/           # Unit tests (configuration, etc.)
│   └── integration_tests/    # Integration tests (graph compilation, routing)
├── pyproject.toml
├── .env.example
└── langgraph.json
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Tavily API key](https://tavily.com) for web search
- An LLM API key (OpenAI-compatible by default)

### Setup

```bash
# 1. Clone and navigate
cd react-agent

# 2. Install dependencies (uv recommended)
pip install uv
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. (Optional) Install MCP support
pip install langchain-mcp-adapters

# 5. (Optional) Add documents for RAG
mkdir docs
echo "# My Knowledge Base" > docs/project-info.md
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | LLM API key (aihubmix or any OpenAI-compatible) |
| `OPENAI_BASE_URL` | ✅ | API base URL (default: `https://aihubmix.com/v1`) |
| `MODEL` | ✅ | Model name, e.g. `openai/deepseek-v4-flash` |
| `TAVILY_API_KEY` | ✅ | Tavily search API key |
| `MCP_CONFIG` | ❌ | JSON string or file path for MCP servers |
| `DOCS_DIR` | ❌ | Custom docs directory (default: `./docs`) |
| `EMBEDDING_MODEL` | ❌ | Embedding model (default: `text-embedding-3-small`) |
| `MAX_SEARCH_RESULTS` | ❌ | Max search results per query (default: 5) |

### Running

**LangGraph Studio (recommended for development):**
```cmd
REM Windows — two separate commands
set PYTHONUTF8=1
langgraph dev --port 1024 --allow-blocking
```

**CLI / direct invocation:**
```bash
python tests/test_trace.py
```

**Programmatic use:**
```python
from react_agent.graph import graph

result = await graph.ainvoke({
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
})
print(result["messages"][-1].content)
```

---

## Testing

```bash
# Full trace test (4 query types across all modes)
python tests/test_trace.py

# Unit tests
pytest tests/unit_tests/ -v

# Integration tests
pytest tests/integration_tests/ -v
```

---

## Design Decisions

### Why `--allow-blocking`?
The `langgraph dev` ASGI server detects synchronous blocking calls to protect the event loop. Our `python_repl` tool uses Python's `eval()` synchronously, and `Chroma.from_documents()` internally calls `tiktoken` → `os.getcwd()`. Both trigger the block detector. We've wrapped the Chroma call in `asyncio.to_thread()`, but `eval()` is CPU-bound and doesn't benefit from threading. The `--allow-blocking` flag is LangGraph's intended escape hatch — production deployments using `langgraph serve` with dedicated workers don't face this constraint.

### Why Text-Based Supervisor Parsing?
The initial implementation used structured text parsing (prefix matching + keyword fallback) for supervisor decisions. This works but is fragile — different LLMs or query phrasings can produce unexpected output formats, causing the parser to default to `FINISH`. A migration to **structured output** (Pydantic models with `with_structured_output()`) is planned to guarantee parseable decisions regardless of model behaviour.

### Why All Tools in Every Mode?
Rather than restricting tools per mode, all tools are available everywhere. The LLM's system prompt guides *which* tools to use when. This keeps the tool registry simple and lets the LLM exercise judgment — a Researcher might still need `python_repl` for a quick calculation within a research task.

---

## License

MIT
