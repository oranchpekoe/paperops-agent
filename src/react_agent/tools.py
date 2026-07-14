"""Shared tool layer for the multi-mode agent framework.

All agent modes (ReAct, Reflection, Plan-Solve) share these tools through
a unified registry.  Add new tools here and they become available to every mode.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, List, cast

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from react_agent.context import Context

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool: web search (Tavily)
# ---------------------------------------------------------------------------


async def search(query: str) -> dict[str, Any] | None:
    """Search the web for current, factual information.

    Use this tool whenever you need up-to-date information, facts about
    recent events, or knowledge beyond your training data cutoff.

    Args:
        query: The search query string.  Be specific and include keywords.

    Returns:
        A dictionary with search results, or None if the search failed.
    """
    runtime = get_runtime(Context)
    # In server mode, runtime.context has the configured values.
    # In direct invocation, runtime.context is None → fall back to a sensible default.
    max_results = (
        runtime.context.max_search_results
        if (runtime is not None and runtime.context is not None)
        else int(os.environ.get("MAX_SEARCH_RESULTS", "5"))
    )
    wrapped = TavilySearch(max_results=max_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


# ---------------------------------------------------------------------------
# Tool: Python REPL (sandboxed)
# ---------------------------------------------------------------------------

# Restricted builtins for safety — only pure functions allowed
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "math": math,
}


def python_repl(code: str) -> str:
    """Execute a Python expression and return the result.

    Use this for calculations, data processing, or any task that requires
    computation.  Only pure expressions are supported — no file I/O, imports
    beyond the math module, or side effects.

    Args:
        code: A Python expression to evaluate (e.g. "sum(range(100))").

    Returns:
        The string representation of the result, or an error message.
    """
    try:
        result = eval(code, {"__builtins__": _SAFE_BUILTINS}, {})
        return str(result)
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# MCP tool support (lazy-loaded)
# ---------------------------------------------------------------------------

_mcp_tools: list = []
_mcp_loaded: bool = False


async def _ensure_mcp_loaded() -> None:
    """Lazy-load MCP tools once.

    Idempotent — the first call loads tools from configured MCP servers;
    subsequent calls are no-ops.
    """
    global _mcp_tools, _mcp_loaded
    if _mcp_loaded:
        return
    _mcp_loaded = True

    config = os.environ.get("MCP_CONFIG", "").strip()
    if not config:
        return

    try:
        from react_agent.mcp import load_mcp_tools

        _mcp_tools = await load_mcp_tools(config)
        if _mcp_tools:
            _logger.info("Loaded %d MCP tool(s)", len(_mcp_tools))
    except Exception as exc:
        _logger.warning(
            "MCP tool loading failed (continuing without MCP tools): %s", exc
        )


def get_all_tools() -> list:
    """Return all currently available tools: built-in + MCP (if loaded) + memory.

    Use this instead of accessing ``TOOLS`` directly when you need the
    full tool set at runtime.
    """
    tools = list(TOOLS) + list(_mcp_tools)

    # Lazy-load memory tools — they depend on Chroma which may not be
    # available in all environments.
    try:
        from react_agent.memory import recall, recall_all, remember

        mem_names = {t.__name__ for t in tools}
        for fn in (remember, recall, recall_all):
            if fn.__name__ not in mem_names:
                tools.append(fn)
    except ImportError:
        pass

    return tools


# ---------------------------------------------------------------------------
# RAG: local document retrieval (lazy-loaded Chroma vector store)
# ---------------------------------------------------------------------------

_retriever = None       # Chroma retriever, populated lazily
_docs_loaded = False    # Whether we've attempted loading


async def _load_documents() -> None:
    """Load documents from ``docs/`` and build a Chroma vector index.

    Idempotent — the first call scans the docs/ directory, chunks every
    ``.txt`` / ``.md`` / ``.pdf`` file, and builds a persistent Chroma
    vector store on disk (``.chroma_db/``).  Subsequent calls within the
    same session are no-ops.

    Across restarts, the persisted index is reused automatically — documents
    are only re-indexed when the ``.chroma_db/`` directory is deleted.
    """
    global _retriever, _docs_loaded
    if _docs_loaded:
        return
    _docs_loaded = True

    # Resolve paths relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    docs_dir = os.environ.get("DOCS_DIR", os.path.join(project_root, "docs"))
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", os.path.join(project_root, ".chroma_db"))

    if not os.path.isdir(docs_dir):
        _logger.info("RAG: docs/ directory not found at %s — skipping", docs_dir)
        return

    # Collect supported files
    from langchain_community.document_loaders import TextLoader

    docs = []
    for fname in sorted(os.listdir(docs_dir)):
        fpath = os.path.join(docs_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            if fname.endswith((".txt", ".md", ".markdown")):
                loader = TextLoader(fpath, encoding="utf-8")
                docs.extend(loader.load())
            elif fname.endswith(".pdf"):
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(fpath)
                    docs.extend(loader.load())
                except ImportError:
                    _logger.info("RAG: skipping %s (PyPDF not installed)", fname)
                    continue
        except Exception as exc:
            _logger.warning("RAG: failed to load %s: %s", fname, exc)
            continue

    if not docs:
        _logger.info("RAG: no loadable documents found in docs/")
        return

    # Chunk
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    _logger.info("RAG: split %d documents into %d chunks", len(docs), len(chunks))

    # Embed — use the same OpenAI-compatible API as the LLM
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_base=os.environ.get("OPENAI_BASE_URL"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Build or load persistent Chroma vector store
    import asyncio

    from langchain_chroma import Chroma

    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        # Reuse existing persisted index — skip re-indexing
        _logger.info("RAG: loading existing vector store from %s", persist_dir)
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="rag_docs",
        )
    else:
        # First run — build index from documents and persist to disk
        _logger.info("RAG: building new vector store → %s", persist_dir)
        db = await asyncio.to_thread(
            Chroma.from_documents,
            documents=chunks,
            embedding=embeddings,
            collection_name="rag_docs",
            persist_directory=persist_dir,
        )

    _retriever = db.as_retriever(search_kwargs={"k": 3})
    _logger.info("RAG: vector store ready with %d chunks", len(chunks))


async def retrieve(query: str, k: int = 3) -> str:
    """Search the local document store for relevant information.

    Use this tool when you need context from uploaded documents — project
    specs, knowledge base articles, manuals, or any content placed in the
    docs/ folder.

    Args:
        query: Natural-language search query.  Be specific.
        k: Number of document chunks to return (default 3, max 10).

    Returns:
        A formatted string with the top-k document excerpts (source and
        content), or a message indicating no documents are available.
    """
    k = max(1, min(k, 10))
    await _load_documents()

    if _retriever is None:
        return (
            "No documents are currently available.  Place .txt, .md, or .pdf "
            "files in the docs/ directory and restart to enable document retrieval."
        )

    docs = await _retriever.ainvoke(query)
    if not docs:
        return "No relevant documents found for that query."

    parts: list[str] = []
    for i, doc in enumerate(docs[:k], 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Document {i} — {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Mini ReAct loop — shared by Supervisor specialists and Plan-Solve executor
# ---------------------------------------------------------------------------


async def run_mini_react_loop(
    model,
    tools: list,
    messages: list,
    *,
    max_rounds: int = 3,
) -> list:
    """Execute a mini ReAct (Reason + Act) loop with tool access.

    The LLM decides whether to call a tool or return a final result.  Tools
    execute and results feed back into the model until no more tool calls are
    requested or *max_rounds* is exhausted.

    This is the core loop shared by:
    - ``supervisor.py``: ``_supervisor_researcher`` / ``_supervisor_executor``
    - ``plan_solve.py``: ``_execute_all`` step execution

    Parameters
    ----------
    model : BaseChatModel
        The model WITH tools already bound (``.bind_tools(tools)``).
    tools : list
        The tool list (used to build ``ToolNode``).
    messages : list
        The initial messages to send (system prompt + human message).
    max_rounds : int
        Maximum number of tool-calling rounds (default 3).

    Returns:
    -------
    list
        The full message list after the loop ends (may include tool responses).
    """
    from langgraph.prebuilt import ToolNode

    for _ in range(max_rounds):
        response = await model.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tool_node = ToolNode(tools)
        tool_result = await tool_node.ainvoke({"messages": [response]})
        for tm in tool_result.get("messages", []):
            messages.append(tm)

    return messages


# ---------------------------------------------------------------------------
# Tool registry (must be at the bottom — all tool functions above)
# ---------------------------------------------------------------------------

TOOLS: List[Callable[..., Any]] = [search, python_repl, retrieve]
"""All tools available to the agent.  Add new tool functions to this list."""


def get_tool_by_name(name: str) -> Callable[..., Any] | None:
    """Look up a tool by its function name."""
    for tool in get_all_tools():
        if getattr(tool, "__name__", "") == name:
            return tool
    return None
