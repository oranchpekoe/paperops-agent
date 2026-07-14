"""Memory management — cross-session fact storage and context compression.

Three complementary memory layers
----------------------------------
1. **Short-term** — current-session message history (existing ``add_messages``
   reducer in ``MainState``).  No changes needed here.
2. **Long-term** — cross-session fact store backed by Chroma.  Facts are
   embedded and stored in the ``agent_memory`` collection (same ``.chroma_db/``
   directory as RAG, different collection).  The agent can explicitly
   ``remember`` and ``recall`` facts via tools, and the graph can auto-extract
   facts after each conversation.
3. **Summary** — when the message list grows too long, older messages are
   compressed into a single summary paragraph by the LLM.  This is a sliding
   window: keep the most recent N messages verbatim, summarise the rest.

All memory components degrade gracefully — if Chroma or the embedding API is
unavailable the agent continues with only short-term memory.
"""

from __future__ import annotations

import contextvars
import logging
import os
from datetime import UTC, datetime

from react_agent.utils import get_message_text

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context variable — allows eval runners to set benchmark mode globally
# without threading a ``benchmark_mode`` flag through every function signature.
# ---------------------------------------------------------------------------

_benchmark_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "benchmark_mode", default=False
)

# ---------------------------------------------------------------------------
# Long-term memory store (Chroma)
# ---------------------------------------------------------------------------

_memory_store: MemoryStore | None = None
_memory_loaded: bool = False


class MemoryStore:
    """Chroma-backed vector store for cross-session facts.

    Each fact is stored as a Document whose ``page_content`` is the fact text
    and whose metadata carries a timestamp and optional tags.  Similarity
    search over embeddings lets us recall the most relevant facts even when
    the user doesn't phrase the query identically.

    Uses the same persistence directory as the RAG vector store
    (``.chroma_db/``) but a separate ``agent_memory`` collection so the two
    don't interfere.
    """

    def __init__(self, persist_dir: str, embeddings) -> None:
        """Initialise the Chroma-backed memory store."""
        from langchain_chroma import Chroma

        self._db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="agent_memory",
        )

    # -- write path ----------------------------------------------------------

    async def store(self, fact: str, metadata: dict | None = None, *, dedup: bool = True) -> str:
        """Store a single fact and return its document ID.

        When *dedup* is True (default), a similarity check is performed first
        and the fact is skipped if a near-identical one already exists
        (cosine distance < 0.05).  This prevents the same fact from being
        stored multiple times across sessions.
        """
        import uuid

        from langchain_core.documents import Document

        fact_text = fact.strip()

        # ── dedup: check for near-identical existing facts ──────────
        if dedup:
            try:
                existing = await self._db.asimilarity_search_with_score(
                    fact_text, k=3
                )
                for doc, score in existing:
                    # Chroma returns cosine *distance* — lower = more similar
                    if score < 0.05:
                        _logger.info(
                            "Memory: skipping duplicate fact (score=%.4f, "
                            "existing_id=%s): '%s'",
                            score,
                            doc.metadata.get("id", "?"),
                            fact_text[:60],
                        )
                        return doc.metadata.get("id", "skipped_duplicate")
            except Exception:
                pass  # Dedup is best-effort; don't block storage on failure

        meta = dict(metadata or {})
        meta.setdefault("timestamp", datetime.now(tz=UTC).isoformat())
        meta.setdefault("type", "user_fact")

        doc = Document(page_content=fact_text, metadata=meta)
        doc_id = str(uuid.uuid4())[:8]
        meta["id"] = doc_id
        await self._db.aadd_documents([doc], ids=[doc_id])
        _logger.info("Memory: stored fact '%s' (id=%s)", fact_text[:60], doc_id)
        return doc_id

    async def store_many(self, facts: list[str]) -> list[str]:
        """Store multiple facts in batch.  Returns their IDs."""
        ids = []
        for fact in facts:
            fact_id = await self.store(fact)
            ids.append(fact_id)
        return ids

    # -- read path -----------------------------------------------------------

    async def recall(self, query: str, k: int = 5, *, score_threshold: float = 0.6) -> list[dict]:
        """Return the top-*k* facts most relevant to *query*.

        Each result is a dict with ``content`` and ``metadata`` keys.

        Only returns facts whose cosine distance is below *score_threshold*
        (lower = more similar).  This prevents semantically unrelated facts
        from being injected as "relevant memory" (e.g. "LangChain founder"
        recalled for a travel query).
        """
        docs_with_scores = await self._db.asimilarity_search_with_score(
            query, k=k
        )
        return [
            {"content": d.page_content, "metadata": d.metadata}
            for d, score in docs_with_scores
            if score < score_threshold
        ]

    async def recall_all(self, k: int = 20) -> list[dict]:
        """Return the *k* most recently stored facts (fallback when no query)."""
        results = self._db.get(limit=k)
        if not results or "documents" not in results:
            return []
        docs = results["documents"]
        metadatas = results.get("metadatas") or [{}] * len(docs)
        out: list[dict] = []
        for content, meta in zip(docs, metadatas):
            # Chroma get() returns raw strings for content
            out.append({"content": str(content), "metadata": dict(meta)})
        return out

    async def delete(self, fact_id: str) -> bool:
        """Delete a fact by ID.  Returns ``True`` on success."""
        try:
            await self._db.adelete([fact_id])
            return True
        except Exception:
            return False

    async def clear_all(self) -> int:
        """Delete all facts.  Returns the number of documents removed."""
        try:
            all_docs = self._db.get()
            count = len(all_docs.get("ids", []))
            if count > 0:
                self._db.delete(ids=all_docs["ids"])
            return count
        except Exception:
            return 0

    async def clear_contaminated(self, patterns: list[str]) -> int:
        """Remove facts whose content contains any of the given *patterns*.

        Returns the number of facts deleted.  Useful for cleaning up benchmark
        data that was accidentally stored as user facts.
        """
        try:
            all_docs = self._db.get()
            if not all_docs or "documents" not in all_docs:
                return 0

            ids_to_delete: list[str] = []
            for doc_id, content in zip(all_docs.get("ids", []), all_docs["documents"]):
                content_lower = str(content).lower()
                for pattern in patterns:
                    if pattern.lower() in content_lower:
                        ids_to_delete.append(doc_id)
                        _logger.info(
                            "Memory: flagged contaminated fact (id=%s): '%s'",
                            doc_id,
                            str(content)[:80],
                        )
                        break

            if ids_to_delete:
                self._db.delete(ids=ids_to_delete)
                _logger.info(
                    "Memory: removed %d contaminated facts", len(ids_to_delete)
                )
            return len(ids_to_delete)
        except Exception as exc:
            _logger.warning("Memory: clear_contaminated failed: %s", exc)
            return 0


# ---------------------------------------------------------------------------
# Lazy loader (idempotent — call freely)
# ---------------------------------------------------------------------------


async def _ensure_memory_loaded() -> MemoryStore | None:
    """Return the singleton ``MemoryStore``, creating it on first call.

    Degrades gracefully: returns ``None`` when Chroma or the embedding API is
    unavailable so callers can fall back to short-term memory only.
    """
    global _memory_store, _memory_loaded
    if _memory_loaded:
        return _memory_store
    _memory_loaded = True

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    persist_dir = os.environ.get(
        "CHROMA_PERSIST_DIR", os.path.join(project_root, ".chroma_db")
    )

    try:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_base=os.environ.get("OPENAI_BASE_URL"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
        _memory_store = MemoryStore(persist_dir, embeddings)
        _logger.info("Memory: store ready at %s", persist_dir)
        return _memory_store

    except Exception as exc:
        _logger.warning(
            "Memory: store unavailable — continuing with short-term memory only. "
            "Reason: %s",
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Context compression (summary memory)
# ---------------------------------------------------------------------------


async def compress_context(
    messages: list,
    model,
    *,
    keep_last: int = 10,
) -> list:
    """Compress old messages into a summary, keeping recent ones verbatim.

    When the message list exceeds ``keep_last * 2``, the older half is fed to
    the LLM to produce a short summary paragraph.  The summary replaces the old
    messages as a single ``SystemMessage``, while the most recent *keep_last*
    messages are kept intact.

    This is a lightweight alternative to full vector-based memory — it prevents
    the prompt from growing beyond the model's context window without losing
    the thread of the conversation.

    Parameters
    ----------
    messages : list
        The full message list to potentially compress.
    model : BaseChatModel
        The LLM to use for summarisation (no tools needed).
    keep_last : int
        Number of recent messages to keep verbatim.  Default 10.

    Returns:
    -------
    list
        The (possibly compressed) message list.
    """
    threshold = keep_last * 2
    if len(messages) <= threshold:
        return messages

    from langchain_core.messages import SystemMessage

    split = len(messages) - keep_last
    old = messages[:split]
    recent = messages[split:]

    # Build a compact transcript of the older messages
    transcript_lines: list[str] = []
    for m in old:
        role = type(m).__name__.replace("Message", "")
        content = get_message_text(m)
        transcript_lines.append(f"[{role}] {content}")

    summary_prompt = (
        "Summarise the following conversation excerpt into a single paragraph "
        "(max 150 words).  Focus on:\n"
        "- Key decisions made and facts learned\n"
        "- User preferences or constraints revealed\n"
        "- Any unfinished tasks or open questions\n\n"
        "Conversation:\n"
        + "\n".join(transcript_lines)
        + "\n\nSummary:"
    )

    try:
        response = await model.ainvoke([SystemMessage(content=summary_prompt)])
        summary = (
            "[Context summary of earlier conversation]\n"
            + str(response.content)
        )
    except Exception as exc:
        _logger.debug("Context compression failed, keeping original: %s", exc)
        summary = (
            f"[Context summary — {len(old)} earlier messages omitted due to length]"
        )

    return [SystemMessage(content=summary)] + recent


# ---------------------------------------------------------------------------
# Fact extraction (auto-memory)
# ---------------------------------------------------------------------------


async def extract_facts(
    messages: list,
    model,
    *,
    max_facts: int = 5,
    user_query: str = "",
) -> list[str]:
    """Ask the LLM to extract key facts from a conversation for long-term storage.

    Returns a list of self-contained fact statements (or an empty list if
    nothing worth remembering was found).

    Parameters
    ----------
    messages : list
        The conversation messages to analyse.
    model : BaseChatModel
        The LLM to use for extraction.
    max_facts : int
        Maximum number of facts to extract (default 5).
    user_query : str
        The original user query, always included so the LLM has context
        even when the conversation has grown beyond the transcript window.
    """
    from langchain_core.messages import SystemMessage

    # Build transcript from the last 20 messages.  In multi-step modes
    # (Plan-Solve / Supervisor) the original user query can be pushed out
    # of this window by tool calls and intermediate steps, so we always
    # prepend the user_query when provided.
    transcript_lines: list[str] = []
    if user_query:
        transcript_lines.append(f"[User original query] {user_query}")
        transcript_lines.append("")  # blank separator
    for m in messages[-20:]:
        role = type(m).__name__.replace("Message", "")
        content = get_message_text(m)
        transcript_lines.append(f"[{role}] {content}")

    prompt = f"""Analyse this conversation and extract up to {max_facts} key facts
that are worth remembering for future interactions.

A "fact worth remembering" is:
- A user preference or constraint ("I prefer Python", "Use short answers")
- A decision or conclusion reached ("Agreed to use Chroma for storage")
- Personal context ("I'm a master's student", "My project deadline is June 15")
- A domain-specific insight the user explicitly asked about
- The user's travel destination, budget range, or planning constraints
- Topics or domains the user has shown interest in

CRITICAL: Even for short/simple queries, always extract at least ONE fact about
what topic the user is interested in.  For example:
- User asks "How do I get to Sanya?" → extract "用户对三亚旅行感兴趣，询问交通方式"
- User asks "What's the weather in Tokyo?" → extract "用户关注东京天气，可能在计划旅行"
- User asks "Python vs JavaScript?" → extract "用户在比较编程语言，关注技术选型"

Do NOT extract:
- Greetings, small talk, or transient details
- Information the user explicitly asked to forget
- Tool call outputs or technical logs
- Generic travel guide content that the agent produced

Important: Extract facts about the USER (their preferences, plans, constraints),
not facts about the destinations discussed.  When in doubt, err on the side of
extracting — a marginally useful fact is better than a missed signal.

Return ONLY a JSON array of strings, with no additional text:
["fact 1", "fact 2"]

Conversation:
{chr(10).join(transcript_lines)}

JSON array of facts:"""

    try:
        # Early return in benchmark mode — skip the LLM call entirely
        # to avoid wasting API quota on eval/test runs.
        if _in_benchmark_mode():
            _logger.info(
                "Memory: benchmark mode active — skipping fact extraction entirely"
            )
            return []

        response = await model.ainvoke([SystemMessage(content=prompt)])
        raw = str(response.content).strip()

        # Parse the JSON array
        import json

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw.rsplit("\n```", 1)[0]

        facts: list[str] = json.loads(raw)
        if not isinstance(facts, list):
            return []

        facts = [f for f in facts if isinstance(f, str) and f.strip()]
        # Filter out facts that look like benchmark data
        facts = [f for f in facts if not _looks_like_benchmark(f)]
        _logger.info(
            "Memory: extracted %d facts from conversation (after benchmark filter)",
            len(facts),
        )
        return facts[:max_facts]

    except json.JSONDecodeError as exc:
        _logger.warning(
            "Fact extraction: JSON parse failed — LLM returned: '%s'… "
            "(error: %s)",
            raw[:200] if 'raw' in dir() else "(no response)",
            exc,
        )
        return []
    except Exception as exc:
        _logger.warning("Fact extraction failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Memory tools (registered in tools.py so every mode can use them)
# ---------------------------------------------------------------------------

# Patterns that strongly indicate a benchmark/eval query (not a real user).
# Used by `remember()` and `extract_facts()` to avoid contaminating long-term
# memory with synthetic test data.
_BENCHMARK_SIGNALS = [
    "plan a 3-day trip to tokyo for a first-time visitor",
    "what is the capital of france",
    "what is 2 + 2",
    "calculate the compound interest",
    "who won the most recent fifa world cup",
    "review this code for bugs",
    "compare python and javascript for web development",
    "research the gdp of japan",
    "a train leaves station a",
    "search for the current population of china",
    "i want to start learning machine learning",
    "what degree am i pursuing",
    "write a short analysis: is python good for ai",
    "remember this:",
]


def _looks_like_benchmark(text: str) -> bool:
    """Return True if *text* matches a known benchmark/eval pattern."""
    if not text:
        return False
    text_lower = text.strip().lower()
    for signal in _BENCHMARK_SIGNALS:
        if signal in text_lower:
            return True
    return False


def set_benchmark_mode(enabled: bool = True) -> None:
    """Set the global benchmark-mode flag for the current async context.

    Call this from eval runners before invoking the graph so that
    ``remember()`` and ``extract_facts()`` can skip storage without needing
    access to the graph's state object.

    Example::

        from react_agent.memory import set_benchmark_mode
        set_benchmark_mode(True)
        result = await graph.ainvoke(...)
    """
    _benchmark_mode.set(enabled)
    _logger.info("Memory: benchmark_mode set to %s", enabled)


def _in_benchmark_mode() -> bool:
    """Return True if the current async context is in benchmark/eval mode."""
    try:
        return _benchmark_mode.get()
    except LookupError:
        return False


async def remember(fact: str) -> str:
    """Store an important fact in long-term memory for future sessions.

    Use this when the user shares preferences, important context,
    decisions, or personal information that should persist across
    conversations.  Facts are stored with embeddings and can be recalled
    by meaning, not just by keyword match.

    Args:
        fact: A clear self-contained statement to remember.

    Returns:
        Confirmation that the fact was stored.
    """
    # Guard (dual-layer): reject facts during benchmark/eval runs OR when
    # the fact text matches known benchmark patterns.
    if _in_benchmark_mode():
        return (
            "⚠️ Benchmark mode is active. "
            "This fact was NOT stored to avoid contaminating long-term memory."
        )
    if _looks_like_benchmark(fact):
        return (
            "⚠️ This fact looks like it came from a test/eval query. "
            "It was NOT stored to avoid contaminating long-term memory."
        )

    store = await _ensure_memory_loaded()
    if store is None:
        return (
            "Memory store is not available (embedding API may be unreachable). "
            "The fact was NOT saved — I'll only remember it for this session."
        )

    fact_id = await store.store(fact)
    return (
        f"✓ Stored in long-term memory (id: {fact_id}). "
        f"I'll recall this in future conversations."
    )


async def recall(query: str) -> str:
    """Search long-term memory for relevant facts from previous conversations.

    Use this at the start of a conversation, or whenever context from
    earlier interactions would help, to retrieve stored facts about the
    user such as preferences, past decisions, and ongoing projects.

    Args:
        query: A natural-language search query describing the kind of
            information to look for.

    Returns:
        Formatted list of matching facts, or a message indicating none found.
    """
    store = await _ensure_memory_loaded()
    if store is None:
        return "Memory store is not available."

    facts = await store.recall(query, k=5)
    if not facts:
        return "No relevant memories found for that query."

    parts: list[str] = []
    for i, f in enumerate(facts, 1):
        ts = f["metadata"].get("timestamp", "unknown")[:19]  # truncate to seconds
        parts.append(f"[{i}] {f['content']}\n   ─ saved {ts}")

    return "\n\n".join(parts)


async def recall_all() -> str:
    """List all facts stored in long-term memory, most recent first.

    Use this for a complete overview of everything stored about the user,
    rather than facts matching a specific search query.

    Returns:
        Formatted list of all stored facts.
    """
    store = await _ensure_memory_loaded()
    if store is None:
        return "Memory store is not available."

    facts = await store.recall_all(k=50)
    if not facts:
        return "No facts stored in long-term memory yet."

    # Sort by timestamp descending
    facts.sort(
        key=lambda f: f["metadata"].get("timestamp", ""),
        reverse=True,
    )

    parts: list[str] = []
    for i, f in enumerate(facts, 1):
        ts = f["metadata"].get("timestamp", "unknown")[:19]
        parts.append(f"[{i}] {f['content']}\n   ─ saved {ts}")

    return "\n\n".join(parts)
