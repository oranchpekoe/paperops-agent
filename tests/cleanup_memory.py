r"""One-shot cleanup script for contaminated long-term memory.

Removes facts that were auto-extracted from benchmark/eval conversations
and mistakenly stored as user attributes.

Usage
-----
.. code-block:: bash

    # Dry-run: show what would be removed
    python tests/cleanup_memory.py

    # Actually delete the contaminated facts
    python tests/cleanup_memory.py --commit

    # Under the conda env:
    conda activate langraph
    set PYTHONIOENCODING=utf-8
    python tests/cleanup_memory.py --commit
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Ensure src/ is importable
_src = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, _src)

from dotenv import load_dotenv
load_dotenv()

# Patterns that identify benchmark-derived facts (case-insensitive match)
CONTAMINATED_PATTERNS = [
    "first-time visitor to Tokyo",
    "Tokyo planning a 3-day trip",
    "Plan a 3-day trip",
    "capital of France",
    "compound interest on $10,000",
    "FIFA World Cup",
    "GDP of Japan",
    "population of China",
    "4-week study plan",
    "dual-degree master's student",
    "I prefer Python",
    "review this code",
    "compare Python and JavaScript",
]


async def cleanup(commit: bool = False) -> None:
    """Run the cleanup."""
    # Standard import — requires conda env with langgraph deps installed
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from react_agent.memory import MemoryStore
    from langchain_openai import OpenAIEmbeddings

    persist_dir = os.environ.get(
        "CHROMA_PERSIST_DIR",
        str(Path(__file__).resolve().parent.parent / ".chroma_db"),
    )

    embeddings = OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_base=os.environ.get("OPENAI_BASE_URL"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )
    store = MemoryStore(persist_dir, embeddings)

    # Show current state
    all_facts = await store.recall_all(k=100)
    print(f"Current facts in memory: {len(all_facts)}")
    print("-" * 60)
    for i, f in enumerate(all_facts, 1):
        content = f["content"][:100]
        meta = f.get("metadata", {})
        print(f"  [{i}] {content}")
        print(f"      id={meta.get('id', '?')}  saved={meta.get('timestamp', '?')[:19]}")
        # Flag contaminated
        for pat in CONTAMINATED_PATTERNS:
            if pat.lower() in content.lower():
                print(f"      ** CONTAMINATED (matches '{pat}')")
                break
        print()

    if not commit:
        print("-" * 60)
        print("DRY RUN — no changes made. Run with --commit to actually delete.")
        return

    # Commit: remove contaminated facts
    removed = await store.clear_contaminated(CONTAMINATED_PATTERNS)
    print("-" * 60)
    print(f"Removed {removed} contaminated fact(s).")

    # Show remaining
    remaining = await store.recall_all(k=100)
    print(f"Remaining facts: {len(remaining)}")
    for i, f in enumerate(remaining, 1):
        print(f"  [{i}] {f['content'][:100]}")


async def nuke_all() -> None:
    """Clear the entire memory store — nuclear option."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from react_agent.memory import MemoryStore
    from langchain_openai import OpenAIEmbeddings

    persist_dir = os.environ.get(
        "CHROMA_PERSIST_DIR",
        str(Path(__file__).resolve().parent.parent / ".chroma_db"),
    )

    embeddings = OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_base=os.environ.get("OPENAI_BASE_URL"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )
    store = MemoryStore(persist_dir, embeddings)
    count = await store.clear_all()
    print(f"Cleared all {count} facts from long-term memory.")


if __name__ == "__main__":
    if "--all" in sys.argv:
        asyncio.run(nuke_all())
    else:
        commit_flag = "--commit" in sys.argv
        asyncio.run(cleanup(commit=commit_flag))
