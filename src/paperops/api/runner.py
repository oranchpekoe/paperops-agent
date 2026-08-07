"""In-process execution registry around persistent LangGraph threads."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph


class JobAlreadyRunningError(RuntimeError):
    """Reject concurrent mutation of the same workflow thread."""


class JobRunner:
    """Schedule graph calls while serialising work per LangGraph thread."""

    def __init__(self, graph: CompiledStateGraph[Any, Any, Any, Any]) -> None:
        """Bind the runner to one application-scoped compiled graph."""
        self.graph = graph
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._known_threads: set[str] = set()
        self._runtime_errors: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def schedule(self, thread_id: str, graph_input: Any) -> None:
        """Start one graph invocation unless that thread is already active."""
        async with self._lock:
            existing = self._tasks.get(thread_id)
            if existing is not None and not existing.done():
                raise JobAlreadyRunningError(
                    f"PaperOps thread is already running: {thread_id}"
                )
            self._known_threads.add(thread_id)
            self._runtime_errors.pop(thread_id, None)
            task = asyncio.create_task(
                self.graph.ainvoke(graph_input, self.config(thread_id)),
                name=f"paperops-{thread_id}",
            )
            self._tasks[thread_id] = task

            def finish(completed: asyncio.Task[Any]) -> None:
                self._finish(thread_id, completed)

            task.add_done_callback(finish)

    def _finish(self, thread_id: str, task: asyncio.Task[Any]) -> None:
        """Consume unexpected task failures and remove the active marker."""
        if self._tasks.get(thread_id) is task:
            self._tasks.pop(thread_id, None)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            self._runtime_errors[thread_id] = f"{type(exception).__name__}: {exception}"

    def is_running(self, thread_id: str) -> bool:
        """Return whether this process is actively mutating a thread."""
        task = self._tasks.get(thread_id)
        return task is not None and not task.done()

    def is_known(self, thread_id: str) -> bool:
        """Return whether the thread was accepted during this process lifetime."""
        return thread_id in self._known_threads

    def runtime_error(self, thread_id: str) -> str | None:
        """Return an unexpected graph failure not represented in domain state."""
        return self._runtime_errors.get(thread_id)

    @property
    def active_count(self) -> int:
        """Return the number of active graph invocations."""
        return sum(not task.done() for task in self._tasks.values())

    async def shutdown(self) -> None:
        """Cancel local executions while leaving their checkpoints recoverable."""
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    @staticmethod
    def config(thread_id: str) -> RunnableConfig:
        """Build the only supported checkpoint configuration shape."""
        return {"configurable": {"thread_id": thread_id}}
