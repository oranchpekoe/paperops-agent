"""Utility & helper functions."""

from __future__ import annotations

import os
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime

from react_agent import prompts
from react_agent.context import Context


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name: String in the format 'provider/model'
            (e.g. ``openai/gpt-4o-mini``, ``anthropic/claude-sonnet-4-5``).
    """
    if "/" not in fully_specified_name:
        raise ValueError(
            f"Invalid model name: '{fully_specified_name}'. "
            f"Expected format: 'provider/model' (e.g. 'openai/gpt-4o-mini'). "
            f"Did you forget the provider prefix? For OpenAI-compatible APIs "
            f"(like aihubmix), use 'openai/{fully_specified_name}'."
        )
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)


def resolve_model(runtime: Optional[Runtime[Context]] = None) -> str:
    """Resolve the model name from the best available source.

    Priority:
    1. ``runtime.context.model`` — LangGraph Server injects this per-request.
    2. ``MODEL`` environment variable — for local / direct invocation.
    3. Hard-coded fallback.
    """
    if runtime is not None and runtime.context is not None:
        return runtime.context.model
    return os.environ.get("MODEL", "openai/gpt-4o-mini")


def resolve_system_prompt(runtime: Optional[Runtime[Context]] = None) -> str:
    """Resolve the system prompt, with the same priority as ``resolve_model``."""
    if runtime is not None and runtime.context is not None:
        return runtime.context.system_prompt
    return os.environ.get("SYSTEM_PROMPT", prompts.SYSTEM_PROMPT)
