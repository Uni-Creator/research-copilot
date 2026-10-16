"""Thin LLM helpers: model factory + structured-output call that also reports token usage."""
from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from copilot.graph.state import Usage

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    pass


def get_llm(model: str, provider: str = "anthropic", *, temperature: float = 0.0, max_tokens: int = 2048) -> Any:
    """Build a LangChain chat model. Imported lazily so tests don't need provider packages."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model, model_provider=provider, temperature=temperature, max_tokens=max_tokens
    )


def usage_from_message(raw: Any) -> Usage:
    meta = getattr(raw, "usage_metadata", None) or {}
    return Usage(
        llm_calls=1,
        input_tokens=int(meta.get("input_tokens", 0)),
        output_tokens=int(meta.get("output_tokens", 0)),
    )


def structured_call(llm: Any, schema: type[T], messages: list) -> tuple[T, Usage]:
    """Invoke ``llm`` constrained to ``schema``; return (parsed, usage).

    Raises StructuredOutputError if the model output could not be parsed.
    """
    out = llm.with_structured_output(schema, include_raw=True).invoke(messages)
    usage = usage_from_message(out.get("raw"))
    parsed = out.get("parsed")
    if parsed is None:
        raise StructuredOutputError(f"could not parse {schema.__name__}: {out.get('parsing_error')!r}")
    return parsed, usage


def text_call(llm: Any, messages: list) -> tuple[str, Usage]:
    """Plain (free-text) completion; returns (text, usage)."""
    msg = llm.invoke(messages)
    content = msg.content
    if isinstance(content, list):  # content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content), usage_from_message(msg)
