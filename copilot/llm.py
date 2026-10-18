"""Thin LLM helpers: model factory + structured-output call that also reports token usage."""
from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from copilot.graph.state import Usage

T = TypeVar("T", bound=BaseModel)


import os

_LIMITER = None


def _rate_limiter():
    """One limiter shared by every model, so TOTAL requests/minute stay under the free-tier cap."""
    global _LIMITER
    rpm = float(os.environ.get("COPILOT_LLM_RPM", "0"))
    if rpm > 0 and _LIMITER is None:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        _LIMITER = InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.1, max_bucket_size=1)
    return _LIMITER


def get_llm(model: str, provider: str = "anthropic", *, temperature: float = 0.0, max_tokens: int = 2048) -> Any:
    from langchain.chat_models import init_chat_model

    kwargs: dict[str, Any] = {"temperature": temperature}
    if provider == "ollama":
        kwargs.update(num_predict=max_tokens, num_ctx=8192)   # Ollama's default context would truncate prompts
    else:
        kwargs["max_tokens"] = max_tokens
    if (limiter := _rate_limiter()) is not None:
        kwargs["rate_limiter"] = limiter
    return init_chat_model(model, model_provider=provider, **kwargs)


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
