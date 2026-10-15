from types import SimpleNamespace

from copilot.graph.state import Paper


class FakeLLM:
    """Mimics ``llm.with_structured_output(schema, include_raw=True).invoke(messages)``."""

    def __init__(self, parsed=None, error: Exception | None = None):
        self.parsed, self.error, self.calls = parsed, error, 0

    def with_structured_output(self, schema, include_raw=True):
        outer = self

        class _Runnable:
            def invoke(self, messages):
                outer.calls += 1
                if outer.error:
                    raise outer.error
                raw = SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 50})
                return {"raw": raw, "parsed": outer.parsed, "parsing_error": None}

        return _Runnable()


def mk(title, year=2023, cites=0, arxiv=None, doi=None, s2=None, abstract="", source="arxiv"):
    return Paper(
        title=title, year=year, citation_count=cites, arxiv_id=arxiv, doi=doi,
        s2_id=s2, abstract=abstract, sources=[source],
    )


def fake_source(catalog):
    """catalog: {query: [Paper, ...]} -> SearchFn that records the queries it receives."""
    calls = []

    def fn(query, filters, limit):
        calls.append(query)
        return [p.model_copy(deep=True) for p in catalog.get(query, [])][:limit]

    fn.calls = calls
    return fn
