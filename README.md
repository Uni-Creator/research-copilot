# Research Copilot (multi-agent)

Research question -> planned search -> ranked papers -> (soon) parsed notes -> cited review.

**Status: Week 1 done** - state schema, Planner, Retriever, graph wiring, CLI, offline tests.

```
START -> planner -> retriever -> END          (today)
retriever -> reader(parallel) -> writer -> critic_agent -> END | writer | planner   (Weeks 2-4)
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY (and optionally S2_API_KEY)
pytest                          # fully offline: fake LLM + fake search sources
python -m copilot.cli "gloss-free sign language production" --year-from 2021 --max-papers 15
```
First run is slow (arXiv asks for >=3 s between requests); responses are cached in `.cache/`.
Each run writes a JSON trace to `traces/`.

## Layout
| Path | Purpose |
|---|---|
| `copilot/graph/state.py` | `CopilotState`, typed contracts, reducers |
| `copilot/agents/planner.py` | question -> sub-questions + queries (structured LLM output, sanitised, fallback plan) |
| `copilot/agents/retriever.py` | parallel search, dedupe, rank fusion, shortlist, follow-up passes |
| `copilot/tools/` | arXiv, Semantic Scholar, HTTP retry/cache/rate-limit, dedupe, ranking |
| `copilot/graph/workflow.py` | graph wiring |
| `copilot/prompts/` | one prompt file per agent |
| `tests/` | offline tests |

## Key design choices
- Agents communicate only via typed state; parallel-safe fields have reducers.
- Retriever ranks with reciprocal-rank fusion + citations + recency + keyword overlap (deterministic, free).
- Loop support is already in: `searched_queries` prevents repeats, follow-up passes grow the shortlist.
- Failures degrade instead of crash: planner falls back to the raw question; a failing source is logged in `errors`.
- Token usage is tracked in `state["usage"]`; `routing.budget_exceeded` is ready for the Critic loop.

## Roadmap
- [x] W1 state, Planner, Retriever, tests
- [ ] W1 left: single-agent baseline script + eval set (5-8 topics, hand-listed must-cite papers)
- [ ] W2 PDF parsing, chunking, embeddings (bge-small), vector store, Reader agent (`Send` fan-out)
- [ ] W3 Writer + full graph
- [ ] W4 Critic (claim-to-chunk verification) + loop edges
- [ ] W5 eval + ablations, W6 FastAPI/Streamlit/Docker/deploy
