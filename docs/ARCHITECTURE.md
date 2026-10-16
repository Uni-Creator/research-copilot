# Architecture

```mermaid
flowchart LR
  Q[Question] --> P[Planner]
  P --> R[Retriever]
  R -->|Send x N, parallel| RD[Reader]
  RD --> W[Writer]
  W --> C[Critic]
  C -->|pass| OUT[Report]
  C -->|revise_draft| W
  C -->|retrieve_more| P
  R -.-> S1[(arXiv)]
  R -.-> S2[(Semantic Scholar)]
  RD -.-> V[(Vector store)]
  W -.-> V
  C -.-> V
```

## Contracts
All agents communicate through one typed `CopilotState` (`copilot/graph/state.py`). Parallel-safe fields have
reducers (`papers` merges duplicates, `notes` merges per-paper results, `usage`/`trace`/`errors` accumulate).
Node names never equal state keys (LangGraph restriction), hence `critique` (key) vs `critic_agent` (node).

## Grounding chain (why citations can be trusted, and where they cannot)
1. **Reader** stores every passage as a `Chunk` with a stable id `<paper_id>#<n>` (or `#abs`).
2. **Writer** sees only labelled evidence (`[C1]...`) retrieved per sub-question. Aliases are mapped back to real
   chunk ids; an alias that does not exist is stripped, and a sentence resting *only* on invented aliases is
   dropped (`Draft.stripped_citations` records how often this happens, which is itself an eval metric).
3. **Critic** treats each cited sentence as a claim and judges it against only the chunks it cites.
4. **Report** shows `[n]` numbers per paper; the table of methods/datasets/metrics comes from Reader notes,
   not from the Writer, so it cannot be hallucinated by the prose step.

Not covered: uncited sentences are not verified (the Writer is instructed to cite every factual sentence, but
nothing enforces it), and the judge is an LLM, so a verdict is evidence, not proof. Spot-check it.

## Loop control
`critic -> planner -> retriever` (retrieve_more) and `critic -> writer` (revise_draft) are bounded by
`max_iterations` (default 2) and `max_total_tokens` (default 400k). When the budget is exhausted the Critic
returns `action=pass, passed=False`: the run finishes with an honest "quality bar not met" flag rather than looping.
A follow-up retrieval only runs queries not yet searched, only reads papers without notes, and appends to the
shortlist instead of reshuffling it.

## Failure behaviour
| Failure | Behaviour |
|---|---|
| Planner LLM error / unparseable output | fallback plan = search the raw question; logged in `errors` |
| One search source down or rate-limited | retried with backoff; if still failing, logged, other source continues |
| PDF missing / unparseable | Reader falls back to the abstract for that paper |
| Reader extraction fails | notes contain the abstract's first sentences only; logged |
| Judge call fails | affected claims marked unsupported ("judge call failed"), never silently passed |
| No papers found | run ends with an explicit error, no draft |
| API run fails | status `error` with the exception; SSE stream ends with a `done` event |

## Cost / latency levers
Cheap model for Planner and Reader (many calls), stronger model for Writer and Critic. Reader fan-out is
parallel, so wall-clock is dominated by the arXiv rate limit (3 s/request, shared across runs) on a cold
cache. Everything fetched is cached on disk; embeddings of already-read papers are reused through the shared
vector store. `--until retrieve` costs one LLM call.

## Known limitations
- Sources are arXiv + Semantic Scholar only. Venues outside them (parts of the ISL / regional literature) will
  be under-represented; add a source behind the `SearchFn` interface in `copilot/tools/sources.py`.
- PDF text extraction is plain text: tables, equations and figures are lost; first 15 pages only.
- Ranking is a deterministic heuristic (rank fusion + citations + recency + keyword overlap), not learned.
- Token usage is tracked; dollar cost is not (prices change).
- The vector store lives in memory (`save`/`load` exist but are not wired to the services); the API keeps the 50
  most recent runs in memory (all runs are also written to disk).
