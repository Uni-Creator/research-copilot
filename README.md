# Research Copilot (multi-agent)

Research question -> planned search -> ranked papers -> parsed full text -> cited review -> verified.

```
START -> planner -> retriever --(parallel Send)--> reader -> writer -> critic_agent -> END
            ^                                         ^          |
            |                                         |          +-- revise_draft
            +------------- retrieve_more -------------+-----------+
```

| Agent | What it does |
|---|---|
| Planner | question -> sub-questions + search queries (structured output, sanitised, fallback plan) |
| Retriever | arXiv + Semantic Scholar in parallel, dedupe, rank fusion, shortlist, follow-up passes |
| Reader (x N, parallel) | PDF -> chunks -> vector store; LLM extracts structured notes (method/datasets/metrics/limits) |
| Writer | evidence-grounded review; citations are short aliases mapped to real chunk ids; invented ones are stripped, and sentences resting only on them are dropped |
| Critic | every cited sentence is a claim; judge checks it against the cited chunk text; coverage review; routes pass / revise / retrieve more (capped loops + token budget) |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers        # real embeddings (bge-small; fine on CPU or a 6 GB GPU)
cp .env.example .env                      # add ANTHROPIC_API_KEY
pytest                                    # 33 tests, fully offline

# full run -> report.md, report.html, run.json
python -m copilot.cli "gloss-free sign language production" --year-from 2021 --out runs/slp

# cheap/fast variants while developing
python -m copilot.cli "..." --until retrieve                 # plan + search only (no PDFs, 1 LLM call)
python -m copilot.cli "..." --no-fulltext --embedder hash    # abstracts only, no model download
```
arXiv asks for >= 3 s between requests, so the first run is slow; PDFs, text and search results are
cached in `.cache/`.

## Evaluation
```bash
python -m eval.run_eval --topics eval/datasets/slp.json --variants baseline,multi,multi_critic --out eval/results
python -m eval.run_eval ... --no-fulltext     # abstracts only for every variant (isolates architecture)
```
| Variant | What it is |
|---|---|
| `baseline` | one search with the raw question, abstracts, one write call |
| `multi` | planner + retriever + readers + writer (no Critic) |
| `multi_critic` | full system |

Metrics: `coverage_shortlist` / `coverage_cited` (recall of your must-cite list), `support_rate`
(judge-verified share of cited claims backed by the cited text; supported=1, partial=0.5),
`stripped_citations`, tokens, seconds.

**Read the numbers carefully**
- `baseline` sees abstracts; `multi*` read full text by default. A gap between them mixes
  "better architecture" with "more evidence". Run with `--no-fulltext` for the controlled comparison.
- `support_rate` is judged by an LLM. Use `--judge-model` with a *different* model than the Critic
  (otherwise the Critic is grading its own homework) and hand-check ~20% of verdicts.
- `eval/datasets/slp.json` is a 1-topic **seed**. Its must-cite titles are from memory: verify them,
  then add 5-8 topics (including ISL-specific ones) with lists you curate.

## Layout
`copilot/graph/` state, routing, workflow - `copilot/agents/` the five agents - `copilot/tools/`
arXiv, Semantic Scholar, PDF, chunking, embeddings, vector store, ranking - `copilot/render.py` report
output - `copilot/baseline.py` - `eval/` metrics + runner + dataset - `copilot/prompts/` one file per agent.

## Roadmap
- [x] W1 state, Planner, Retriever
- [x] W2 PDF/chunk/embed/store, Reader with parallel fan-out
- [x] W3 Writer, full graph, rendering
- [x] W4 Critic + loops
- [x] W5 baseline + eval harness (needs your dataset + a live run)
- [ ] W6 FastAPI (SSE traces) + Streamlit UI, Docker, deploy
- [ ] Stretch: Gap Finder agent, human-in-the-loop paper approval, citation-graph expansion
