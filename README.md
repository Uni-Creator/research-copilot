# Research Copilot

A multi-agent system that turns a research question into a **cited literature review whose citations are
checked against the source text**. Built with LangGraph; runs as a CLI, a Streamlit app, or an HTTP API with
live event streaming.

```
question -> Planner -> Retriever -> Reader x N (parallel) -> Writer -> Critic --pass--> report
              ^                                               ^         |
              +---------------- retrieve_more ----------------+---------+ revise_draft
```

| Agent | Job |
|---|---|
| **Planner** | question -> sub-questions + search queries (structured output, sanitised, fallback plan) |
| **Retriever** | arXiv + Semantic Scholar in parallel, dedupe, rank fusion, shortlist, follow-up passes |
| **Reader** (x N) | PDF -> chunks -> vector store; extracts method / datasets / metrics / limitations |
| **Writer** | evidence-grounded review; invented citations are stripped, and sentences resting only on them dropped |
| **Critic** | verifies each cited claim against the cited passage; reviews coverage; routes pass / revise / retrieve more |

Details, failure behaviour and limitations: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-app.txt
pip install sentence-transformers          # real embeddings (bge-small; fine on CPU or a 6 GB GPU)
cp .env.example .env                       # set ANTHROPIC_API_KEY
make test                                  # 41 tests, fully offline (fake LLMs + fake search sources)

make ui                                    # Streamlit app, live agent trace
python -m copilot.cli "gloss-free sign language production" --year-from 2021 --out runs/slp
make api                                   # HTTP API on :8000 (see deploy/API.md)
```
Cheap modes while developing: `--until retrieve` (plan + search, one LLM call), `--no-fulltext` (abstracts
only), `COPILOT_EMBEDDER=hash` (no model download; not semantic).

## Interfaces
- **UI** (`ui/app.py`): live, detailed agent trace (reasoning, plan, searches, papers chosen and why, per-paper reading, every fact-check verdict); tabs for Report, Papers, per-claim Verification, Trace, Downloads.
- **API** (`copilot/service.py`): `POST /runs`, `GET /runs/{id}/events` (SSE, resumable), reports as md/html/json.
  Optional `COPILOT_API_KEY`. See [deploy/API.md](deploy/API.md).
- **CLI** (`copilot/cli.py`): prints every step live (`--brief` for one line each); writes `report.md`, `report.html`, `run.json`.

## Deploy
`docker compose up --build` (UI :7860, API :8000) or follow [deploy/HUGGINGFACE.md](deploy/HUGGINGFACE.md).
**Set `COPILOT_UI_PASSWORD` / `COPILOT_API_KEY` on anything public**: every run spends LLM tokens. CI
(`.github/workflows/ci.yml`) runs ruff + the offline tests on every push.

## Evaluation
```bash
python -m eval.run_eval --topics eval/datasets/slp.json --variants baseline,multi,multi_critic --out eval/results
python -m eval.run_eval ... --no-fulltext --judge-model <a different model than the Critic>
```
Variants: `baseline` (one search, abstracts, one write call) / `multi` (no Critic) / `multi_critic` (full).
Metrics: `coverage_shortlist`, `coverage_cited` (recall of your must-cite list), `support_rate`
(judge-verified; supported=1, partial=0.5), `stripped_citations`, tokens, seconds.

**Interpretation rules**
- `baseline` reads abstracts and `multi*` read full text, so compare with `--no-fulltext` to isolate the
  architecture from the extra evidence.
- Use a judge model different from the Critic's, and hand-check ~20% of verdicts.
- `eval/datasets/slp.json` is a 1-topic seed whose titles come from memory: verify them and add 5-8 topics
  you curate (including ISL-specific ones).

### Results (fill in after your runs)
| Variant | coverage_shortlist | coverage_cited | support_rate | stripped_citations | tokens | seconds |
|---|---|---|---|---|---|---|
| baseline | | | | | | |
| multi | | | | | | |
| multi_critic | | | | | | |

## Project layout
`copilot/graph` state, routing, workflow - `copilot/agents` the five agents - `copilot/tools` arXiv, Semantic
Scholar, PDF, chunking, embeddings, vector store, ranking - `copilot/factory.py` shared resources -
`copilot/service.py` API - `copilot/render.py` reports - `copilot/baseline.py` - `ui/` - `eval/` - `deploy/` - `docs/`.

## Status
Built and tested offline (41 tests): pipeline, loops, API, UI, rendering, eval harness. **Not yet exercised
against live LLM/arXiv/Semantic Scholar/real PDFs, and the Docker image has not been built** in the
environment this was developed in; expect some prompt tuning and PDF edge cases on first live runs.
Not built (stretch): Gap Finder agent, human-in-the-loop paper approval, citation-graph expansion.
