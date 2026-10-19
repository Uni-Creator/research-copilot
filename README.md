# Research Copilot

A multi-agent system that turns a research question into a **cited literature review whose citations are checked against the source text**. Built with LangGraph; runs as a CLI, a Streamlit app, or an HTTP API with live event streaming.

```text
question -> Planner -> Retriever -> Reader x N (parallel) -> Writer -> Critic --pass--> report
             ^                                               ^         |
             +---------------- retrieve_more ----------------+---------+ revise_draft
```

| Agent | Job |
|---|---|
| **Planner** | question -> sub-questions + search queries (structured output, sanitised, fallback plan) |
| **Retriever** | arXiv + OpenAlex in parallel (optional Semantic Scholar), dedupe, MMR diversity ranking, shortlist, follow-up passes |
| **Reader** (x N) | PDF -> chunks -> vector store; extracts method / datasets / metrics / limitations |
| **Writer** | evidence-grounded review; invented citations are stripped, and sentences resting only on them dropped |
| **Critic** | verifies each cited claim against the cited passage; reviews coverage; routes pass / revise / retrieve more |

Details, failure behaviour and limitations: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-app.txt
pip install sentence-transformers          # real embeddings (bge-small; fine on CPU or a 6 GB GPU)
cp .env.example .env                       # configure GOOGLE_API_KEY and OPENALEX_API_KEY
make test                                  # 81 tests, fully offline (fake LLMs + fake search sources)

make ui                                    # Streamlit app, live agent trace
python -m copilot.cli "gloss-free sign language production" --year-from 2021 --out runs/slp
make api                                   # HTTP API on :8000 (see deploy/API.md)
```

Cheap modes while developing: `--until retrieve` (plan + search, one LLM call), `--no-fulltext` (abstracts only), `COPILOT_EMBEDDER=hash` (no model download; not semantic).

## Interfaces

- **UI** (`ui/app.py`): live, detailed agent trace (reasoning, plan, searches, papers chosen and why, per-paper reading, every fact-check verdict); tabs for Report, Papers, per-claim Verification, Trace, Downloads.
- **API** (`copilot/service.py`): `POST /runs`, `GET /runs/{id}/events` (SSE, resumable), reports as md/html/json. Optional `COPILOT_API_KEY`. See [deploy/API.md](deploy/API.md).
- **CLI** (`copilot/cli.py`): prints every step live (`--brief` for one line each); writes `report.md`, `report.html`, `run.json`.

## Deploy

`docker compose up --build` (UI :7860, API :8000) or follow [deploy/HUGGINGFACE.md](deploy/HUGGINGFACE.md).

**Set `COPILOT_UI_PASSWORD` / `COPILOT_API_KEY` on anything public**: every run spends LLM tokens. CI (`.github/workflows/ci.yml`) runs ruff + the offline tests on every push.

## Evaluation

```bash
python -m eval.run_eval --topics eval/datasets/slp.json --variants baseline,multi,multi_critic --out eval/results
python -m eval.run_eval ... --no-fulltext --judge-model <a different model than the Critic>
```

Variants:

- `baseline`: one search, abstracts, one write call.
- `multi`: multi-agent pipeline without the Critic.
- `multi_critic`: full multi-agent pipeline with Critic verification and revision.

### Latest results

Evaluation topic: `slp_architectures` — comparing architectures for sign language production. Results below are from the latest recorded run.

| Metric | Baseline | Multi | Multi + Critic |
|---|---:|---:|---:|
| Known-paper pool coverage | 0.0% | 100.0% | 100.0% |
| Known-paper shortlist coverage | 0.0% | 100.0% | 100.0% |
| Known-paper cited coverage | 0.0% | 100.0% | 100.0% |
| Precision@5 | 0.0% | 40.0% | 60.0% |
| Recall@5 | 0.0% | 66.67% | 100.0% |
| nDCG@5 | 0.000 | 0.384 | 0.618 |
| MRR | 0.000 | 0.250 | 0.333 |
| Average precision | 0.000 | 0.372 | 0.478 |
| Mean subquestion coverage | 73.91% | 83.46% | 83.89% |
| Claim support rate | 84.09% | 90.00% | 93.10% |
| Unsupported claims | 1 | 0 | 0 |
| Papers cited | 13 | 16 | 15 |
| Claims generated | 22 | 25 | 29 |
| Stripped citations | 0 | 0 | 0 |
| Total tokens | 4,597 | 17,635 | 31,436 |
| Runner time | 18.36 s | 229.95 s | 250.30 s |
| Evaluation time | 25.43 s | 25.87 s | 30.72 s |
| **Total runtime** | **43.8 s** | **255.8 s** | **281.0 s** |
| Errors | 0 | 0 | 0 |

*Coverage metrics are based on the benchmark's required-paper list. They measure known-item retrieval, not the overall relevance of every retrieved paper. Precision, recall, and ranking metrics depend on the relevance judgments used by the evaluation harness.*

### Interpretation

- **Baseline:** Fastest and least expensive, with an 84.09% claim support rate, but it did not retrieve any of the benchmark's required papers.
- **Multi:** Achieved full known-paper coverage, improved subquestion coverage, and a 90.00% claim support rate. It took approximately 4.3 minutes.
- **Multi + Critic:** Best reported retrieval and synthesis metrics in this run, with 100% Recall@5 and a 93.10% claim support rate. It took approximately 4.7 minutes and used the most tokens.

The Critic variant improved support rate by 3.10 percentage points over `multi`, while adding approximately 25 seconds of runtime. Its mean subquestion coverage improved by only 0.43 percentage points.

These results are preliminary: they cover one topic and should not be treated as proof of general performance. Claim support depends on the evaluator's judgments, and zero unsupported claims does not independently establish factual correctness. Validate relevance judgments, manually audit a sample of citations, and evaluate additional topics before selecting a production configuration.

### Evaluation guidelines

- The baseline reads abstracts, while multi-agent variants can read full text. Use `--no-fulltext` across variants to isolate architectural differences from access to additional evidence.
- Use a judge model different from the Critic's model, and hand-check approximately 20% of verdicts.
- `eval/datasets/slp.json` is a one-topic seed whose titles came from memory. Verify the titles and add 5–8 curated topics, including ISL-specific ones.
- Preserve both known-paper metrics and independent relevance judgments. Do not interpret zero known-paper coverage as proof that every retrieved paper is irrelevant.
- Report quality alongside cost: support rate, nDCG@5, subquestion coverage, token usage, runtime, and errors.

## Project layout

- `copilot/graph`: state, routing, workflow
- `copilot/agents`: the five agents (planner, retriever, reader, writer, critic)
- `copilot/tools`: arXiv, OpenAlex, Semantic Scholar, PDF extraction, chunking, embeddings, vector store, MMR re-ranking
- `copilot/factory.py`: shared resources and search adapters
- `copilot/service.py`: HTTP API and event streaming
- `copilot/render.py`: markdown and HTML report generation
- `copilot/baseline.py`: baseline single-pass implementation
- `ui/`: Streamlit interface with live agent trace inspection
- `eval/`: evaluation harness, ranked retrieval metrics (Precision@k, Recall@k, nDCG@k, MRR, MAP, subquestion coverage), and benchmark datasets
- `deploy/`: deployment configuration (Docker, Hugging Face, HTTP API)
- `docs/`: architecture and project documentation

## Status

- **Tested offline**: 81 automated tests passing covering pipeline graphs, loops, API, UI, MMR re-ranking, and retrieval/evaluation metrics.
- **Exercised against live services**: Validated against live Gemini LLMs (Gemini 3.5 Flash Lite), arXiv, and OpenAlex.
- **Evaluation framework**: Full ranked-retrieval metrics suite (Precision@k, Recall@k, nDCG@k, MRR, MAP, subquestion coverage, claim support rate).
- Not built (stretch): Gap Finder agent, human-in-the-loop paper approval, citation-graph expansion.