# File Index & Symbol Directory

This index catalogs key source files across `research-copilot`, detailing their responsibilities, public interfaces, entry points, and associated tests.

---

## 1. Core Framework & Orchestration (`copilot/`)

| File | Responsibilities | Key Classes & Functions | Associated Tests |
|---|---|---|---|
| [copilot/config.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/config.py) | Environment settings, budget caps, model selections, cache TTLs | `Settings`, `Settings.from_env` | [test_agents.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_agents.py), [test_eval.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_eval.py), [test_full_graph.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_full_graph.py), [test_graph.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_graph.py) |
| [copilot/factory.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/factory.py) | Long-lived pipeline factory sharing rate limiters, vector store, and embedder | `CopilotFactory`, `__call__` | [test_factory_ui.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_factory_ui.py), [test_service.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_service.py) |
| [copilot/llm.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/llm.py) | Multi-provider LLM adapter (Google GenAI / Anthropic) with token usage tracking | `get_llm`, `UsageTrackingCallback` | Tested via agent tests |
| [copilot/progress.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/progress.py) | Formatting `TraceEvent` for CLI streaming and Streamlit UI display | `format_event_text`, `format_event_md` | [test_progress.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_progress.py) |
| [copilot/render.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/render.py) | Markdown and HTML report generation; persistence of run artifacts | `render_markdown`, `render_html`, `save_run`, `dumps_state` | [test_full_graph.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_full_graph.py) |
| [copilot/utils.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/utils.py) | Text normalization and deterministic hashing utilities | `normalize_title`, `short_hash` | Tested via state and ranking tests |
| [copilot/baseline.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/baseline.py) | Single-pass baseline without planner or critic (control group for evaluations) | `run_baseline` | [test_eval.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_eval.py) |

---

## 2. Graph State & Routing (`copilot/graph/`)

| File | Responsibilities | Key Classes & Functions | Associated Tests |
|---|---|---|---|
| [copilot/graph/state.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/graph/state.py) | Typed contracts, parallel merge reducers, Pydantic domain models | `CopilotState`, `Paper`, `SearchFilters`, `ResearchPlan`, `PaperNotes`, `Chunk`, `Draft`, `CriticReport`, `Usage`, `TraceEvent`, `merge_papers`, `merge_dicts`, `merge_usage`, `make_initial_state` | [test_state.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_state.py), [test_dataset.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_dataset.py), [test_full_graph.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_full_graph.py) |
| [copilot/graph/routing.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/graph/routing.py) | Conditional edge logic: parallel Reader fan-out and Critic looping | `budget_exceeded`, `iterations_left`, `route_after_retriever`, `route_after_critic` | [test_graph.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_graph.py) |
| [copilot/graph/workflow.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/graph/workflow.py) | Compiles LangGraph StateGraph, manages ablation truncations (`until`) | `Copilot`, `build_copilot`, `build_graph`, `default_fetcher` | [test_graph.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_graph.py), [test_full_graph.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_full_graph.py) |

---

## 3. Agent Implementations (`copilot/agents/`)

| File | Responsibilities | Key Classes & Functions | Associated Tests |
|---|---|---|---|
| [copilot/agents/planner.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/agents/planner.py) | Query deconstruction into sub-questions and academic search terms | `make_planner_node`, `fallback_plan`, `to_plan`, `PlannedSubQuestion`, `PlannerOutput` | [test_planner.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_planner.py), [test_full_graph.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_full_graph.py) |
| [copilot/agents/retriever.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/agents/retriever.py) | Search execution, candidate deduplication, MMR re-ranking | `make_retriever_node` | [test_full_graph.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_full_graph.py), [test_retrieval_metrics.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_retrieval_metrics.py) |
| [copilot/agents/reader.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/agents/reader.py) | PDF fetching, text chunking, vector indexing, structured note extraction | `make_reader_node`, `select_context`, `NotesOut` | [test_agents.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_agents.py), [test_reading.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_reading.py) |
| [copilot/agents/writer.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/agents/writer.py) | Evidence retrieval, alias assignment, citation resolution and hallucination filtering | `make_writer_node`, `gather_evidence`, `resolve_citations`, `generate_draft`, `EvidenceItem` | [test_agents.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_agents.py) |
| [copilot/agents/critic.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/agents/critic.py) | Strict citation-to-passage claim verification and sub-question coverage evaluation | `make_critic_node`, `split_claims`, `verify_claims`, `support_rate`, `VerdictItem`, `CriticReport` | [test_agents.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_agents.py), [test_eval.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_eval.py) |

---

## 4. Tools & Retrieval Utilities (`copilot/tools/`)

| File | Responsibilities | Key Classes & Functions | Associated Tests |
|---|---|---|---|
| [copilot/tools/arxiv.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/arxiv.py) | arXiv Atom XML API client with query expansion and date filtering | `search_arxiv`, `build_arxiv_query`, `parse_arxiv_feed` | [test_tools.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_tools.py) |
| [copilot/tools/openalex.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/openalex.py) | OpenAlex REST API client, query sanitization, inverted abstract reconstruction | `search_openalex`, `clean_search_query`, `reconstruct_abstract`, `parse_openalex_work` | [test_openalex.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_openalex.py) |
| [copilot/tools/semantic_scholar.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/semantic_scholar.py) | Semantic Scholar academic graph API client with rate limit handling | `search_semantic_scholar`, `parse_s2_item` | [test_tools.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_tools.py) |
| [copilot/tools/sources.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/sources.py) | Search source factory routing queries to arXiv, OpenAlex, or Semantic Scholar | `build_sources` | [test_tools.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_tools.py) |
| [copilot/tools/ranking.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/ranking.py) | MMR re-ranking combining citation signals, query overlap, and Jaccard diversity | `rank_papers`, `RankWeights`, `MMRWeights` | [test_retrieval_metrics.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_retrieval_metrics.py) |
| [copilot/tools/pdf.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/pdf.py) | PyMuPDF text extractor with pagination bounds, header/footer stripping | `PdfFetcher`, `parse_pdf_bytes`, `clean_text` | [test_reading.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_reading.py) |
| [copilot/tools/chunking.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/chunking.py) | Text chunking with sliding character windows and stable chunk IDs | `chunk_pages`, `abstract_chunk_id`, `paper_id_of` | [test_reading.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_reading.py) |
| [copilot/tools/embeddings.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/embeddings.py) | BGE-small SentenceTransformer embedder with HashingEmbedder test fallback | `get_embedder`, `Embedder`, `HashingEmbedder`, `SentenceTransformerEmbedder` | [test_reading.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_reading.py) |
| [copilot/tools/vectorstore.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/vectorstore.py) | In-memory cosine similarity vector store indexing chunks | `VectorStore` | [test_reading.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_reading.py), [test_agents.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_agents.py) |
| [copilot/tools/http.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/http.py) | HTTP client wrapper with disk caching, token-bucket rate limiter, and backoff | `DiskCache`, `RateLimiter`, `get_with_retry` | [test_tools.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_tools.py) |
| [copilot/tools/papers.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/papers.py) | Index supporting duplicate paper lookups by DOI, arXiv ID, or title hash | `PaperIndex` | [test_state.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_state.py) |

---

## 5. Applications & Runtime Entry Points

| File | Responsibilities | Key Symbols / Commands | Associated Tests |
|---|---|---|---|
| [copilot/cli.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/cli.py) | Command-line interface with real-time trace streaming and output serialization | `main`, `python -m copilot.cli` | Tested via CLI execution |
| [copilot/service.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/service.py) | FastAPI service with background job queue, SSE event stream, and API keys | `create_app`, `RunManager`, `RunRequest` | [test_service.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_service.py) |
| [ui/app.py](file:///home/prime/Desktop/Projects/research-copilot/ui/app.py) | Streamlit web application with password gating, live traces, and tabbed inspection | `streamlit run ui/app.py`, `run_pipeline`, `show_result` | [test_factory_ui.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_factory_ui.py) |
| [scripts/start.sh](file:///home/prime/Desktop/Projects/research-copilot/scripts/start.sh) | Container entrypoint switching between UI (`APP=ui`) and API (`APP=api`) | `./scripts/start.sh` | Tested via container deployment |
| [scripts/codebase_map.py](file:///home/prime/Desktop/Projects/research-copilot/scripts/codebase_map.py) | AST dependency analyzer, cycle detector, and git blast-radius calculator | `main`, `CodebaseAnalyzer`, `get_git_status` | Direct execution / `make map` |

---

## 6. Evaluation Benchmark (`eval/`)

| File | Responsibilities | Key Classes & Functions | Associated Tests |
|---|---|---|---|
| [eval/metrics.py](file:///home/prime/Desktop/Projects/research-copilot/eval/metrics.py) | IR and verification metrics: Coverage, P@k, R@k, nDCG@k, MRR, MAP, Support Rate | `evaluate_run`, `match_paper`, `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `mrr`, `subquestion_coverage` | [test_eval.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_eval.py), [test_retrieval_metrics.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_retrieval_metrics.py) |
| [eval/run_eval.py](file:///home/prime/Desktop/Projects/research-copilot/eval/run_eval.py) | Benchmark execution harness running ablations across evaluation topics | `main`, `evaluate`, `make_variants`, `results_table` | [test_eval.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_eval.py) |
| [eval/verify_dataset.py](file:///home/prime/Desktop/Projects/research-copilot/eval/verify_dataset.py) | Validates benchmark datasets against live academic APIs | `verify_dataset`, `resolve`, `format_report` | [test_verify_dataset.py](file:///home/prime/Desktop/Projects/research-copilot/tests/test_verify_dataset.py) |
