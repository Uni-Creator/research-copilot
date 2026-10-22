# Module Dependency Graph & Coupling Intelligence

This document details the module-level dependency structure, coupling metrics, reverse dependencies, and architectural layer boundaries of `research-copilot`.

## Architecture Layers & Dependency Direction

The codebase adheres to a unidirectional dependency hierarchy:

```
[Entry Points & Frontends] (ui/app.py, copilot/service.py, copilot/cli.py, eval/run_eval.py)
           │
           ▼
[Factory & Orchestration] (copilot/factory.py, copilot/graph/workflow.py, copilot/graph/routing.py)
           │
           ▼
[Agent Implementations]   (copilot/agents/{planner, retriever, reader, writer, critic}.py)
           │
           ▼
[Domain Tools & Adapters] (copilot/tools/{arxiv, openalex, pdf, chunking, embeddings, vectorstore, ranking}.py)
           │
           ▼
[Foundation & State]      (copilot/graph/state.py, copilot/config.py, copilot/utils.py)
```

**Circular Dependency Analysis**:
- Cycle Detection: **0 circular imports detected** (verified via AST analysis in `scripts/codebase_map.py`).
- Topology: Strict Directed Acyclic Graph (DAG).

---

## Complete Module Dependency Diagram

```mermaid
graph TD
    subgraph copilot["copilot"]
        copilot["copilot"]
    end
    subgraph copilot_agents["copilot.agents"]
        copilot_agents["agents"]
        copilot_agents_critic["critic"]
        copilot_agents_planner["planner"]
        copilot_agents_reader["reader"]
        copilot_agents_retriever["retriever"]
        copilot_agents_writer["writer"]
    end
    subgraph copilot_core["copilot.core"]
        copilot_baseline["baseline"]
        copilot_cli["cli"]
        copilot_config["config"]
        copilot_factory["factory"]
        copilot_llm["llm"]
        copilot_progress["progress"]
        copilot_render["render"]
        copilot_service["service"]
        copilot_utils["utils"]
    end
    subgraph copilot_graph["copilot.graph"]
        copilot_graph["graph"]
        copilot_graph_routing["routing"]
        copilot_graph_state["state"]
        copilot_graph_workflow["workflow"]
    end
    subgraph copilot_tools["copilot.tools"]
        copilot_tools["tools"]
        copilot_tools_arxiv["arxiv"]
        copilot_tools_chunking["chunking"]
        copilot_tools_embeddings["embeddings"]
        copilot_tools_http["http"]
        copilot_tools_openalex["openalex"]
        copilot_tools_papers["papers"]
        copilot_tools_pdf["pdf"]
        copilot_tools_ranking["ranking"]
        copilot_tools_semantic_scholar["semantic_scholar"]
        copilot_tools_sources["sources"]
        copilot_tools_vectorstore["vectorstore"]
    end
    subgraph eval["eval"]
        eval["eval"]
        eval_metrics["metrics"]
        eval_run_eval["run_eval"]
        eval_verify_dataset["verify_dataset"]
    end
    subgraph ui["ui"]
        ui["ui"]
        ui_app["app"]
    end
    copilot_agents_critic --> copilot_config
    copilot_agents_critic --> copilot_graph_routing
    copilot_agents_critic --> copilot_graph_state
    copilot_agents_critic --> copilot_llm
    copilot_agents_critic --> copilot_progress
    copilot_agents_critic --> copilot_tools_chunking
    copilot_agents_planner --> copilot_graph_state
    copilot_agents_planner --> copilot_llm
    copilot_agents_planner --> copilot_progress
    copilot_agents_reader --> copilot_config
    copilot_agents_reader --> copilot_graph_state
    copilot_agents_reader --> copilot_llm
    copilot_agents_reader --> copilot_progress
    copilot_agents_reader --> copilot_tools_chunking
    copilot_agents_reader --> copilot_tools_vectorstore
    copilot_agents_retriever --> copilot_config
    copilot_agents_retriever --> copilot_graph_state
    copilot_agents_retriever --> copilot_progress
    copilot_agents_retriever --> copilot_tools_papers
    copilot_agents_retriever --> copilot_tools_ranking
    copilot_agents_retriever --> copilot_tools_sources
    copilot_agents_writer --> copilot_config
    copilot_agents_writer --> copilot_graph_state
    copilot_agents_writer --> copilot_llm
    copilot_agents_writer --> copilot_progress
    copilot_agents_writer --> copilot_tools_chunking
    copilot_agents_writer --> copilot_tools_vectorstore
    copilot_baseline --> copilot_agents_retriever
    copilot_baseline --> copilot_agents_writer
    copilot_baseline --> copilot_config
    copilot_baseline --> copilot_graph_state
    copilot_baseline --> copilot_tools_chunking
    copilot_baseline --> copilot_tools_sources
    copilot_cli --> copilot_config
    copilot_cli --> copilot_graph_state
    copilot_cli --> copilot_graph_workflow
    copilot_cli --> copilot_progress
    copilot_cli --> copilot_render
    copilot_factory --> copilot_config
    copilot_factory --> copilot_graph_workflow
    copilot_factory --> copilot_tools_embeddings
    copilot_factory --> copilot_tools_sources
    copilot_factory --> copilot_tools_vectorstore
    copilot_graph_routing --> copilot_config
    copilot_graph_routing --> copilot_graph_state
    copilot_graph_state --> copilot_utils
    copilot_graph_workflow --> copilot_agents_critic
    copilot_graph_workflow --> copilot_agents_planner
    copilot_graph_workflow --> copilot_agents_reader
    copilot_graph_workflow --> copilot_agents_retriever
    copilot_graph_workflow --> copilot_agents_writer
    copilot_graph_workflow --> copilot_config
    copilot_graph_workflow --> copilot_graph_routing
    copilot_graph_workflow --> copilot_graph_state
    copilot_graph_workflow --> copilot_llm
    copilot_graph_workflow --> copilot_tools_embeddings
    copilot_graph_workflow --> copilot_tools_http
    copilot_graph_workflow --> copilot_tools_pdf
    copilot_graph_workflow --> copilot_tools_sources
    copilot_graph_workflow --> copilot_tools_vectorstore
    copilot_llm --> copilot_graph_state
    copilot_progress --> copilot_graph_state
    copilot_render --> copilot_graph_state
    copilot_render --> copilot_tools_chunking
    copilot_service --> copilot_config
    copilot_service --> copilot_factory
    copilot_service --> copilot_graph_state
    copilot_service --> copilot_render
    copilot_tools_arxiv --> copilot_graph_state
    copilot_tools_arxiv --> copilot_tools_http
    copilot_tools_arxiv --> copilot_utils
    copilot_tools_chunking --> copilot_graph_state
    copilot_tools_embeddings --> copilot_utils
    copilot_tools_openalex --> copilot_graph_state
    copilot_tools_openalex --> copilot_tools_http
    copilot_tools_openalex --> copilot_utils
    copilot_tools_papers --> copilot_graph_state
    copilot_tools_pdf --> copilot_graph_state
    copilot_tools_pdf --> copilot_tools_http
    copilot_tools_ranking --> copilot_graph_state
    copilot_tools_ranking --> copilot_utils
    copilot_tools_semantic_scholar --> copilot_graph_state
    copilot_tools_semantic_scholar --> copilot_tools_http
    copilot_tools_sources --> copilot_config
    copilot_tools_sources --> copilot_graph_state
    copilot_tools_sources --> copilot_tools_arxiv
    copilot_tools_sources --> copilot_tools_http
    copilot_tools_sources --> copilot_tools_openalex
    copilot_tools_vectorstore --> copilot_graph_state
    copilot_tools_vectorstore --> copilot_tools_embeddings
    eval_metrics --> copilot_agents_critic
    eval_metrics --> copilot_graph_state
    eval_metrics --> copilot_tools_chunking
    eval_metrics --> copilot_utils
    eval_run_eval --> copilot_baseline
    eval_run_eval --> copilot_config
    eval_run_eval --> copilot_graph_state
    eval_run_eval --> copilot_graph_workflow
    eval_run_eval --> copilot_llm
    eval_run_eval --> copilot_tools_sources
    eval_run_eval --> eval_metrics
    eval_verify_dataset --> copilot_config
    eval_verify_dataset --> copilot_graph_state
    eval_verify_dataset --> copilot_tools_sources
    eval_verify_dataset --> eval_metrics
    ui_app --> copilot_config
    ui_app --> copilot_factory
    ui_app --> copilot_graph_state
    ui_app --> copilot_progress
    ui_app --> copilot_render
```

---

## Reverse Dependencies & Coupling Metrics

The table below shows the coupling metrics for key internal modules:
- **Fan-In (Incoming / Reverse Dependencies)**: Number of modules importing this module. High fan-in indicates an architectural foundation or common contract; changes here have a wide blast radius.
- **Fan-Out (Outgoing Dependencies)**: Number of internal modules imported by this module.

| Module | Fan-In (Reverse Dependencies) | Fan-Out | Reverse Dependencies (Importers) |
|---|---|---|---|
| `copilot.graph.state` | 39 | 1 | copilot.agents.critic, copilot.agents.planner, copilot.agents.reader, copilot.agents.retriever, copilot.agents.writer, copilot.baseline, copilot.cli, copilot.graph.routing, copilot.graph.workflow, copilot.llm, copilot.progress, copilot.render, copilot.service, copilot.tools.arxiv, copilot.tools.chunking, copilot.tools.openalex, copilot.tools.papers, copilot.tools.pdf, copilot.tools.ranking, copilot.tools.semantic_scholar, copilot.tools.sources, copilot.tools.vectorstore, eval.metrics, eval.run_eval, eval.verify_dataset, tests.conftest, tests.test_agents, tests.test_dataset, tests.test_eval, tests.test_full_graph, tests.test_graph, tests.test_openalex, tests.test_planner, tests.test_progress, tests.test_reading, tests.test_retrieval_metrics, tests.test_state, tests.test_tools, ui.app |
| `copilot.config` | 18 | 0 | copilot.agents.critic, copilot.agents.reader, copilot.agents.retriever, copilot.agents.writer, copilot.baseline, copilot.cli, copilot.factory, copilot.graph.routing, copilot.graph.workflow, copilot.service, copilot.tools.sources, eval.run_eval, eval.verify_dataset, tests.test_agents, tests.test_eval, tests.test_full_graph, tests.test_graph, ui.app |
| `copilot.progress` | 8 | 1 | copilot.agents.critic, copilot.agents.planner, copilot.agents.reader, copilot.agents.retriever, copilot.agents.writer, copilot.cli, tests.test_progress, ui.app |
| `copilot.tools.chunking` | 7 | 1 | copilot.agents.critic, copilot.agents.reader, copilot.agents.writer, copilot.baseline, copilot.render, eval.metrics, tests.test_reading |
| `copilot.utils` | 6 | 0 | copilot.graph.state, copilot.tools.arxiv, copilot.tools.embeddings, copilot.tools.openalex, copilot.tools.ranking, eval.metrics |
| `copilot.tools.vectorstore` | 6 | 2 | copilot.agents.reader, copilot.agents.writer, copilot.factory, copilot.graph.workflow, tests.test_agents, tests.test_reading |
| `copilot.tools.sources` | 6 | 5 | copilot.agents.retriever, copilot.baseline, copilot.factory, copilot.graph.workflow, eval.run_eval, eval.verify_dataset |
| `copilot.tools.http` | 6 | 0 | copilot.graph.workflow, copilot.tools.arxiv, copilot.tools.openalex, copilot.tools.pdf, copilot.tools.semantic_scholar, copilot.tools.sources |
| `copilot.tools.embeddings` | 6 | 1 | copilot.factory, copilot.graph.workflow, copilot.tools.vectorstore, tests.test_agents, tests.test_full_graph, tests.test_reading |
| `copilot.llm` | 6 | 1 | copilot.agents.critic, copilot.agents.planner, copilot.agents.reader, copilot.agents.writer, copilot.graph.workflow, eval.run_eval |
| `eval.metrics` | 5 | 4 | eval.run_eval, eval.verify_dataset, tests.test_dataset, tests.test_eval, tests.test_retrieval_metrics |
| `copilot.graph.workflow` | 5 | 14 | copilot.cli, copilot.factory, eval.run_eval, tests.test_full_graph, tests.test_graph |
| `copilot.agents.critic` | 5 | 6 | copilot.graph.workflow, eval.metrics, tests.test_agents, tests.test_eval, tests.test_full_graph |
| `copilot.render` | 4 | 2 | copilot.cli, copilot.service, tests.test_full_graph, ui.app |
| `copilot.factory` | 4 | 5 | copilot.service, tests.test_factory_ui, tests.test_service, ui.app |
| `copilot.agents.planner` | 4 | 3 | copilot.graph.workflow, tests.test_full_graph, tests.test_graph, tests.test_planner |
| `copilot.tools.ranking` | 3 | 2 | copilot.agents.retriever, tests.test_retrieval_metrics, tests.test_tools |
| `copilot.tools.arxiv` | 3 | 3 | copilot.tools.sources, tests.test_retrieval_metrics, tests.test_tools |
| `copilot.agents.writer` | 3 | 6 | copilot.baseline, copilot.graph.workflow, tests.test_agents |
| `copilot.agents.reader` | 3 | 6 | copilot.graph.workflow, tests.test_agents, tests.test_full_graph |
| `copilot.tools.pdf` | 2 | 2 | copilot.graph.workflow, tests.test_reading |
| `copilot.tools.papers` | 2 | 1 | copilot.agents.retriever, tests.test_state |
| `copilot.tools.openalex` | 2 | 3 | copilot.tools.sources, tests.test_openalex |
| `copilot.graph.routing` | 2 | 2 | copilot.agents.critic, copilot.graph.workflow |
| `copilot.baseline` | 2 | 6 | eval.run_eval, tests.test_eval |
| `copilot.agents.retriever` | 2 | 6 | copilot.baseline, copilot.graph.workflow |
| `eval.verify_dataset` | 1 | 4 | tests.test_verify_dataset |
| `eval.run_eval` | 1 | 7 | tests.test_eval |
| `copilot.tools.semantic_scholar` | 1 | 2 | tests.test_tools |
| `copilot.service` | 1 | 4 | tests.test_service |
| `ui.app` | 0 | 5 | (None - runtime entry point) |
| `ui` | 0 | 0 | (None - runtime entry point) |
| `eval` | 0 | 0 | (None - runtime entry point) |
| `copilot.tools` | 0 | 0 | (None - runtime entry point) |
| `copilot.graph` | 0 | 0 | (None - runtime entry point) |
| `copilot.cli` | 0 | 5 | (None - runtime entry point) |
| `copilot.agents` | 0 | 0 | (None - runtime entry point) |
| `copilot` | 0 | 0 | (None - runtime entry point) |
---

## Dynamic & Runtime Dependencies

Some dependencies cannot be fully identified from static top-level imports alone:

1. **LLM Provider Backends** (`copilot/llm.py`):
   - `langchain_google_genai.ChatGoogleGenerativeAI` is dynamically resolved at runtime when `model_provider="google_genai"`.
   - `langchain_anthropic.ChatAnthropic` is loaded when `model_provider="anthropic"`.
   - `COPILOT_LLM_RPM` dynamically attaches an in-memory token-bucket rate limiter.
2. **Embedding Model Weights** (`copilot/tools/embeddings.py`):
   - When `COPILOT_EMBEDDER="bge"`, `sentence_transformers.SentenceTransformer` is imported lazily and downloads `BAAI/bge-small-en-v1.5` from Hugging Face Hub (cached under `~/.cache/huggingface/`).
   - When `COPILOT_EMBEDDER="hash"`, a deterministic local MD5 token hashing embedder is used (no PyTorch/model weights required).
3. **PDF Text Extraction** (`copilot/tools/pdf.py`):
   - Uses `pymupdf` (C-extension binary `fitz`) at runtime to read in-memory byte streams of downloaded PDFs.
4. **FastAPI Background Threads** (`copilot/service.py`):
   - Executes `Copilot.run_live()` inside a `concurrent.futures.ThreadPoolExecutor` worker thread, streaming event dicts over thread-safe `threading.Condition` queues to HTTP clients.
