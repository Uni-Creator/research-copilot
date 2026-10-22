# Change Impact & Blast Radius Intelligence

This guide defines the downstream consequences of modifying modules within `research-copilot`. Use this reference to understand reverse dependencies, assess architectural risks, and run targeted test suites before submitting changes.

---

## High-Risk Core Contracts

Three modules form the structural backbone of the repository. Changes to their interfaces require special caution:

### 1. `copilot/graph/state.py` (Critical Risk)
- **Fan-In**: 22 internal modules (All agents, workflow, service, CLI, tools, render, eval).
- **Blast Radius**: Modifying `CopilotState`, `Paper`, `PaperNotes`, `Draft`, `Chunk`, or state reducers (`merge_papers`, `merge_dicts`, `merge_usage`) impacts the entire pipeline.
- **Contract Rules**:
  - Never rename state keys without checking LangGraph node name collision rules (node name must NOT equal state key).
  - Parallel-written fields MUST retain reducer annotations.
- **Required Tests**:
  ```bash
  .venv/bin/pytest tests/test_state.py tests/test_agents.py tests/test_graph.py tests/test_full_graph.py -q
  ```

### 2. `copilot/config.py` (High Risk)
- **Fan-In**: 11 internal modules.
- **Blast Radius**: Adding, removing, or changing defaults of `Settings` fields directly affects runtime initialization across all agents, HTTP timeouts, cache directories, and model selection.
- **Required Tests**:
  ```bash
  .venv/bin/pytest tests/test_agents.py tests/test_full_graph.py tests/test_graph.py tests/test_eval.py -q
  ```

### 3. `copilot/graph/workflow.py` & `routing.py` (High Risk)
- **Fan-In**: Factory, CLI, Service, and all test suites.
- **Blast Radius**: Modifying LangGraph node wiring, conditional edges, `Send("reader")` fan-out, or ablation truncation stages (`until`) alters pipeline control flow.
- **Required Tests**:
  ```bash
  .venv/bin/pytest tests/test_graph.py tests/test_full_graph.py -q
  ```

---

## Subsystem Blast Radius Matrix

| Modified Module | Risk Level | Downstream Modules Affected | Recommended Test Suite |
|---|---|---|---|
| [copilot/graph/state.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/graph/state.py) | **CRITICAL** | `agents.*`, `workflow`, `service`, `cli`, `render`, `tools.*`, `eval.*` | `pytest tests/test_state.py tests/test_full_graph.py` |
| [copilot/config.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/config.py) | **HIGH** | `agents.*`, `workflow`, `factory`, `service`, `cli`, `routing`, `sources` | `pytest tests/test_agents.py tests/test_graph.py` |
| [copilot/graph/workflow.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/graph/workflow.py) | **HIGH** | `factory`, `cli`, `service`, `eval.run_eval` | `pytest tests/test_graph.py tests/test_full_graph.py` |
| [copilot/graph/routing.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/graph/routing.py) | **HIGH** | `workflow`, `critic` | `pytest tests/test_graph.py` |
| [copilot/factory.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/factory.py) | **MEDIUM** | `service`, `ui.app` | `pytest tests/test_factory_ui.py tests/test_service.py` |
| [copilot/llm.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/llm.py) | **MEDIUM** | `agents.planner`, `agents.reader`, `agents.writer`, `agents.critic`, `workflow` | `pytest tests/test_agents.py tests/test_full_graph.py` |
| [copilot/progress.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/progress.py) | **LOW** | `agents.*`, `cli`, `ui.app` | `pytest tests/test_progress.py` |
| [copilot/render.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/render.py) | **LOW** | `cli`, `service`, `ui.app` | `pytest tests/test_full_graph.py` |
| [copilot/agents/planner.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/agents/planner.py) | **MEDIUM** | `workflow` | `pytest tests/test_planner.py tests/test_full_graph.py` |
| [copilot/agents/retriever.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/agents/retriever.py) | **MEDIUM** | `workflow`, `baseline` | `pytest tests/test_retrieval_metrics.py tests/test_full_graph.py` |
| [copilot/agents/reader.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/agents/reader.py) | **MEDIUM** | `workflow` | `pytest tests/test_reading.py tests/test_agents.py` |
| [copilot/agents/writer.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/agents/writer.py) | **MEDIUM** | `workflow`, `baseline` | `pytest tests/test_agents.py` |
| [copilot/agents/critic.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/agents/critic.py) | **MEDIUM** | `workflow`, `eval.metrics` | `pytest tests/test_agents.py tests/test_eval.py` |
| [copilot/tools/sources.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/sources.py) | **MEDIUM** | `retriever`, `factory`, `baseline`, `workflow` | `pytest tests/test_tools.py` |
| [copilot/tools/arxiv.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/arxiv.py) | **LOW** | `tools.sources` | `pytest tests/test_tools.py tests/test_retrieval_metrics.py` |
| [copilot/tools/openalex.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/openalex.py) | **LOW** | `tools.sources` | `pytest tests/test_openalex.py` |
| [copilot/tools/semantic_scholar.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/semantic_scholar.py) | **LOW** | `tools.sources` | `pytest tests/test_tools.py` |
| [copilot/tools/pdf.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/pdf.py) | **LOW** | `workflow`, `factory` | `pytest tests/test_reading.py` |
| [copilot/tools/chunking.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/chunking.py) | **MEDIUM** | `reader`, `writer`, `critic`, `baseline`, `render` | `pytest tests/test_reading.py tests/test_agents.py` |
| [copilot/tools/vectorstore.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/vectorstore.py) | **MEDIUM** | `reader`, `writer`, `factory`, `workflow` | `pytest tests/test_reading.py tests/test_agents.py` |
| [copilot/tools/embeddings.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/embeddings.py) | **LOW** | `vectorstore`, `factory`, `workflow` | `pytest tests/test_reading.py tests/test_full_graph.py` |
| [copilot/tools/ranking.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/ranking.py) | **LOW** | `retriever` | `pytest tests/test_retrieval_metrics.py` |
| [copilot/tools/http.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/http.py) | **MEDIUM** | `arxiv`, `openalex`, `pdf`, `sources`, `workflow` | `pytest tests/test_tools.py` |
| [copilot/tools/papers.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/tools/papers.py) | **LOW** | `retriever` | `pytest tests/test_state.py` |
| [eval/metrics.py](file:///home/prime/Desktop/Projects/research-copilot/eval/metrics.py) | **MEDIUM** | `eval.run_eval`, `eval.verify_dataset` | `pytest tests/test_eval.py tests/test_dataset.py` |
| [eval/run_eval.py](file:///home/prime/Desktop/Projects/research-copilot/eval/run_eval.py) | **LOW** | (None - evaluation entry point) | `pytest tests/test_eval.py` |
| [copilot/service.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/service.py) | **LOW** | (None - HTTP service entry point) | `pytest tests/test_service.py` |
| [copilot/cli.py](file:///home/prime/Desktop/Projects/research-copilot/copilot/cli.py) | **LOW** | (None - CLI entry point) | `pytest tests/test_full_graph.py` |
| [ui/app.py](file:///home/prime/Desktop/Projects/research-copilot/ui/app.py) | **LOW** | (None - Streamlit UI entry point) | `pytest tests/test_factory_ui.py` |

---

## Automated Impact Assessment Workflow

To dynamically check the impact of your working tree modifications before committing:

```bash
# 1. Run incremental impact calculation
python scripts/codebase_map.py --incremental   # (or `make map`)

# 2. Run the recommended tests reported by the tool
.venv/bin/pytest <recommended_tests> -q

# 3. Verify documentation links and DAG consistency
python scripts/codebase_map.py --check         # (or `make map-check`)
```
