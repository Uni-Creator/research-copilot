# Runtime Entry Points & Interface Registry

This document catalogs all execution entry points, service APIs, CLI commands, background workers, and evaluation runners in `research-copilot`.

---

## 1. Command-Line Interface (CLI)

The CLI runs the full research pipeline locally in-process, printing real-time event traces to the terminal and saving outputs to disk.

- **Source File**: [copilot/cli.py](../../copilot/cli.py)
- **Invocation**:
  ```bash
  python -m copilot.cli "How does reinforcement learning from human feedback compare to direct preference optimization?" [options]
  ```

### CLI Arguments & Options

| Option | Type | Default | Description |
|---|---|---|---|
| `question` | Positional | (Required) | The research question to investigate. |
| `--year-from` | Integer | None | Earliest publication year filter. |
| `--year-to` | Integer | None | Latest publication year filter. |
| `--max-papers` | Integer | 20 | Maximum number of papers to shortlist and read. |
| `--until` | Choice | `critic` | Stop pipeline early: `retrieve`, `read`, `write`, `critic`. |
| `--out` | Path | `traces/last-run` | Output directory where artifacts will be saved. |
| `--no-fulltext` | Flag | False | Read paper abstracts only (skips PDF downloading; faster and cheaper). |
| `--embedder` | Choice | `bge` | Override embedder: `bge` (SentenceTransformer) or `hash` (offline test embedder). |
| `--brief` | Flag | False | Print one-line summary per step instead of detailed event payloads. |

### Artifacts Written
When execution completes, `copilot/render.py:save_run` writes:
- `<out_dir>/report.md`: Complete literature review report with bibliography.
- `<out_dir>/report.html`: Self-contained HTML report with CSS styling and method tables.
- `<out_dir>/run.json`: Complete serialized `CopilotState` dictionary including plan, notes, and traces.

---

## 2. HTTP REST & Server-Sent Events (SSE) API

Production-ready web service wrapping `CopilotFactory` with background execution, thread pool concurrency control, and real-time SSE streaming.

- **Source File**: [copilot/service.py](../../copilot/service.py)
- **Invocation**:
  ```bash
  uvicorn copilot.service:create_app --factory --port 8000
  ```
  Or via Makefile: `make api`

### HTTP Endpoints

| Method | Path | Description | Response Codes |
|---|---|---|---|
| `POST` | `/runs` | Start a new research task in background worker thread. | `202 Accepted` (`{"id": "...", "links": {...}}`) |
| `GET` | `/runs/{id}` | Poll current run status, short summary, token usage, and errors. | `200 OK`, `404 Not Found` |
| `GET` | `/runs/{id}/events` | Real-time Server-Sent Events (SSE) stream. Emits agent trace events and finishes with a `done` event. Supports HTTP `Last-Event-ID` header. | `200 OK` (`text/event-stream`) |
| `GET` | `/runs/{id}/report.md` | Finished Markdown report. | `200 OK` (`text/plain; charset=utf-8`) |
| `GET` | `/runs/{id}/report.html`| Styled HTML report. | `200 OK` (`text/html; charset=utf-8`) |
| `GET` | `/runs/{id}/run.json` | Complete serialized run state dictionary. | `200 OK` (`application/json`) |
| `GET` | `/health` | Health check reporting active and queued runs. | `200 OK` (`{"status": "ok", "runs_cached": N}`) |

### Authentication & Concurrency
- **API Key**: If environment variable `COPILOT_API_KEY` is set, requests require header `x-api-key: <key>` or query parameter `?api_key=<key>` (for EventSource SSE clients).
- **Worker Pool**: Uses `concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)`.
- **Memory Retention**: Keeps the most recent 50 runs in memory, automatically evicting oldest runs. All runs are also persisted to disk under `runs/<id>/`.

---

## 3. Interactive Web UI (Streamlit)

Graphical interface featuring real-time agent execution progress, claim verification tabs, and direct downloads.

- **Source File**: [ui/app.py](../../ui/app.py)
- **Invocation**:
  ```bash
  streamlit run ui/app.py
  ```
  Or via Makefile: `make ui`

### Key UI Features
- **Security**: Optional password protection gated by `COPILOT_UI_PASSWORD`.
- **Dynamic Feedback**: Real-time expandable execution status displaying agent thinking, search terms, and verified claims.
- **Tabbed Results**:
  1. *Report*: Markdown literature review.
  2. *Papers*: Interactive table of shortlisted papers, citation counts, and extracted notes.
  3. *Verification*: Breakdown of claim check verdicts (`supported`, `partial`, `unsupported`) with cited passage snippets.
  4. *Trace*: Complete timestamped audit trail of agent activities.
  5. *Download*: Direct download links for Markdown, HTML, and JSON representations.

---

## 4. Evaluation & Benchmark Harness

Tools for evaluating retrieval quality, claim grounding, and ablation performance against curated benchmark datasets.

- **Benchmark Runner**: [eval/run_eval.py](../../eval/run_eval.py)
  ```bash
  python -m eval.run_eval --topics eval/datasets/slp.json --out eval/results
  # or via Makefile
  make eval
  ```
- **Dataset Validator**: [eval/verify_dataset.py](../../eval/verify_dataset.py)
  ```bash
  python -m eval.verify_dataset --dataset eval/datasets/slp.json
  ```
  Validates that golden paper DOIs and titles in benchmark datasets resolve against academic APIs.

---

## 5. Container & Production Entrypoint

- **Container Script**: [scripts/start.sh](../../scripts/start.sh)
- **Container Invocation**:
  ```bash
  # Starts UI by default (port 7860)
  ./scripts/start.sh
  # Starts API when APP=api (port 8000)
  APP=api PORT=8000 ./scripts/start.sh
  ```
- **Hugging Face Spaces**: Automatically disables Streamlit CORS and XSRF if `SPACE_ID` is present, allowing iframe embedding.

---

## 6. Codebase Map & Maintenance Entrypoint

- **Post-Change All-in-One Synchronizer**: [scripts/sync_codebase.py](../../scripts/sync_codebase.py)
  ```bash
  make sync
  # or: uv run python scripts/sync_codebase.py [--full-test]
  ```
  Runs linter, executes targeted (or full) tests, regenerates the Mermaid diagram and reverse dependency table in `docs/codebase/dependency-graph.md`, updates `dependency-inventory.json`, and validates doc links.

- **Dependency Map & Blast Radius Analyzer**: [scripts/codebase_map.py](../../scripts/codebase_map.py)
  ```bash
  # Check git changes and report blast radius / recommended tests:
  python scripts/codebase_map.py --incremental   # (or `make map`)

  # Validate documentation links and verify 0 cycles:
  python scripts/codebase_map.py --check         # (or `make map-check`)

  # Rebuild dependency inventory:
  python scripts/codebase_map.py --rebuild
  ```

---

## 7. Test Suite Entrypoints

- **Full Suite**:
  ```bash
  .venv/bin/pytest -q
  ```
- **Targeted Subsystem Suites**:
  - State & reducers: `.venv/bin/pytest tests/test_state.py -q`
  - Agent behavior: `.venv/bin/pytest tests/test_agents.py tests/test_planner.py -q`
  - Graph & routing: `.venv/bin/pytest tests/test_graph.py tests/test_full_graph.py -q`
  - Tools & APIs: `.venv/bin/pytest tests/test_tools.py tests/test_openalex.py tests/test_reading.py -q`
  - Web API & SSE: `.venv/bin/pytest tests/test_service.py -q`
  - Evaluation suite: `.venv/bin/pytest tests/test_eval.py tests/test_dataset.py -q`
