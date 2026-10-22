# Codebase Documentation & Dependency Intelligence System

Welcome to the persistent codebase documentation for `research-copilot`. This documentation suite provides an architectural and dependency map of the repository to enable fast, targeted navigation without unnecessary full-codebase scans.

## Documentation Index

| Document | Purpose | Key Content |
|---|---|---|
| [architecture.md](architecture.md) | Subsystem architecture & contracts | LangGraph state graph, agent roles, reducers, failure recovery, budget guards |
| [dependency-graph.md](dependency-graph.md) | Module dependency graph & coupling | Mermaid diagrams, reverse dependencies, cyclic check (0 cycles), coupling metrics |
| [file-index.md](file-index.md) | File directory & symbols | File responsibilities, public classes, functions, and linked tests |
| [data-flow.md](data-flow.md) | End-to-end execution flows | Question -> Plan -> Multi-source Retrieval -> PDF Chunking -> Reading -> Writing -> Critic verification loop |
| [entrypoints.md](entrypoints.md) | Runtime entry points & interfaces | CLI, FastAPI SSE Service, Streamlit UI, Evaluation harness, Docker entrypoint |
| [change-impact.md](change-impact.md) | Blast radius & test selection | Downstream impact of modifying specific modules, reverse deps, and test targets |

## How to Use This Documentation

When working on a coding task:
1. **Identify the subsystem**: Consult the [architecture.md](architecture.md) or [file-index.md](file-index.md) to locate the relevant files for your feature or bug fix.
2. **Review data flows**: Check [data-flow.md](data-flow.md) to understand state contracts, reducers, and upstream/downstream expectations.
3. **Assess blast radius**: Consult [change-impact.md](change-impact.md) and [dependency-graph.md](dependency-graph.md) to determine reverse dependencies and which test suites must be executed.
4. **Inspect source code selectively**: Open only the relevant source modules and test files needed to complete the task.
5. **Run maintenance tool**: After modifying code, run `python scripts/codebase_map.py --incremental` (or `make map`) to review affected modules and recommended tests.
6. **Verify documentation**: Run `python scripts/codebase_map.py --check` (or `make map-check`) to ensure no broken links or stale dependency inventories exist.

## Automated Maintenance

- `make sync`: **All-in-one post-change workflow** ([scripts/sync_codebase.py](../../scripts/sync_codebase.py)). Runs linter, targeted/full tests, regenerates the Mermaid graph and reverse dependency tables in `dependency-graph.md`, updates `dependency-inventory.json`, and verifies link integrity.
- `make map`: Analyzes current git working tree modifications and prints affected modules and recommended tests ([scripts/codebase_map.py](../../scripts/codebase_map.py)).
- `make map-check`: Validates documentation links and verifies that the dependency graph remains a strict DAG (0 cycles).
- `python scripts/codebase_map.py --rebuild`: Re-scans all modules and updates `docs/codebase/dependency-inventory.json`.
