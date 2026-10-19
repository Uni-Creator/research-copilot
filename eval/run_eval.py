
"""Run benchmark ablations with persistent logs and incremental results.

Examples:
    python -m eval.run_eval --topics eval/datasets/slp.json \
        --variants baseline,multi,multi_critic --out eval/results

    COPILOT_MAX_PAPERS=5 python -u -m eval.run_eval \
        --topics eval/datasets/slp.json --variants baseline \
        --limit 1 --no-fulltext --out eval/results-debug
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable

from copilot.baseline import run_baseline
from copilot.config import Settings
from copilot.graph.state import CopilotState, SearchFilters
from copilot.graph.workflow import build_copilot
from copilot.llm import get_llm
from copilot.tools.sources import SearchFn, build_sources
from eval.metrics import DEFAULT_KS, evaluate_run

Runner = Callable[
    [str, SearchFilters],
    tuple[CopilotState, Callable[[str], str | None]],
]

# Base metrics averaged across successfully executed runs.
# Ranked-retrieval columns (precision_at_k etc.) are appended dynamically from --ks.
BASE_COLUMNS = [
    "coverage_pool",
    "coverage_shortlist",
    "coverage_cited",
    "mrr",
    "average_precision",
    "mean_subquestion_coverage",
    "support_rate",
    "n_claims",
    "n_papers_cited",
    "stripped_citations",
    "tokens",
    "judge_tokens",
    "seconds",
]


def _ranked_columns(ks: tuple[int, ...]) -> list[str]:
    cols: list[str] = []
    for k in sorted(ks):
        cols += [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}"]
    return cols


def build_columns(ks: tuple[int, ...]) -> list[str]:
    """Return the full ordered column list for a given set of k values."""
    # Put ranked-retrieval metrics right after coverage_cited
    base = list(BASE_COLUMNS)
    idx = base.index("coverage_cited") + 1
    return base[:idx] + _ranked_columns(ks) + base[idx:]


COLUMNS = build_columns(DEFAULT_KS)  # default; overridden at runtime


def configure_logging(out: Path, level: str = "INFO") -> logging.Logger:
    """Log to the terminal and to a persistent file."""
    out.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("eval.run_eval")
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(
        out / "run_eval.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _as_mapping(value: Any) -> dict[str, Any]:
    """Convert common state/model objects to a dictionary when possible."""
    if isinstance(value, Mapping):
        return dict(value)

    if is_dataclass(value) and not isinstance(value, type):
        try:
            return asdict(value)
        except Exception:
            pass

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            result = model_dump()
            if isinstance(result, Mapping):
                return dict(result)
        except Exception:
            pass

    try:
        return vars(value)
    except (TypeError, AttributeError):
        return {}


def _field(value: Any, *names: str) -> Any:
    """Read the first available field from a mapping or object."""
    mapping = _as_mapping(value)
    for name in names:
        if name in mapping:
            return mapping[name]
        try:
            result = getattr(value, name)
            if result is not None:
                return result
        except (AttributeError, TypeError):
            pass
    return None


def _paper_line(paper: Any) -> str:
    """Format either a paper object, mapping, or plain string."""
    if isinstance(paper, str):
        return paper

    title = _field(paper, "title", "paper_title", "name")
    if not isinstance(title, str):
        title = repr(paper)[:300]

    arxiv_id = _field(paper, "arxiv_id", "arxivId", "arxiv")
    doi = _field(paper, "doi")
    source = _field(paper, "source", "provider")

    identifiers = []
    if arxiv_id:
        identifiers.append(f"arXiv={arxiv_id}")
    if doi:
        identifiers.append(f"DOI={doi}")
    if source:
        identifiers.append(f"source={source}")

    suffix = f" [{', '.join(map(str, identifiers))}]" if identifiers else ""
    return f"{title}{suffix}"


def log_state_diagnostics(
    logger: logging.Logger,
    state: Any,
    max_papers: int = 10,
) -> None:
    """Log available state keys and common retrieval collections.

    Field names vary between implementations. Missing fields are skipped;
    the function does not modify the state or assume a specific schema.
    """
    data = _as_mapping(state)
    if not data:
        logger.info("State diagnostics: unable to convert state to a mapping")
        logger.info("State representation: %s", repr(state)[:1500])
        return

    logger.info("State fields: %s", ", ".join(sorted(data.keys())))

    collection_names = (
        "papers",
        "retrieved_papers",
        "paper_pool",
        "candidates",
        "ranked_papers",
        "shortlist",
        "selected_papers",
        "chunks",
        "citations",
        "errors",
        "messages",
    )

    for name in (
        "papers", "shortlist", "plan", "errors",
        "trace", "draft", "usage",
    ):
        value = data.get(name)

        logger.info(
            "State.%s: type=%s",
            name,
            type(value).__name__,
        )

        if value is None:
            logger.info("  value=None")
            continue

        if name == "errors":
            if isinstance(value, (list, tuple, set)):
                for index, error in enumerate(value, 1):
                    logger.error("  errors[%d]: %r", index, error)
            else:
                logger.error("  errors value: %r", value)
            continue

        if name in {"papers", "shortlist"}:
            if isinstance(value, (list, tuple, set)):
                logger.info("  count=%d", len(value))
                for index, item in enumerate(list(value)[:max_papers], 1):
                    logger.info("  [%d] %s", index, _paper_line(item))
            else:
                logger.info("  value=%s", repr(value)[:3000])
            continue

        if name == "trace":
            logger.info("  value=%s", repr(value)[:6000])
        elif name == "draft":
            logger.info("  value=%s", repr(value)[:2500])
        else:
            logger.info("  value=%s", repr(value)[:3000])


def _mean(values: list[Any]) -> float | None:
    vals = [
        value for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return round(statistics.mean(vals), 3) if vals else None


def results_table(
    results: list[dict[str, Any]],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> str:
    """Build an aggregate Markdown table for each variant."""
    variants = list(dict.fromkeys(row["variant"] for row in results))
    columns = build_columns(ks) + ["eval_errors"]

    lines = [
        "| Variant | " + " | ".join(columns) + " | topics | failed |",
        "|---|" + "---|" * (len(columns) + 2),
    ]

    for variant in variants:
        rows = [r for r in results if r["variant"] == variant]
        successful = [r for r in rows if not r.get("execution_error")]

        cells = []
        for column in columns[:-1]:  # drop trailing eval_errors placeholder
            value = _mean([r.get(column) for r in successful])
            cells.append("-" if value is None else str(value))

        # This is the average number of evaluation/state errors, not exceptions
        # raised by the runner. Runner exceptions are counted under "failed".
        error_count = _mean([r.get("errors") for r in successful])
        cells.append("-" if error_count is None else str(error_count))

        lines.append(
            f"| {variant} | "
            + " | ".join(cells)
            + f" | {len(successful)} | {len(rows) - len(successful)} |"
        )

    return "\n".join(lines)


def save_results(out: Path, results: list[dict[str, Any]]) -> None:
    """Atomically persist completed runs after every topic/variant."""
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "results.json"
    json_tmp = out / "results.json.tmp"
    json_tmp.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    json_tmp.replace(json_path)

    table_path = out / "results.md"
    table_tmp = out / "results.md.tmp"
    table_tmp.write_text(results_table(results) + "\n", encoding="utf-8")
    table_tmp.replace(table_path)


def evaluate(
    topics: list[dict[str, Any]],
    variants: dict[str, Runner],
    judge_llm: Any,
    settings: Settings,
    logger: logging.Logger | None = None,
    on_result: Callable[[list[dict[str, Any]]], None] | None = None,
    ks: tuple[int, ...] = DEFAULT_KS,
    # backward-compat alias for tests that call evaluate(..., log=fn)
    log: Callable[..., None] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run each variant, recording timings, diagnostics, and failures."""
    if logger is None and log is not None:
        # tests pass a bare callable; wrap it in a minimal logger-like shim
        class _LogShim(logging.Logger):
            def __init__(self, fn: Callable[..., None]) -> None:
                super().__init__("eval.run_eval")
                self._fn = fn

            def _emit(self, level: str, msg: str, *args: Any) -> None:
                try:
                    self._fn(level, msg % args if args else msg)
                except Exception:
                    pass

            def info(self, msg: str, *args: Any, **kw: Any) -> None:  # type: ignore[override]
                self._emit("INFO", msg, *args)

            def warning(self, msg: str, *args: Any, **kw: Any) -> None:  # type: ignore[override]
                self._emit("WARNING", msg, *args)

            def error(self, msg: str, *args: Any, **kw: Any) -> None:  # type: ignore[override]
                self._emit("ERROR", msg, *args)

            def exception(self, msg: str, *args: Any, **kw: Any) -> None:  # type: ignore[override]
                self._emit("ERROR", msg, *args)

        logger = _LogShim(log)  # type: ignore[assignment]
    elif logger is None:
        logger = logging.getLogger("eval.run_eval")
    results: list[dict[str, Any]] = []
    total = len(topics) * len(variants)
    completed = 0

    for topic_index, topic in enumerate(topics, start=1):
        filters = SearchFilters(**topic.get("filters", {}))

        for name, runner in variants.items():
            completed += 1
            topic_id = topic["id"]
            question = topic["question"]

            logger.info("=" * 72)
            logger.info(
                "Run %d/%d | topic=%s | variant=%s",
                completed, total, topic_id, name,
            )
            logger.info("Question: %s", question)
            logger.info("Filters: %s", filters)
            logger.info("Starting runner")

            started = time.perf_counter()
            runner_seconds = 0.0
            evaluation_seconds = 0.0
            state = None
            row: dict[str, Any] = {
                "topic": topic_id,
                "variant": name,
                "execution_error": None,
            }

            try:
                runner_started = time.perf_counter()
                state, chunk_text = runner(question, filters)
                runner_seconds = time.perf_counter() - runner_started

                logger.info("Runner completed in %.1fs", runner_seconds)
                log_state_diagnostics(logger, state)

                logger.info("Starting metric evaluation")
                evaluation_started = time.perf_counter()
                metrics = evaluate_run(
                    topic,
                    state,
                    chunk_text,
                    judge_llm,
                    settings.verify_batch_size,
                    ks=ks,
                )
                evaluation_seconds = time.perf_counter() - evaluation_started

                # Preserve the evaluator's "error" field. Do not overwrite it.
                if isinstance(metrics, Mapping):
                    row.update(metrics)
                else:
                    raise TypeError(
                        f"evaluate_run returned {type(metrics).__name__}, "
                        "expected a mapping"
                    )

                logger.info(
                    "Metric evaluation completed in %.1fs",
                    evaluation_seconds,
                )

                logger.info(
                    "Metrics | pool=%s shortlist=%s cited=%s "
                    "support=%s claims=%s papers_cited=%s tokens=%s "
                    "judge_tokens=%s errors=%s evaluator_error=%r",
                    row.get("coverage_pool"),
                    row.get("coverage_shortlist"),
                    row.get("coverage_cited"),
                    row.get("support_rate"),
                    row.get("n_claims"),
                    row.get("n_papers_cited"),
                    row.get("tokens"),
                    row.get("judge_tokens"),
                    row.get("errors"),
                    row.get("error"),
                )

                missed = row.get("missed_in_shortlist")
                if missed:
                    logger.warning(
                        "Required papers missing from shortlist (%d):",
                        len(missed),
                    )
                    for title in missed:
                        logger.warning("  - %s", title)

                if row.get("errors"):
                    logger.warning(
                        "Evaluation/state error count is %s",
                        row["errors"],
                    )

                if row.get("error") is not None:
                    logger.warning(
                        "Evaluator reported error: %r", row["error"]
                    )

            except KeyboardInterrupt:
                logger.warning(
                    "Interrupted during topic=%s variant=%s; "
                    "previous completed runs have been saved",
                    topic_id, name,
                )
                raise

            except Exception as exc:
                logger.exception(
                    "Run failed | topic=%s | variant=%s",
                    topic_id, name,
                )
                row["error"] = repr(exc)
                row["execution_error"] = repr(exc)
                row["execution_traceback"] = traceback.format_exc()

                # Keep whatever diagnostics are available after a failure.
                if state is not None:
                    log_state_diagnostics(logger, state)

            finally:
                elapsed = time.perf_counter() - started
                row["runner_seconds"] = round(runner_seconds, 2)
                row["evaluation_seconds"] = round(evaluation_seconds, 2)
                row["seconds"] = round(elapsed, 1)

            results.append(row)

            logger.info(
                "Finished %d/%d | topic=%s | variant=%s | total=%.1fs | "
                "runner=%.1fs | evaluation=%.1fs | execution_error=%s",
                completed,
                total,
                topic_id,
                name,
                row["seconds"],
                row["runner_seconds"],
                row["evaluation_seconds"],
                row.get("execution_error"),
            )

            # Checkpoint after every completed run, not only at benchmark end.
            if on_result is not None:
                on_result(results)

    return results


def make_variants(
    names: list[str],
    settings: Settings,
    sources: dict[str, SearchFn],
    fulltext: bool,
) -> dict[str, Runner]:
    variants: dict[str, Runner] = {}
    kwargs: dict[str, Any] = {} if fulltext else {"fetch_pages": None}

    for name in names:
        if name == "baseline":
            llm = get_llm(
                settings.writer_model,
                settings.model_provider,
                max_tokens=3000,
            )
            variants[name] = (
                lambda question, filters, llm=llm: run_baseline(
                    question, filters, settings, llm, sources
                )
            )

        elif name in ("multi", "multi_critic"):
            copilot = build_copilot(
                settings,
                until="write" if name == "multi" else "critic",
                sources=sources,
                **kwargs,
            )
            variants[name] = (
                lambda question, filters, cp=copilot: (
                    cp.run(question, filters),
                    cp.chunk_text,
                )
            )

        else:
            raise SystemExit(f"Unknown variant {name!r}")

    return variants


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run literature-review evaluation variants with diagnostics."
    )
    parser.add_argument("--topics", default="eval/datasets/slp.json")
    parser.add_argument("--variants", default="baseline,multi,multi_critic")
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-fulltext", action="store_true")
    parser.add_argument("--embedder", choices=["bge", "hash"])
    parser.add_argument(
        "--judge-model",
        help="Defaults to the critic model; use another model for independence.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--ks",
        default="",
        help="Comma-separated k values for Precision/Recall/nDCG@k (default: 5,10,20).",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    logger = configure_logging(out, args.log_level)

    try:
        settings = Settings.from_env()
        if args.embedder:
            settings = replace(settings, embedder=args.embedder)

        topics_path = Path(args.topics)
        logger.info("Loading evaluation dataset: %s", topics_path)
        dataset = json.loads(topics_path.read_text(encoding="utf-8"))
        topics = dataset["topics"]

        if not isinstance(topics, list):
            raise ValueError("The dataset's 'topics' field must be a list.")

        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be greater than zero.")
            topics = topics[:args.limit]

        names = [name.strip() for name in args.variants.split(",") if name.strip()]
        if not names:
            raise ValueError("Specify at least one variant.")

        logger.info("Dataset: %s", dataset.get("name", topics_path))
        logger.info("Topics selected: %d", len(topics))
        logger.info("Variants: %s", ", ".join(names))
        logger.info("Full-text fetching enabled: %s", not args.no_fulltext)
        logger.info("Embedder override: %s", args.embedder or "default")
        logger.info("Output directory: %s", out.resolve())

        sources = build_sources(settings)
        judge = get_llm(
            args.judge_model or settings.critic_model,
            settings.model_provider,
            max_tokens=2000,
        )
        variants = make_variants(
            names,
            settings,
            sources,
            fulltext=not args.no_fulltext,
        )

        # Create valid empty checkpoint files before the first run.
        results: list[dict[str, Any]] = []
        save_results(out, results)

        def checkpoint(current: list[dict[str, Any]]) -> None:
            save_results(out, current)
            logger.info(
                "Checkpoint saved: %d completed runs -> %s",
                len(current),
                out.resolve(),
            )

        results = evaluate(
            topics,
            variants,
            judge,
            settings,
            logger,
            on_result=checkpoint,
            ks=tuple(int(k) for k in (args.ks or "").split(",") if k.strip()) or DEFAULT_KS,
        )

        _ks = tuple(int(k) for k in (args.ks or "").split(",") if k.strip()) or DEFAULT_KS
        table = results_table(results, ks=_ks)
        print("\n" + table)
        logger.info("Benchmark complete. Results saved to %s", out.resolve())
        return 0

    except KeyboardInterrupt:
        logger.warning(
            "Benchmark interrupted. Completed results, if any, remain in %s",
            out.resolve(),
        )
        return 130

    except Exception:
        logger.exception("Benchmark setup or execution failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
