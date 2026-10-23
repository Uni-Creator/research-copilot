# Installing the evaluation pipeline patch

Copy the files in this folder into the matching paths in the Research Copilot repository. Keep the new `eval/retrieval_metrics.py`, `eval/validate_dataset.py`, `eval/tests/`, `eval/EVALUATION_METHODS.md`, and `eval/datasets/slp_v2.json` files. Replace `eval/metrics.py`, `eval/run_eval.py`, and `eval/verify_dataset.py` with the supplied versions.

The patch preserves existing metric field names for compatibility. The legacy `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `mrr`, and `average_precision` fields remain based on `must_cite`. New `judged_*` metrics use independent graded `relevance_judgments` and are `null` until those judgments are added.

## Run sequence

```bash
# Offline checks; no LLM or source API calls
uv run python -m unittest discover -s eval/tests -v
uv run python -m eval.validate_dataset --topics eval/datasets/slp_v2.json
uv run python -m py_compile eval/metrics.py eval/run_eval.py eval/retrieval_metrics.py eval/verify_dataset.py

# Verify the paper IDs against configured sources (requires network/API access)
uv run python -m eval.verify_dataset --topics eval/datasets/slp_v2.json

# Smoke run, all variants using abstracts for a fairer architecture comparison
COPILOT_MAX_PAPERS=10 uv run python -u -m eval.run_eval \
  --topics eval/datasets/slp_v2.json \
  --variants baseline,multi,multi_critic \
  --limit 1 --no-fulltext --judge-model <INDEPENDENT_JUDGE_MODEL> \
  --out eval/results/slp-smoke
```

The evaluator now writes a raw JSON artifact per topic/variant under `eval/results/.../artifacts/`. Inspect these before trusting a metric. Artifacts include the candidate pool, shortlist, planner output, draft, trace, claim checks, and evidence text used for verification. To create a pooled annotation file for human relevance labels, run `uv run python -m eval.pool_judgments --artifacts-dir eval/results/slp-smoke/artifacts --topics eval/datasets/slp_v2.json --out eval/annotation_pool.json`. This script does not assign grades automatically. They can contain full abstracts and retrieved evidence, so do not commit them blindly if the project has data-sharing restrictions.

## Important limitation

The supplied dataset has empty relevance judgments by design. This avoids fabricating gold labels. Until the pooled candidates are labeled, open-ended judged precision/recall/nDCG and human-judged aspect coverage will be `null`; known-item coverage and support verification still run. See `eval/EVALUATION_METHODS.md` for definitions and annotation instructions.
