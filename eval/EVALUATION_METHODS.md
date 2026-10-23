# Research Copilot Evaluation Pipeline

This document describes the evaluation pipeline added for Research Copilot: what each metric measures, why it is included, how it is calculated, and how to interpret it. The design separates **known-item retrieval** from **open-ended research quality** so that failing to retrieve a small fixed list is not mistaken for proving every other retrieved paper irrelevant.

## 1. Evaluation goals

The pipeline measures five distinct dimensions:

1. **Known-item discovery:** Can the retriever find papers that the dataset explicitly says should be discoverable?
2. **Judged relevance and ranking:** Are retrieved papers relevant to the question, based on explicit graded judgments rather than membership in the must-cite list?
3. **Research-aspect coverage:** Does the shortlist contain papers that cover the question's important subtopics?
4. **Citation and claim grounding:** Does the cited source text support the claims in the generated review?
5. **Operational quality:** What are the latency, token cost, errors, iteration counts, and trace characteristics?

These scores are reported separately. There is intentionally no single combined score: a weighted aggregate could hide a severe weakness (for example, excellent recall but unsupported claims).

## 2. Dataset schema

The supplied `eval/datasets/slp_v2.json` preserves the six SLP topics and their existing `must_cite` anchors. It adds:

- `topic_aspects`: named research dimensions that can be audited across variants.
- `relevance_judgments`: an initially empty list for curated paper-level judgments.
- `relevance_judgment_instructions`: annotation guidance.

A relevance-judgment entry has this shape:

```json
{
  "title": "Paper title",
  "arxiv_id": "2401.01234",
  "doi": "10.xxxx/example",
  "grade": 2,
  "aspects": ["discrete_motion_tokens", "evaluation_limitations"],
  "rationale": "Directly studies discrete sign-motion token generation."
}
```

Only include identifiers that are known. `grade` is required and means:

- `0`: not relevant to the research question
- `1`: partially relevant or useful background
- `2`: directly relevant to at least one part of the question

`aspects` must contain IDs from that topic's `topic_aspects`. Use `rationale` to record why a paper received its label. The runtime does not manufacture judgments. Because the initial v2 dataset intentionally has no paper-level relevance judgments, the `judged_*` metrics return `null` until you annotate the pooled candidate set. `null` means **not evaluated**, not a score of zero.

### How to build a reliable judgment pool

1. Run all variants on the same topic and save their raw candidate pools.
2. Take the union of the candidates across variants; deduplicate by DOI/arXiv ID, then title.
3. Hide the producing variant/rank from annotators where practical.
4. Have a human label relevance 0/1/2 and applicable topic aspects. Ideally, double-label a subset and resolve disagreements.
5. Keep `must_cite` as a separate known-item list. Do not label a paper irrelevant just because it is not in `must_cite`.
6. Version the dataset whenever labels change, and record annotation provenance/rationale.

For stronger evaluation, label every paper in the pooled candidate set. Ranking metrics are less trustworthy when only a small or biased subset is judged.

## 3. Known-item retrieval metrics

These metrics use `must_cite`, which is explicitly a list of retrieval anchors and **not an exhaustive bibliography**.

### Paper identity matching

`match_paper` tries normalized identifiers first:

1. arXiv ID (normalizes URL forms and strips version suffixes)
2. DOI (normalizes `doi.org` URL and case)
3. title token Jaccard similarity with threshold `0.80`

Identifiers take precedence over title similarity. This reduces false misses caused by title formatting and versioned arXiv URLs. Title-only matching remains a fallback, so any fuzzy matching policy should be audited when a benchmark paper appears missing.

### Stage coverage

For a set of expected papers \(G\) and papers present at stage \(S\):

\[
\operatorname{coverage}(S)=\frac{|\{g\in G: g\text{ matches a paper in }S\}|}{|G|}
\]

- `coverage_pool`: present in the full retrieved candidate pool
- `coverage_shortlist`: retained in the shortlist selected for reading
- `coverage_cited`: cited in the final draft

These expose where known-item recall is lost: discovery, selection, or synthesis. If `must_cite` is empty, the metric is unavailable (`null`).

### Ranked known-item metrics

The backward-compatible fields `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `mrr`, and `average_precision` are calculated against `must_cite` only. The output includes `known_item_metric_basis: "must_cite_list"` to make this explicit.

- **Precision@k:** number of must-cite papers in the top k divided by k.
- **Recall@k:** number of must-cite papers in the top k divided by the total number of must-cite anchors.
- **nDCG@k:** discounted ranking gain for must-cite papers, normalized by the ideal ranking. In this known-item view, must-cite relevance is binary (1/0).
- **MRR:** reciprocal rank of the first must-cite paper; zero if none appears in the ranked pool.
- **Average precision:** mean precision at ranks where a must-cite is found, normalized by the number of must-cite anchors.

These answer: “Did we find and rank the specific papers the benchmark named?” They do **not** answer: “Were all retrieved papers relevant?”

## 4. Graded relevance metrics

The `judged_*` metrics use `relevance_judgments`, independently of `must_cite`. This is the appropriate layer for open-ended retrieval evaluation.

- **judged_precision@k:** among judged papers in the top-k window, the fraction with grade 1 or 2.
- **judged_recall@k:** judged relevant papers found in the top-k window divided by all grade-1/2 papers in the curated judgment set.
- **judged_nDCG@k:** ranking quality using graded gains: `gain = 2^grade - 1`, discounted by rank and normalized against the ideal ordering.
- **judged_mrr:** reciprocal rank of the first positively judged paper, after filtering the ranking to judged papers.
- **judged_average_precision:** average precision over the judged ranking and judged relevant set.
- **judgment_coverage_at_k:** fraction of returned top-k papers that have a judgment. This is essential context for interpreting the relevance scores.

Unjudged papers are **not silently treated as irrelevant** for judged precision, recall, MRR, or average precision. For judged nDCG, only judged items in the top-k window contribute gain, at their original ranks; the judgment coverage score shows how complete the top-k labels are. A small judgment pool can bias results, so pool and label candidates from all variants before comparing them.

When no judgments are available, these metrics are `null`. Do not report them as zero or use them to rank variants.

## 5. Topic / subquestion coverage

Two different mechanisms are available and should not be conflated.

### Planner subquestion lexical coverage

`subquestion_coverage` and `mean_subquestion_coverage` compare tokens in each planner-generated subquestion/query against tokens in shortlisted paper titles and abstracts. The output field `subquestion_coverage_method` is `lexical_token_overlap_title_abstract`.

This is cheap and deterministic, useful for detecting obvious missing vocabulary or planner/query issues. It is **not semantic relevance**, does not show that a paper contains supporting evidence, and can be inflated by common words or repeated terminology. Use it as a diagnostic only.

### Human-judged topic-aspect coverage

`judged_aspect_coverage` is available once paper judgments are annotated with `aspects`. An aspect counts as covered when at least one positively judged paper tagged with that aspect is present in the shortlist. The output `aspect_coverage` reports each aspect separately.

Aspects with no positive paper judgment remain `null` rather than being treated as uncovered. This prevents an incomplete annotation set from falsely penalizing a variant.

## 6. Claim-level citation support

The evaluator mechanically extracts sentences containing `[cite:...]` tags and passes each claim with only its cited evidence to `verify_claims`.

Verdict weights:

- `supported`: 1.0
- `partial`: 0.5
- `unsupported`: 0.0

\[
\operatorname{support\_rate}=\frac{N_{supported}+0.5N_{partial}}{N_{judged\ claims}}
\]

Outputs include `support_rate`, counts by verdict, `unsupported_claims`, and `claim_checks` with claim text, chunk IDs, verdict, and judge rationale. The detailed checks are written into each raw run artifact so the result can be audited.

### Evidence provenance

The verifier first tries to load the cited text chunk. If unavailable, it falls back to the paper abstract. The pipeline reports `evidence_basis_counts` for:

- `retrieved_text_chunk`
- `abstract_fallback`
- `missing`

Abstract-only checks should not be described as full-text verification. A high support rate based mostly on abstracts is weaker evidence than a high rate against the cited full-text passages.

`n_claim_judge_errors` identifies judge-call failures separately. A failed judge call is currently represented as unsupported by the critic helper, so review this count before interpreting support rate. `citation_reference_validity` checks whether citation chunk IDs resolve to known retrieved papers; it does not prove that a valid source supports the claim.

The judge model should differ from the Critic model where possible. Manually audit at least 20% of verdicts during development, including supported, partial, and unsupported cases. Record disagreements and revise the rubric or prompts instead of assuming the LLM judge is ground truth.

## 7. Multi-agent trajectory and reliability diagnostics

The raw artifact for each `(topic, variant)` records the serialized state, plan, papers, shortlist, draft, trace, errors, claim checks, and evidence text used during verification. Aggregate metrics also include:

- `iterations`: number of critic/revision iterations
- `trace_event_count`: number of trace events
- `route_action_counts`: observed `pass`, `revise_draft`, and `retrieve_more` route events, when present in trace data
- `errors`: state-level error count
- `n_papers_retrieved`, `n_papers_shortlisted`, `n_papers_cited`
- `tokens`, `judge_tokens`, `runner_seconds`, `evaluation_seconds`, `seconds`

Use traces to inspect handoffs, source failures, empty retrievals, runaway loops, and claim/evidence mismatches. Aggregate numbers alone rarely explain why a multi-agent run failed. The raw state and claim-level audit artifact should be inspected for regressions.

Recommended testing layers:

1. **Unit tests:** ID normalization, paper matching, metric edge cases, schema validation.
2. **Agent/contract tests:** planner output shape, source adapter failures, reader-to-writer evidence handoff, critic verdict mapping.
3. **End-to-end tests:** run the graph on a small fixed topic set and assert that the state terminates, required fields exist, and costs stay within budget.
4. **Regression tests:** replay saved topics and compare metrics and traces after code, prompt, or model changes.
5. **Fault tests:** simulate one source failing, missing evidence chunks, malformed planner output, and judge-call failure.

## 8. Cost and comparison protocol

For fair architecture comparisons:

- Use the same dataset, source configuration, `max_papers`, embedding model, and judge model.
- Use `--no-fulltext` across all variants when isolating orchestration effects from extra full-text evidence. Run a separate full-text experiment when measuring the complete system.
- Keep model/provider, prompt versions, and code revision fixed for a comparison.
- Run each topic/variant more than once when model nondeterminism matters; report mean and spread rather than a single best run.
- Compare quality and cost together. The Critic may improve grounding but increase runtime and token use.
- Avoid using the same model for generation, critique, and evaluation when an independent judge is available.

## 9. Commands

From the repository root:

```bash
# 1) Run offline metric tests (does not call any API)
uv run python -m unittest discover -s eval/tests -v

# 2) Validate dataset structure offline
uv run python -m eval.validate_dataset --topics eval/datasets/slp_v2.json

# 3) Verify known-item metadata against configured live sources (network/API access required)
uv run python -m eval.verify_dataset --topics eval/datasets/slp_v2.json

# 4) Run a quick, abstract-only comparison
COPILOT_MAX_PAPERS=10 uv run python -u -m eval.run_eval \
  --topics eval/datasets/slp_v2.json \
  --variants baseline,multi,multi_critic \
  --limit 1 --no-fulltext \
  --out eval/results/slp-smoke

# 5) Pool candidates across variants for human relevance labeling
uv run python -m eval.pool_judgments \
  --artifacts-dir eval/results/slp-smoke/artifacts \
  --topics eval/datasets/slp_v2.json \
  --out eval/annotation_pool.json

# 6) Run the full benchmark with full text enabled
uv run python -u -m eval.run_eval \
  --topics eval/datasets/slp_v2.json \
  --variants baseline,multi,multi_critic \
  --out eval/results/slp-full
```

Outputs include:

- `results.json`: compact per-topic/per-variant metrics
- `results.md`: aggregate table
- `run_eval.log`: execution and metric logs
- `metadata.json`: dataset, variants, judge model, K values, and run settings
- `artifacts/<topic>__<variant>.json`: raw state, papers, plan, trace, claim checks, and evidence text used for verification

The live-source dataset verification and full pipeline commands need the project's dependencies and configured credentials. The new pure-uv run python relevance metric tests are offline.

## 10. Reading results responsibly

- `coverage_pool = 0` means none of the named must-cite anchors matched the candidate pool; it does not mean all retrieved papers were irrelevant.
- `coverage_cited = 1` means all must-cite anchors were cited; it does not mean the citations support their claims.
- `judged_precision_at_k` and `judged_nDCG@k` are only meaningful when the judgment pool is broad enough and `judgment_coverage_at_k` is high.
- High lexical subquestion coverage can coexist with irrelevant papers or unsupported claims.
- High claim support can coexist with poor recall if the system writes a narrow answer from a small set of sources.
- A zero support rate must be investigated alongside `n_claim_judge_errors`, evidence provenance, and claim-check rationales.
- One topic is a smoke test, not evidence of general performance. Build a diverse multi-topic benchmark and report per-topic results as well as aggregate results.
