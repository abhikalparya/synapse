# Evaluation datasets

The benchmark measures agreement with curated reference structures and does not
claim that there is only one universally correct learning graph.

| File | Role |
| --- | --- |
| `learning_graph_eval_v1.jsonl` | 40-case quality set (backward compatible; extra fields allowed) |
| `learning_graph_quality_v1.jsonl` | Same 40 cases with required/optional/acceptable/alias annotations on the worst cases |
| `learning_graph_reliability_v1.jsonl` | Deterministic adversarial fixtures (no LLM) |
| `graph_audit_eval_v1.jsonl` | Known structural/semantic graph issues for audit scoring |
| `pricing_v1.json` | Versioned USD/million-token table |

Regenerate quality JSONL from Python:

```bash
cd backend
python -m app.evaluation.golden_v1
python -m app.evaluation.reliability_cases
python -m app.evaluation.audit_cases
```

## Quality schema

Each line is one JSON object. Required fields are unchanged from v1.

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Stable unique id |
| `category` | yes | Domain label |
| `difficulty` | yes | `beginner` / `intermediate` / `advanced` |
| `goal` | yes | Learning objective (ingest input) |
| `input_notes` | no | Optional pasted notes; `null` if unused |
| `gold_topics` | yes | Reference topic titles (backward compatible) |
| `gold_dependencies` | yes | `[from, to]` in **Synapse direction**: `from` requires `to` |
| `required_topics` | no | Must be present for completeness; default = `gold_topics` |
| `optional_topics` | no | Reasonable extras; not hallucinations; not required for recall |
| `required_dependencies` | no | Must-have edges; default = `gold_dependencies` |
| `acceptable_dependencies` | no | Valid alternatives that are not mandatory (count toward precision) |
| `topic_aliases` / `aliases` | no | `{ "Gold Title": ["synonym", ...] }` — curated by hand, never LLM-invented |
| `allowed_extra_topics` | no | Alias of optional extras (legacy name) |
| `gold_topic_summaries` | no | Expand/quiz/ask latency prompts only |
| `dataset_version` | no | e.g. `learning_graph_quality_v1` |
| `notes` | no | Why this reference graph is shaped this way |

If `required_topics` is set, other `gold_topics` are treated as optional.

## Matching

Titles are compared without an LLM judge:

1. Unicode NFKD, casefold, punctuation stripped, articles dropped, light plural stemming
2. Curated `aliases` / `topic_aliases`
3. Token containment (shorter title’s tokens ⊆ longer) or Jaccard ≥ 0.5
4. Single tokens shorter than 4 characters do **not** match via containment (`SQL` ≠ `SQL Injection`)

Reversed required edges are classified as `WRONG_DEPENDENCY_DIRECTION` and counted in
`dependency_direction_error_rate` rather than only as one false positive plus one false negative.

## Design notes

- 40 quality cases, 4 per domain, mixed difficulty.
- Gold graphs are small (typically 4–7 nodes) and acyclic by construction.
- Worst/ambiguous cases got aliases and optional/acceptable structure; the rest are unchanged.
- Reliability fixtures are hand-written invalid outputs, not random model samples.
