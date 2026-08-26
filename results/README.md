# Evaluation results

Synapse separates **version-controlled evaluation inputs** from **generated outputs**.

## Version controlled

| Location | Contents |
| --- | --- |
| `data/eval/` | Golden quality/reliability/audit datasets, pricing table |
| `data/curriculum/` | Frozen domain inventories |
| `results/canonical/` | Small curated benchmark evidence (baseline references) |

## Generated locally (not committed)

These directories are created by `python -m app.evaluation.runner` and related analysis commands:

| Directory | Typical contents |
| --- | --- |
| `results/benchmarks/` | Quality, reliability, audit, and stability run JSON + markdown |
| `results/failure_analysis/` | Offline `--analyze`, `--rescore`, attribution reports |
| `results/curriculum/` | Inventory checks, domain coverage, v1/v2 comparisons |

Re-run benchmarks anytime from the datasets under `data/eval/`. You do not need historical run files to reproduce scores.

## Canonical artifacts

Files under `results/canonical/` are intentionally kept in Git as reference evidence:

| File | Purpose |
| --- | --- |
| `2026-08-25_104759_quality_gpt-4o-mini_baseline.json` | Baseline quality benchmark on the 40-case set (mapped expansion run) |
| `2026-08-24_153229_reliability_none.json` | Deterministic reliability suite (no LLM) |
| `2026-08-25_111954_inventory_v2_targeted_comparison.json` | Databases / data engineering inventory v1 vs v2 comparison summary |

Markdown summaries alongside each JSON are included for quick reading.

## Runtime data (also not committed)

- `backend/data/synapse.db` — local graph database
- `backend/data/llm_usage.jsonl` — optional LLM usage log (`SYNAPSE_LOG_LLM_USAGE`)
- `backend/data/proposal_events.jsonl` — proposal lifecycle events

Evaluation runs can redirect DB and event logs via `SYNAPSE_DB_PATH` and `SYNAPSE_PROPOSAL_EVENTS_PATH` (see `backend/app/evaluation/isolation.py`).
