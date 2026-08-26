# Synapse evaluation

Measures graph generation quality, deterministic reliability, and audit behavior against golden datasets. Evaluation reuses production validators (`build_topics_and_dependencies`, cycle checks) so scores reflect real Synapse behavior—not a parallel implementation.

## Start here (core modules)

| Module | Role |
| --- | --- |
| `runner.py` | CLI entrypoint (`python -m app.evaluation.runner`) |
| `benchmark.py` | Quality benchmark orchestration |
| `metrics.py` | Deterministic topic/dependency matching and F1 scores |
| `schemas.py` | Shared datatypes (`EvalExample`, `GeneratedGraph`, …) |
| `dataset.py` | Load/filter JSONL golden cases |
| `baselines.py` | Linear and direct-LLM comparison systems |
| `synapse_system.py` | Production validation path on raw ingest JSON |
| `reliability.py` | Adversarial fixtures (no LLM) |
| `audit_eval.py` | Audit scoring against fixtures |
| `reporting.py` | Write JSON + markdown artifacts |
| `cost.py` / `latency.py` | Token cost estimates and latency summaries |
| `isolation.py` | Temp DB + event log for safe eval runs |
| `inspect.py` | Offline `--analyze` / `--rescore` on stored runs |
| `proposal_metrics.py` | Aggregates from proposal event log |

Golden datasets live in `data/eval/` (see `data/eval/README.md`).

## Historical / experimental modules

These support closed experiments and offline diagnosis. They are **not** product ingest paths:

| Area | Examples |
| --- | --- |
| Concept-First (rejected) | `concept_first_system.py`, `concept_first_compare.py` |
| Coverage recovery (rejected) | `coverage_recovery_system.py`, `coverage_recovery_compare.py` |
| Representation alignment | `representation_alignment_analysis.py` |
| Domain prior analysis | `curriculum_prior_system.py`, `curriculum_prior_analysis.py`, `final_40_case_comparison.py` |
| Edge classifier experiments | `edge_classifier_system.py`, `edge_classifier_prompt_ab.py` |
| Failure attribution | `stability_analysis.py`, `persistent_failure_attribution.py`, `pure_relationship_analysis.py`, `missing_concept_information.py`, `node_edge_attribution.py`, `inventory_attribution.py` |
| Matching calibration | `matching_calibration.py`, `edge_ambiguity.py`, `topic_equivalence.py` |

Product default generation remains **baseline**. Experimental runtime strategies are opt-in via the ingest API (`domain_curriculum_prior`, `domain_prior_edge_classifier`).

## Common commands

From the repo root:

```bash
make test                  # full pytest suite (no LLM)
make eval-reliability      # deterministic reliability benchmark
make eval-audit            # audit benchmark, structural only (--no-llm)
make curriculum-check      # offline inventory validation
```

From `backend/`:

```bash
# Quality benchmark (requires LLM API key)
python -m app.evaluation.runner --benchmark quality --systems synapse direct linear --limit 5

# Multi-model comparison
python -m app.evaluation.runner --benchmark quality --models gpt-4o-mini gpt-4o --skip-ops-latency

# Re-score stored generations without new LLM calls
python -m app.evaluation.runner --rescore ../results/benchmarks/<run>.json

# Failure analysis from an existing quality artifact
python -m app.evaluation.runner --analyze ../results/benchmarks/<run>.json

# Proposal metrics from event log
python -m app.evaluation.runner --proposal-metrics-only
```

Or from the repo root: `python scripts/run_eval.py --help`

## Where results go

| Output | Default path | Committed? |
| --- | --- | --- |
| Benchmark runs | `results/benchmarks/` | No — regenerate from datasets |
| Failure analysis | `results/failure_analysis/` | No |
| Curriculum reports | `results/curriculum/` | No |
| Reference evidence | `results/canonical/` | Yes — small curated set |

See `results/README.md` for the retention policy.

## Rescore and model comparison

- **`--rescore PATH`** — recompute metrics on stored per-example generations (no LLM).
- **`--matching-modes strict fair curated_alias`** — compare matching policies on the same artifact.
- **`--models M1 M2`** — run quality benchmark once per model, emit a comparison summary.
- **`--repetitions N`** / **`--generations N`** — multi-generation stability runs; analyze with `--stability-analysis`.

Artifacts record model, provider, prompt hash, seed, and per-example outputs for diffing across runs.

## Intentionally not covered yet

- Retrieval / scoped Q&A grounding evaluation
- Agent or MCP tool-selection evaluation
- Automated CI quality gates (requires API keys)
- Expand / reshape / audit **quality** beyond ingest-graph benchmarks (latency suite only)
- LLM-as-judge scoring (matching stays deterministic)

## Tests

Evaluation tests live in `backend/tests/test_*.py` (metrics, benchmark, reliability, curriculum, etc.). Run `make test` from the repo root.
