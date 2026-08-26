# Curriculum priors (experimental)

Versioned, human-reviewed **domain concept inventories** used only by the opt-in
`domain_curriculum_prior` generation strategy.

## Purpose

Synapse's evaluation showed that most persistent missing concepts are
**goal-derived** curriculum prerequisites (open-world inference from a learning
goal), not extractable from source documents (the quality dataset is mostly
goal-only).

Open-ended concept invention (coverage recovery / Concept-First) did not improve
quality. This prior replaces invention with **closed-world selection** from a
bounded, reviewed inventory.

## Separation from evaluation gold

These files are **not** gold evaluation answers.

Do **not**:

- auto-generate inventories from `data/eval/*.jsonl`
- copy `gold_topics` / `gold_dependencies` into inventories
- load failure-analysis artifacts to mint concepts
- treat inventory membership as a score target at runtime
- import `data/eval/` from runtime curriculum modules

It is acceptable for an inventory title to *overlap* a gold title when that
overlap reflects general domain curriculum knowledge. Inventories must be authored
from public/general curriculum knowledge and reviewed independently.

Planning metadata (`domain_prioritization_v1.json`) may use eval/failure artifacts
to rank domains. That file is **not** loaded by generation runtime.

## Versioning

Files are named `{domain}_v{N}.json`. Freeze a version before any live benchmark
run. Do not silently expand an inventory after observing model failures on that
run. Improvements require `{domain}_v2` (or later) with review rationale.

Each domain-prior execution records `inventory_version` and `inventory_hash` for
reproducibility.

## Domain resolution and fallback

1. Explicit `curriculum_domain` / `SYNAPSE_CURRICULUM_DOMAIN`
2. Case → domain map (`case_domain_map_v1.json`)
3. Optional category → domain hints in the case map

If unresolved or inventory missing:

- **Product default:** fall back to **baseline** (`fallback_reason` recorded)
- **Strict callers:** set `require_domain_prior=true` → `DOMAIN_UNRESOLVED` /
  `DOMAIN_PRIOR_UNAVAILABLE` (no silent pretend-prior)

## Review policy

Before freezing a version, verify:

1. Unique concept IDs and titles
2. Non-empty descriptions
3. Aliases do not ambiguously map to multiple concepts
4. Optional inventory prerequisites reference known IDs only
5. No cycles in inventory prerequisite edges
6. Provenance / source / created_at / reviewed_at / review_status recorded

## Inventory expansion workflow

1. Select domain (use `domain_prioritization_v1.json`; freeze the batch list)
2. Draft inventory from independent domain knowledge
3. Validate schema (`python -m app.evaluation.runner --curriculum-inventory-check`)
4. Review concept quality
5. Review aliases
6. Review prerequisite structure
7. Freeze version (do not mutate after freeze)
8. Run coverage gate (evaluation-only; not a construction signal)
9. Run domain-prior benchmark vs baseline
10. Compare; keep or revise as **v2** (never silently overwrite v1 after seeing gold failures)

## Experiment configuration

- `case_domain_map_v1.json` — benchmark case → domain (experiment config, not gold)
- `experiment_config_v1.json` — selection caps, fallback, coverage-gate thresholds
- `domain_prioritization_v1.json` — planning ranks only

## Current supported domains (frozen inventories)

| Domain | Version |
| --- | --- |
| compiler_construction | v1 |
| distributed_systems | v1 |
| stream_processing | v1 |
| cloud_computing | v1 |
| frontend_engineering | v1 |
| backend_engineering | v1 |
| databases | v1 |
| data_engineering | v1 |
| security | v1 |
| machine_learning | v1 |

Production Synapse remains **baseline** regardless of experiment outcomes.
`domain_prior_edge_classifier` remains experimental-only and is not the default
dependency path for domain-prior product use.

## Inventory v2 Review

`databases_v2` and `data_engineering_v2` revise **prerequisite structure**, not
benchmark Topic F1.

What changed (summary):

- **databases:** fix inverted SQL chain (SQL is foundational; SELECT/Joins require it);
  merge near-duplicate Relations into Tables; fold Database Indexing overview into
  Indexes; make Replication a direct Distributed Databases prerequisite; rewrite
  descriptions to state direct presuppositions.
- **data_engineering:** ETL requires Extract+Transform+Load as components; decouple
  Warehouses from Star Schema curriculum order; Batch no longer forced through ETL;
  Freshness requires Source Systems; clearer stage-direction wording.

Why: v1 showed acceptable topic coverage but Required Edge F1 regressions on these
two domains. Inventory edges are not the dependency generator, but concept set,
abstraction, and descriptions are what selection/dependency prompts see.

How reviewed: independent domain rationale per change_log entry; v1 frozen and hashed;
offline health diff before any live call; targeted benchmark only for these domains.

v2 is not claimed to be universally better—keep/reject per Required Edge metrics after
the targeted comparison.

### Targeted v2 benchmark outcome (n=3, curated_alias + edge_calibrated)

| Domain | Decision | Notes |
| --- | --- | --- |
| databases | KEEP_V2 | Required Edge F1 0.083 → 0.296; mild Topic F1 drop |
| data_engineering | KEEP_V2 | Required Edge F1 0.083 → 0.461; Topic F1 also up |

Active files: `databases_v2.json`, `data_engineering_v2.json` (v1 retained, unchanged).
