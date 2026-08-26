# Synapse project status

Last updated: 2026-08-25

## Current production status

| Item | Value |
| --- | --- |
| Production default | **baseline** |
| Domain curriculum prior | **opt-in experimental** |
| Edge classifier | **experimental only** (not promoted) |
| Proposal lifecycle | LLM → validate → review → apply → rollback |

## Experimental features

- `domain_curriculum_prior` — closed-world selection from frozen inventories
- `domain_prior_edge_classifier` — same selection + pair classification (costly; domain-sensitive)

## Supported domains (inventories)

| Domain | Version | Notes |
| --- | --- | --- |
| compiler_construction | v1 | frozen |
| distributed_systems | v1 | frozen |
| stream_processing | v1 | frozen |
| cloud_computing | v1 | frozen |
| frontend_engineering | v1 | frozen |
| backend_engineering | v1 | frozen |
| databases | **v2** | prerequisite-structure revision |
| data_engineering | **v2** | prerequisite-structure revision |
| security | v1 | frozen |
| machine_learning | v1 | frozen |

Coverage: **33 / 40** quality cases mapped (82.5%). Unmapped: mathematics (4) + remaining programming cases (3).

## Known limitations

- Seven benchmark cases lack inventories → product fallback to baseline
- Domain prior increases latency/cost vs baseline
- Edge quality still domain-dependent
- Expanding coverage requires human-reviewed inventories (no gold-derived construction)

## Evaluation status

Experimentation phase **complete**. Domain-prior track frozen after databases/data_engineering v2.

Key reference artifacts (committed under `results/canonical/`):

- `2026-08-25_104759_quality_gpt-4o-mini_baseline.json` — mapped expansion baseline
- `2026-08-25_111954_inventory_v2_targeted_comparison.json` — v2 keep decision
- `2026-08-24_153229_reliability_none.json` — deterministic reliability suite

Ad-hoc benchmark output goes to `results/benchmarks/` and is not committed (see `results/README.md`).

## Closed experiments (do not reopen)

| Experiment | Outcome |
| --- | --- |
| Concept-First | Rejected |
| Inventory pruning | Rejected as primary path |
| Open-ended coverage recovery | Rejected |
| Representation alignment | Measurement-only / limited |
| Edge-classifier prompt A/B | Not supported for promotion |

Historical evaluation modules and benchmark artifacts are retained for reproducibility. They are **not** product ingest strategies.

## Open engineering work

- Demo / portfolio presentation polish
- Optional: inventories for remaining mathematics / programming cases (separate review cycle)
- No further architecture or prompt experiments planned

## Code boundary

| Layer | Location |
| --- | --- |
| Production | `services/ingest.py` baseline path, topics/proposals/graph |
| Experimental runtime | `domain_curriculum_prior`, `domain_prior_edge_classifier`, `curriculum/` |
| Evaluation | `app/evaluation/` |
| Historical artifacts | `results/canonical/` (reference); local runs under `results/benchmarks/` |
