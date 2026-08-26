# Curriculum inventory guide

How to add or revise a **domain curriculum prior** inventory without leaking evaluation gold.

## Rules

Do **not**:

- Copy `gold_topics` / `gold_dependencies` into inventories
- Construct concepts from failure-analysis or stability artifacts
- Mutate a frozen `{domain}_vN.json` after a scored run
- Import `data/eval/` from curriculum runtime code

Do:

- Author from independent domain knowledge
- Record provenance / review metadata
- Bump version (`v2`, `v3`, …) for structural changes
- Keep production default on **baseline**

## Workflow

1. **Prioritize** the domain (case count, product relevance, missing-concept impact). Planning metadata may use eval artifacts; the inventory itself must not.
2. **Draft** `{domain}_v1.json` (or `_vN` for revisions) under `data/curriculum/` using the existing schema: `id`, `title`, `description`, `aliases`, `level`, optional `prerequisite_ids`.
3. **Validate** offline (no API key):

   ```bash
   cd backend && python3 -m app.evaluation.runner --curriculum-inventory-check
   ```

4. **Review concepts** — reusable learning units; avoid lesson titles and filler.
5. **Review aliases** — no ambiguous cross-concept aliases.
6. **Review prerequisite edges** — direct “A requires B” only; avoid curriculum-order and transitive density.
7. **Freeze** the version; wire it in `experiment_config_v1.json` `inventory_files` and map cases in `case_domain_map_v1.json`.
8. **Coverage gate** — evaluation-only gold overlap check; not a construction signal.
9. **Benchmark** domain prior vs baseline on mapped cases for that domain.
10. **Decide** KEEP / NEEDS_REVIEW. Do not silently overwrite after seeing gold failures—ship a new version instead.

## Metadata checklist

- `domain`, `version`, `source` / `provenance`
- `created_at`, `reviewed_at`, `review_status`
- For revisions: `parent_version`, `change_log[]` with `change_type`, `old`, `new`, `reason`, `rationale`

## Product fallback

Unresolved domain or missing inventory → **baseline**, with `fallback_reason` in proposal `generation_meta` (`DOMAIN_UNRESOLVED` / `DOMAIN_PRIOR_UNAVAILABLE`), unless the caller sets `require_domain_prior=true`.
