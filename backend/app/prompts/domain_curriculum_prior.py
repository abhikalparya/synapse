"""Prompts for closed-world domain curriculum prior selection (experimental)."""

from __future__ import annotations

from app.curriculum.inventory import DomainInventory


PROMPT_VERSION = "domain_curriculum_prior@v1"


def domain_curriculum_prior_metadata(domain: str, inventory_version: str) -> dict[str, str]:
    return {
        "generation_strategy": "domain_curriculum_prior",
        "prompt_variant": "domain_curriculum_prior",
        "prompt_version": PROMPT_VERSION,
        "curriculum_domain": domain,
        "curriculum_inventory_version": inventory_version,
    }


def build_selection_prompt(
    goal_text: str,
    inventory: DomainInventory,
    *,
    max_required: int,
) -> str:
    lines = []
    for c in inventory.concepts:
        alias = f" aliases={list(c.aliases)}" if c.aliases else ""
        lines.append(f"- id={c.id} | title={c.title} | level={c.level}{alias}")
        lines.append(f"  description: {c.description}")
    catalog = "\n".join(lines)
    return f"""You select prerequisite learning concepts from a CLOSED, reviewed domain inventory.

You must NOT invent new concept titles or IDs.
You may ONLY refer to concept_id values listed below.

Domain: {inventory.domain} (inventory {inventory.version})

Learning goal / request:
---
{goal_text.strip()}
---

Approved inventory:
{catalog}

Task:
Select a COMPACT set of REQUIRED prerequisite concepts for this goal.
Prefer direct prerequisites over dumping the whole catalog.
Select at most {max_required} REQUIRED concepts.

Kinds:
- REQUIRED: include in the learning graph
- RELEVANT_BUT_OPTIONAL: mention only if helpful; do not treat as required
- IRRELEVANT / OUT_OF_SCOPE: do not select

Respond with ONLY JSON:
{{
  "selected_concepts": [
    {{
      "concept_id": "exact.id.from.inventory",
      "kind": "REQUIRED",
      "reason": "short justification",
      "confidence": 0.0
    }}
  ]
}}

confidence is a ranking signal only (not a calibrated probability).
Every concept_id MUST exist in the inventory. Unknown IDs will be rejected.
"""


def build_dependency_prompt(
    goal_text: str,
    selected_titles: list[str],
    *,
    concept_details: list[tuple[str, str]] | None = None,
) -> str:
    detail_lines = []
    if concept_details:
        for title, desc in concept_details:
            detail_lines.append(f"- {title}: {desc}")
    else:
        for t in selected_titles:
            detail_lines.append(f"- {t}")
    catalog = "\n".join(detail_lines)
    allowed = ", ".join(repr(t) for t in selected_titles)
    return f"""You generate DIRECT prerequisite dependencies among an already selected concept set.

Edge semantics: [from, to] means from REQUIRES to (to is the prerequisite).

You MUST only use topic titles from this closed list:
{allowed}

Do NOT invent new topics.
Do NOT add endpoints outside the list.
Prefer direct prerequisites; avoid redundant transitive edges when a shorter chain exists.

Learning goal:
---
{goal_text.strip()}
---

Selected concepts:
{catalog}

Respond with ONLY JSON:
{{
  "dependencies": [
    {{"from": "Exact Title", "to": "Exact Title", "confidence": 0.7}}
  ]
}}
"""
