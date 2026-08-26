"""Prompts for closed-world dependency pair classification (experimental).

Variants:
  edge_classifier_baseline — original classifier prompt (default)
  edge_classifier_fewshot_directness — same task + synthetic DIRECT/TRANSITIVE exemplars

Production ingest default remains baseline; this module is opt-in only.
"""

from __future__ import annotations

import hashlib
import os
from typing import Literal

from app.curriculum.edge_candidates import CandidatePair
from app.curriculum.selection import SelectedConcept

EdgeClassifierPromptVariant = Literal[
    "edge_classifier_baseline",
    "edge_classifier_fewshot_directness",
]

EDGE_CLASSIFIER_PROMPT_VARIANTS: tuple[EdgeClassifierPromptVariant, ...] = (
    "edge_classifier_baseline",
    "edge_classifier_fewshot_directness",
)

# Synthetic, domain-neutral exemplars only — must never include eval concepts.
FEWSHOT_DIRECTNESS_BLOCK = """
EXAMPLES (synthetic; follow the same DIRECT vs TRANSITIVE rules):

Example 1 — DIRECT prerequisite:
  Candidate: Functions → Variables
  Decision: REQUIRED
  Why: Understanding variables is directly necessary before understanding how functions
  manipulate values. This is a direct prerequisite, not merely related.

Example 2 — TRANSITIVE (do NOT emit as direct):
  Suppose: Functions → Variables and Variables → Data Types.
  Candidate: Functions → Data Types
  Decision: NOT_REQUIRED
  Why: Data Types is only indirectly upstream through Variables. Prefer the direct edge;
  do not mark the transitive shortcut REQUIRED.

Example 3 — RELATED but not required:
  Candidate: Databases → Caching
  Decision: NOT_REQUIRED
  Why: Related engineering topics are not automatically direct prerequisites.

Example 4 — Curriculum ordering is not a prerequisite:
  Candidate: Testing → Deployment
  Decision: NOT_REQUIRED
  Why: Being commonly taught before another topic does not establish a direct
  prerequisite relationship.
""".strip()

# Frozen baseline body (without few-shot). Used for hash stability tests.
_BASELINE_INSTRUCTIONS = """You classify DIRECT prerequisite relationships among a CLOSED set of concepts.

Edge semantics (Synapse):
  from → to  means  from REQUIRES to
  (to must be understood BEFORE from).

Only classify DIRECT prerequisites for this learning goal.
Do NOT mark as REQUIRED:
- general relatedness
- "often taught before"
- indirect / transitive prerequisites
- optional supporting knowledge
- topical similarity

If A requires B and B requires C, then A → C is usually NOT_REQUIRED unless C is itself a direct prerequisite of A.

You may ONLY use from_id / to_id values listed in the candidate pairs.
Do NOT invent concept IDs.
Do NOT invent new pairs."""

_RESPONSE_TAIL = """For each candidate pair, decide:
- REQUIRED — target is a direct prerequisite of source for this goal
- NOT_REQUIRED — not a direct prerequisite
- UNCERTAIN — only if you cannot decide; UNCERTAIN does not create an edge

Respond with ONLY JSON:
{{
  "decisions": [
    {{"from_id": "exact.id", "to_id": "exact.id", "decision": "REQUIRED"}}
  ]
}}

Include a decision for every candidate pair when possible.
Optional short "reason" fields are allowed but not required."""


def resolve_edge_classifier_prompt_variant(
    variant: str | None = None,
) -> EdgeClassifierPromptVariant:
    """Resolve from argument or ``SYNAPSE_EDGE_CLASSIFIER_PROMPT`` (default: baseline)."""
    raw = (
        variant
        if variant is not None
        else os.environ.get("SYNAPSE_EDGE_CLASSIFIER_PROMPT") or "edge_classifier_baseline"
    ).strip()
    key = raw.casefold().replace("-", "_")
    aliases = {
        "edge_classifier_baseline": "edge_classifier_baseline",
        "baseline": "edge_classifier_baseline",
        "edge_classifier": "edge_classifier_baseline",
        "domain_prior_edge_classifier": "edge_classifier_baseline",
        "edge_classifier_fewshot_directness": "edge_classifier_fewshot_directness",
        "fewshot_directness": "edge_classifier_fewshot_directness",
        "fewshot": "edge_classifier_fewshot_directness",
        "directness": "edge_classifier_fewshot_directness",
    }
    resolved = aliases.get(key)
    if resolved is None:
        raise ValueError(
            f"Unknown edge-classifier prompt variant {variant!r}; "
            f"choose one of {list(EDGE_CLASSIFIER_PROMPT_VARIANTS)}"
        )
    return resolved  # type: ignore[return-value]


def edge_classifier_prompt_body(variant: EdgeClassifierPromptVariant | str | None = None) -> str:
    """Static instruction body for hashing (excludes per-request goal/pairs)."""
    resolved = resolve_edge_classifier_prompt_variant(
        variant if isinstance(variant, str) else variant
    )
    if resolved == "edge_classifier_fewshot_directness":
        return f"{_BASELINE_INSTRUCTIONS}\n\n{FEWSHOT_DIRECTNESS_BLOCK}\n\n{_RESPONSE_TAIL}"
    return f"{_BASELINE_INSTRUCTIONS}\n\n{_RESPONSE_TAIL}"


def edge_classifier_prompt_hash(variant: EdgeClassifierPromptVariant | str | None = None) -> str:
    return hashlib.sha256(edge_classifier_prompt_body(variant).encode("utf-8")).hexdigest()[:16]


def edge_classifier_metadata(
    domain: str,
    inventory_version: str,
    *,
    variant: EdgeClassifierPromptVariant | str | None = None,
) -> dict[str, str]:
    resolved = resolve_edge_classifier_prompt_variant(
        variant if isinstance(variant, str) else variant
    )
    body_hash = edge_classifier_prompt_hash(resolved)
    return {
        "generation_strategy": "domain_prior_edge_classifier",
        "prompt_variant": resolved,
        "edge_classifier_prompt_variant": resolved,
        "prompt_version": f"{resolved}@{body_hash}",
        "prompt_hash": body_hash,
        "curriculum_domain": domain,
        "curriculum_inventory_version": inventory_version,
    }


def build_edge_classification_prompt(
    goal_text: str,
    selected: list[SelectedConcept],
    pairs: list[CandidatePair],
    *,
    concept_descriptions: dict[str, str] | None = None,
    variant: EdgeClassifierPromptVariant | str | None = None,
) -> str:
    resolved = resolve_edge_classifier_prompt_variant(
        variant if isinstance(variant, str) else variant
    )
    concept_lines = []
    for s in selected:
        desc = (concept_descriptions or {}).get(s.concept_id, "")
        extra = f" — {desc}" if desc else ""
        concept_lines.append(f"- id={s.concept_id} | title={s.title}{extra}")
    pair_lines = [
        f"- from_id={p.from_id} ({p.from_title}) → to_id={p.to_id} ({p.to_title})"
        for p in pairs
    ]
    fewshot = ""
    if resolved == "edge_classifier_fewshot_directness":
        fewshot = f"\n{FEWSHOT_DIRECTNESS_BLOCK}\n"

    return f"""{_BASELINE_INSTRUCTIONS}
{fewshot}
Learning goal:
---
{goal_text.strip()}
---

Selected concepts:
{chr(10).join(concept_lines)}

Candidate pairs to classify ({len(pairs)}):
{chr(10).join(pair_lines)}

{_RESPONSE_TAIL}
"""
