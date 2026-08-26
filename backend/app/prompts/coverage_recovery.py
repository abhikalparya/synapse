"""Prompt for experimental targeted coverage recovery (not a full graph redesign)."""

from __future__ import annotations

COVERAGE_RECOVERY_JSON_SCHEMA = """You are reviewing an EXISTING prerequisite learning graph.

The graph may be structurally valid but conceptually incomplete.

You are NOT being asked to redesign the graph.
You are NOT being asked to add optional curriculum material.
You are NOT being asked to produce a full replacement topic list.
You are NOT being asked to invent a larger syllabus.

You must identify ONLY prerequisite concepts that are genuinely necessary to understand
one or more EXISTING graph topics in the context of the stated learning objective.

Classify each idea into exactly one category:

1. REQUIRED_MISSING_PREREQUISITE — necessary background for understanding at least one
   existing topic toward the objective; without it the graph has a material gap.
2. OPTIONAL_NICE_TO_HAVE — enrichment that is not required.
3. RELATED_BUT_NOT_REQUIRED — related technology or theme that should not be added.
4. OUT_OF_SCOPE — outside the learning objective.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary).
- Prefer a SMALL number of high-value missing prerequisites over many plausible additions.
- Prefer one concept that unlocks multiple relationships over many loosely related ones.
- Do not propose tutorial/module labels, broad curriculum headings, or generic filler.
- Do not duplicate concepts already present (including near-duplicates).
- Every REQUIRED_MISSING_PREREQUISITE must name at least one existing topic that depends on it
  and propose at least one dependency edge using Synapse semantics:
  [from, to] means "from requires to" (to is the prerequisite).
- "from" must be an existing topic title from the graph (or the new candidate only when
  proposing edges among new candidates — prefer attaching to existing topics).
- "to" is normally the new candidate title (the missing prerequisite).

Exact JSON shape:
{
  "candidates": [
    {
      "category": "REQUIRED_MISSING_PREREQUISITE" | "OPTIONAL_NICE_TO_HAVE" | "RELATED_BUT_NOT_REQUIRED" | "OUT_OF_SCOPE",
      "title": "Concept Title",
      "summary": "One short sentence describing the concept.",
      "reason": "Why this is required / why it is not.",
      "target_topics": ["Existing Topic That Depends On It"],
      "relationships": [{"from": "Existing Topic", "to": "Concept Title"}],
      "confidence": 0.0
    }
  ]
}

An empty "candidates" list is a valid and preferred outcome when the graph is already
sufficiently complete for the objective."""


def build_coverage_recovery_prompt(
    *,
    learning_objective: str,
    topics: list[dict[str, str]],
    edges: list[tuple[str, str]],
) -> str:
    topic_lines = "\n".join(
        f"- {t['title']}: {t.get('summary') or '(no summary)'}" for t in topics
    )
    parts = [
        COVERAGE_RECOVERY_JSON_SCHEMA,
        f"Learning objective:\n{learning_objective.strip()}",
        f"Current topics:\n{topic_lines or '(none)'}",
    ]
    if edges:
        edge_lines = "\n".join(f"- {a} requires {b}" for a, b in edges)
        parts.append(f"Current dependency edges:\n{edge_lines}")
    else:
        parts.append("There are currently no dependency edges in this graph.")
    return "\n\n".join(parts)
