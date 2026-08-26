"""Reusable prompts for generating a topic + prerequisite-dependency DAG.

Before-state analysis (baseline / historical ingest prompt)
---------------------------------------------------------
1. Topic naming: only "short topic name", "specific, and non-overlapping". No ban on
   tutorial/module/introduction headings.
2. Dependency creation: "from cannot be understood without to first". No explicit
   distinction between *direct* vs *indirect* prerequisites.
3. Not distinguished: teaching sequence vs conceptual dependency (list order is
   "foundational-first" but edges define structure — models still emit syllabus chains).
4. Not prevented: introductory labels, module names, or redundant transitive edges
   (only "DAG: no cycles" is stated).

The ``concept_direct_prerequisite`` variant adds those constraints. Production default
remains ``baseline`` until an A/B quality run supports switching; pass
``variant="concept_direct_prerequisite"`` (or ``SYNAPSE_INGEST_PROMPT_VARIANT``) to opt in.
"""

from __future__ import annotations

import hashlib
import os
from typing import Literal

PromptVariant = Literal["baseline", "concept_direct_prerequisite"]

PROMPT_VARIANTS: tuple[PromptVariant, ...] = ("baseline", "concept_direct_prerequisite")

# ---------------------------------------------------------------------------
# baseline — exact historical curriculum-design ingest schema (unchanged text)
# ---------------------------------------------------------------------------
INGEST_JSON_SCHEMA = """You are a curriculum-design assistant. Given the goal/content below, break it into a \
small set of discrete topics and the prerequisite relationships between them.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary before or after).
- Use this exact shape and key names:
  {
    "topics": [
      {
        "title": "short topic name",
        "summary": "1-3 sentence overview of this topic",
        "confidence": <number between 0 and 1>
      },
      ...
    ],
    "dependencies": [
      {"from": "<topic title that REQUIRES the other>", "to": "<topic title that is the PREREQUISITE>"},
      ...
    ]
  }
- A dependency entry means "from" cannot be understood without "to" first; "to" is the prerequisite.
- Every "from" and "to" value MUST exactly match a "title" from the "topics" list (case-sensitive).
- The dependencies must form a DAG: no topic may (transitively) depend on itself.
- Use 3-8 topics for a focused goal; more only if the goal is genuinely broad.
- List topics roughly foundational-first, but the "dependencies" edges (not list order) define the real structure.
- Titles should be short, specific, and non-overlapping; summaries factual, with no invented specifics beyond
  what the goal/content implies.
- confidence (your calibration, 0-1, per topic):
  High (0.8-1.0): this topic and its role in the sequence is well-established, standard knowledge.
  Medium (0.4-0.8): reasonable inference from the goal/content, or a topic whose exact placement is debatable.
  Low (0.0-0.4): speculative, a stretch from the given goal/content, or you are largely guessing it belongs."""

# ---------------------------------------------------------------------------
# concept_direct_prerequisite — concept nodes + direct edges only
# ---------------------------------------------------------------------------
INGEST_CONCEPT_DIRECT_PREREQUISITE = """You are a learning-dependency graph designer. Given the goal/content below, \
produce a small prerequisite graph of reusable concepts — NOT a tutorial, course outline, or lesson plan.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary before or after).
- Use this exact shape and key names:
  {
    "topics": [
      {
        "title": "short concept name",
        "summary": "1-3 sentence overview of this concept",
        "confidence": <number between 0 and 1>
      },
      ...
    ],
    "dependencies": [
      {"from": "<concept that REQUIRES the other>", "to": "<concept that is the DIRECT PREREQUISITE>"},
      ...
    ]
  }
- A dependency entry means "from" cannot be meaningfully understood without first understanding "to".
  "to" is the prerequisite. Edge direction must follow that convention.

TOPIC RULES:
- Each topic must be a reusable concept or skill (knowledge that could be a node in a prerequisite graph).
- Prefer titles like: "Variables and Data Types", "Control Flow", "Functions", "Linear Algebra", "Probability".
- Avoid tutorial/syllabus wording: "Introduction to…", "… Basics", "Lesson N", "Module N",
  "Getting Started with…", "Advanced Concepts", "Overview of…", "Chapter…".
- Do not invent topics merely to mimic a course chapter sequence.

DEPENDENCY RULES:
- Create an edge ONLY when the learner must understand the prerequisite *directly* before the dependent concept.
- Ask: "Must a learner understand A before they can meaningfully understand B?" If not a direct need, omit the edge.
- Do NOT add an edge merely because A is commonly taught before B, they share a curriculum, or A helps eventually.
- Prefer DIRECT prerequisites. Avoid redundant transitive edges: if A requires B and B requires C, do NOT also add
  A requires C unless C is independently a direct prerequisite of A.
- Prefer the smallest graph that captures the meaningful conceptual prerequisites for this goal.
- Do not create dense all-to-all edges. Do not pad with optional side topics.

CORE CONCEPT COMPLETENESS:
- Before finalizing, check that major foundational concepts required for the goal are present.
- Do not expand into a full encyclopedia curriculum; balance minimality with conceptual completeness.

STRUCTURAL RULES:
- Every "from" and "to" value MUST exactly match a "title" from the "topics" list (case-sensitive).
- The dependencies must form a DAG: no topic may (transitively) depend on itself.
- Use 3-8 topics for a focused goal; more only if the goal is genuinely broad.
- List topics roughly foundational-first, but the "dependencies" edges (not list order) define the real structure.
- Summaries factual, with no invented specifics beyond what the goal/content implies.
- confidence (your calibration, 0-1, per topic):
  High (0.8-1.0): this concept and its role is well-established, standard knowledge.
  Medium (0.4-0.8): reasonable inference from the goal/content, or exact placement is debatable.
  Low (0.0-0.4): speculative, a stretch from the given goal/content, or you are largely guessing it belongs."""

_VARIANT_BODIES: dict[str, str] = {
    "baseline": INGEST_JSON_SCHEMA,
    "concept_direct_prerequisite": INGEST_CONCEPT_DIRECT_PREREQUISITE,
}


def resolve_prompt_variant(variant: str | None = None) -> PromptVariant:
    """Resolve variant from argument or ``SYNAPSE_INGEST_PROMPT_VARIANT`` (default: baseline)."""
    raw = (variant if variant is not None else os.environ.get("SYNAPSE_INGEST_PROMPT_VARIANT") or "baseline").strip()
    key = raw.casefold().replace("-", "_")
    aliases = {
        "baseline": "baseline",
        "concept_direct_prerequisite": "concept_direct_prerequisite",
        "concept": "concept_direct_prerequisite",
        "concept_direct": "concept_direct_prerequisite",
    }
    resolved = aliases.get(key)
    if resolved is None:
        raise ValueError(f"Unknown ingest prompt variant {variant!r}; choose one of {list(PROMPT_VARIANTS)}")
    return resolved  # type: ignore[return-value]


def ingest_prompt_body(variant: PromptVariant | str | None = None) -> str:
    return _VARIANT_BODIES[resolve_prompt_variant(variant if isinstance(variant, str) else variant)]


def prompt_version_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def prompt_metadata(variant: PromptVariant | str | None = None) -> dict[str, str]:
    resolved = resolve_prompt_variant(variant if isinstance(variant, str) else variant)
    body = _VARIANT_BODIES[resolved]
    return {
        "prompt_variant": resolved,
        "prompt_version": f"{resolved}@{prompt_version_hash(body)}",
        "prompt_hash": prompt_version_hash(body),
    }


def build_ingest_prompt(
    source_text: str,
    known_topic_titles: list[str] | None = None,
    *,
    variant: PromptVariant | str | None = None,
) -> str:
    """Full prompt string passed to the LLM (instruction + goal/content, minus already-known titles)."""
    body_schema = ingest_prompt_body(variant)
    body = source_text.strip()
    titles_block = ""
    if known_topic_titles:
        lines = "\n".join(f"- {t}" for t in known_topic_titles[:250])
        titles_block = f"\n\nExisting topic titles (avoid exact duplicates unless truly the same topic):\n{lines}\n"
    return f"{body_schema}{titles_block}\n\nGoal / content:\n---\n{body}\n---"
