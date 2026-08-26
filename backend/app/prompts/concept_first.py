"""Prompts for the experimental Concept-First generation pipeline.

Stage 1 emits concepts only. Stage 3 emits dependencies only, constrained to a
finalized inventory. Production ingest default remains the baseline joint prompt
in ``app.prompts.ingest``; these prompts are used only when generation strategy
``concept_first`` is selected explicitly.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

CONCEPT_GENERATION_PROMPT = """You are a learning-concept inventory designer. Given the goal/content below, \
list the reusable concepts a learner must understand for this goal.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary before or after).
- Use this exact shape and key names:
  {
    "concepts": [
      {
        "title": "short concept name",
        "description": "optional 1-2 sentence overview",
        "reason": "optional why this concept belongs for the goal"
      },
      ...
    ]
  }
- Each item must be a learnable concept or skill that can participate in prerequisite relationships.
- Prefer titles like: "Variables", "Control Flow", "Functions", "Git", "Linear Algebra".
- Avoid structural/tutorial labels unless they genuinely name a concept: "Module 1", "Lesson 2",
  "Introduction", "Advanced Topics", "Miscellaneous", "Overview", "Getting Started".
- Do NOT invent course-chapter scaffolding just to pad a syllabus.
- Do NOT output dependencies or edges in this step.
- Use 3-10 concepts for a focused goal; more only if the goal is genuinely broad.
- Titles should be short, specific, and non-overlapping."""

DEPENDENCY_GENERATION_PROMPT = """You are a learning-dependency graph designer. Given a FIXED inventory of concepts \
and a learning goal, propose ONLY direct prerequisite relationships between those concepts.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary before or after).
- Use this exact shape and key names:
  {{
    "dependencies": [
      {{"from": "<concept that REQUIRES the other>", "to": "<concept that is the DIRECT PREREQUISITE>"}},
      ...
    ]
  }}
- A dependency means "from" cannot be meaningfully understood without first understanding "to".
- Every "from" and "to" value MUST exactly match one title from the provided concept inventory
  (case-sensitive). Do NOT invent new concept titles.
- Prefer DIRECT prerequisites. Avoid redundant transitive edges when a shorter path already exists.
- Do not create self-loops. The graph must be a DAG.
- Omit an edge when the prerequisite is only loosely related, commonly taught earlier, or optional.

Concept inventory (use these titles exactly):
{inventory_block}

Goal / content:
---
{source}
---"""


def concept_prompt_version() -> str:
    body = CONCEPT_GENERATION_PROMPT + "\n" + DEPENDENCY_GENERATION_PROMPT
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def build_concept_generation_prompt(source_text: str) -> str:
    body = source_text.strip()
    return f"{CONCEPT_GENERATION_PROMPT}\n\nGoal / content:\n---\n{body}\n---"


def build_dependency_generation_prompt(source_text: str, concept_titles: Sequence[str]) -> str:
    lines = "\n".join(f"- {t}" for t in concept_titles)
    inventory_block = lines if lines else "- (empty inventory)"
    return DEPENDENCY_GENERATION_PROMPT.format(
        inventory_block=inventory_block,
        source=source_text.strip(),
    )


def concept_first_prompt_metadata() -> dict[str, str]:
    return {
        "generation_strategy": "concept_first",
        "prompt_variant": "concept_first_staged",
        "prompt_version": f"concept_first_staged@{concept_prompt_version()}",
        "prompt_hash": concept_prompt_version(),
    }
