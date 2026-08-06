"""Generative roadmap creation: goal / topic-dump / ingested notes -> a reviewable Proposal.

Nothing here writes to the topics/ store -- ``generate_roadmap`` only builds and persists a
``Proposal`` record. Actually creating topics/dependencies happens in ``apply_proposal``
(see ``routes/proposals.py``), which is the only place a graph mutation can originate from.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from app.models.proposal import Proposal, ProposedDependency, ProposedTopic, SkippedProposedDependency
from app.prompts.roadmap import build_roadmap_generation_prompt
from app.services.file_handler import read_raw_note, resolve_raw_note_file
from app.services.llm import call_llm
from app.services.proposals import save_proposal
from app.services.topics import load_all_topics, would_create_cycle

logger = logging.getLogger(__name__)


def _review_confidence_threshold() -> float:
    raw = os.environ.get("ROADMAP_REVIEW_CONFIDENCE_THRESHOLD", "0.6").strip()
    try:
        t = float(raw)
    except ValueError:
        return 0.6
    return max(0.0, min(1.0, t))


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _build_source_text(
    goal: str | None,
    topics: list[str] | None,
    filenames: list[str] | None,
) -> tuple[str, list[str], str]:
    """Combine goal / topic-dump / raw-note text into one prompt source; returns (text, errors, source_label)."""
    parts: list[str] = []
    errors: list[str] = []
    label_parts: list[str] = []

    if goal and goal.strip():
        parts.append(f"Goal: {goal.strip()}")
        label_parts.append(f"goal: {goal.strip()[:80]!r}")

    if topics:
        dump = "\n".join(f"- {t.strip()}" for t in topics if t.strip())
        if dump:
            parts.append(f"Topic dump:\n{dump}")
            label_parts.append(f"{len([t for t in topics if t.strip()])} topic dump entries")

    resolved_notes = 0
    for name in filenames or []:
        path = resolve_raw_note_file(name)
        if path is None:
            errors.append(f"raw note not found: {name}")
            continue
        try:
            text = read_raw_note(path)
        except OSError as exc:
            errors.append(f"failed to read {name}: {exc}")
            continue
        if text.strip():
            parts.append(f"Note ({name}):\n{text.strip()}")
            resolved_notes += 1
    if resolved_notes:
        label_parts.append(f"{resolved_notes} note(s)")

    return "\n\n".join(parts), errors, ", ".join(label_parts) or "unspecified source"


def _parse_roadmap_json(raw: str) -> dict[str, Any]:
    cleaned = _strip_json_fences(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object")
    return data


async def generate_roadmap(
    *,
    goal: str | None,
    topics: list[str] | None,
    filenames: list[str] | None,
) -> Proposal:
    """
    Call the LLM for a topic + dependency DAG from the given source(s) and build (and persist)
    a pending Proposal -- no topics or dependencies are written to the graph. Every candidate
    dependency is checked against the same DAG-cycle invariant from Phase 1, in-memory against
    the other proposed edges; edges that would close a cycle (or reference an unknown title)
    are skipped and reported, never silently dropped. Topics below the confidence threshold
    are flagged ``needs_review`` for the reviewer, not auto-rejected.
    """
    source_text, source_errors, source_label = _build_source_text(goal, topics, filenames)
    if not source_text.strip():
        raise ValueError("Provide at least one of: goal, topics, filenames (with resolvable content)")

    known_titles = sorted({str(r.get("title", "")).strip() for r in load_all_topics() if r.get("title")})
    prompt = build_roadmap_generation_prompt(source_text, known_topic_titles=known_titles)
    raw = await call_llm(prompt)
    data = _parse_roadmap_json(raw)

    raw_topics = data.get("topics")
    raw_deps = data.get("dependencies")
    if not isinstance(raw_topics, list) or not raw_topics:
        raise ValueError("LLM response did not include a non-empty 'topics' list")
    if not isinstance(raw_deps, list):
        raw_deps = []

    threshold = _review_confidence_threshold()
    proposed_topics: list[ProposedTopic] = []
    title_to_temp_id: dict[str, str] = {}
    errors: list[str] = list(source_errors)

    for row in raw_topics:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        summary = str(row.get("summary", "")).strip()
        try:
            confidence = float(row.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        temp_id = uuid.uuid4().hex
        proposed_topics.append(
            ProposedTopic(
                temp_id=temp_id,
                title=title,
                summary=summary,
                confidence=confidence,
                needs_review=confidence <= threshold,
            ),
        )
        title_to_temp_id[title.casefold()] = temp_id

    proposed_dependencies: list[ProposedDependency] = []
    skipped_dependencies: list[SkippedProposedDependency] = []
    accepted_dep_dicts: list[dict[str, str]] = []

    for row in raw_deps:
        if not isinstance(row, dict):
            continue
        from_title = str(row.get("from", "")).strip()
        to_title = str(row.get("to", "")).strip()
        from_id = title_to_temp_id.get(from_title.casefold())
        to_id = title_to_temp_id.get(to_title.casefold())
        if from_id is None or to_id is None:
            skipped_dependencies.append(
                SkippedProposedDependency(from_title=from_title, to_title=to_title, reason="unknown topic reference"),
            )
            continue
        if would_create_cycle(from_id, to_id, accepted_dep_dicts):
            skipped_dependencies.append(
                SkippedProposedDependency(
                    from_title=from_title,
                    to_title=to_title,
                    reason="would create a cycle with other proposed dependencies",
                ),
            )
            continue
        accepted_dep_dicts.append({"from_topic_id": from_id, "to_topic_id": to_id})
        proposed_dependencies.append(ProposedDependency(from_temp_id=from_id, to_temp_id=to_id))

    proposal = Proposal(
        id=uuid.uuid4().hex,
        status="pending",
        source=source_label,
        topics=proposed_topics,
        dependencies=proposed_dependencies,
        skipped_dependencies=skipped_dependencies,
        errors=errors,
        created_at=datetime.now(timezone.utc),
    )
    save_proposal(proposal)

    logger.info(
        "Roadmap proposal %s built: topics=%s dependencies=%s skipped=%s needs_review=%s",
        proposal.id,
        len(proposed_topics),
        len(proposed_dependencies),
        len(skipped_dependencies),
        sum(1 for t in proposed_topics if t.needs_review),
    )
    return proposal
