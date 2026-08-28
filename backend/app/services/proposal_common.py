"""Shared construction of proposed topics/dependencies from an LLM's
``{"topics": [...], "dependencies": [...]}`` output -- used by both ingest and expand.
(Reshape builds its own Proposal directly: its output shape includes merges/edits/
removals that this shared builder doesn't produce.)
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any

from app.models.proposal import ProposedDependency, ProposedTopic, SkippedProposedDependency
from app.services.topic_identity import canonical_topic_title
from app.services.topics import would_create_cycle


def review_confidence_threshold() -> float:
    """Shared across ingest/expand/reshape -- topics at or below this confidence get
    flagged ``needs_review`` rather than being auto-rejected (Phase 4's review-not-gate
    design)."""
    raw = os.environ.get("ROADMAP_REVIEW_CONFIDENCE_THRESHOLD", "0.6").strip()
    try:
        t = float(raw)
    except ValueError:
        return 0.6
    return max(0.0, min(1.0, t))


def strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_llm_json_object(raw: str) -> dict[str, Any]:
    cleaned = strip_json_fences(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object")
    return data


def build_topics_and_dependencies(
    raw_topics: list[Any],
    raw_deps: list[Any],
    *,
    confidence_threshold: float,
    extra_title_to_id: dict[str, str] | None = None,
    existing_topics: list[dict] | None = None,
) -> tuple[list[ProposedTopic], list[ProposedDependency], list[SkippedProposedDependency]]:
    """
    Turn an LLM's raw topics/dependencies lists into ProposedTopic/ProposedDependency
    records with temp ids, confidence flagging, and an in-memory DAG-cycle check against
    the other edges being proposed in this same call. ``extra_title_to_id`` pre-seeds the
    title->id resolution map with EXISTING (real) topic titles the LLM was told it could
    reference directly (e.g. the anchor topic in an expand call) -- edges naming those
    resolve to the real id rather than minting a temp_id.

    ``existing_topics`` seeds the canonical identity map with live graph topics. A
    generated topic matching an existing identity is resolved to that real id instead
    of becoming a new ProposedTopic.
    """
    proposed_topics: list[ProposedTopic] = []
    title_to_id: dict[str, str] = dict(extra_title_to_id or {})
    canonical_to_id: dict[str, str] = {}

    for title, topic_id in title_to_id.items():
        key = canonical_topic_title(title)
        if key:
            canonical_to_id.setdefault(key, topic_id)

    for topic in existing_topics or []:
        title = str(topic.get("title", "")).strip()
        topic_id = str(topic.get("id", "")).strip()
        key = canonical_topic_title(title)
        if not key or not topic_id:
            continue
        canonical_to_id.setdefault(key, topic_id)
        title_to_id.setdefault(title.casefold(), topic_id)

    for row in raw_topics:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        canonical_title = canonical_topic_title(title)
        if not canonical_title:
            continue
        summary = str(row.get("summary", "")).strip()
        try:
            confidence = float(row.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        existing_id = canonical_to_id.get(canonical_title)
        if existing_id is not None:
            title_to_id[title.casefold()] = existing_id
            continue

        temp_id = uuid.uuid4().hex
        proposed_topics.append(
            ProposedTopic(
                temp_id=temp_id,
                title=title,
                summary=summary,
                confidence=confidence,
                needs_review=confidence <= confidence_threshold,
            ),
        )
        canonical_to_id[canonical_title] = temp_id
        title_to_id[title.casefold()] = temp_id

    proposed_dependencies: list[ProposedDependency] = []
    skipped_dependencies: list[SkippedProposedDependency] = []
    accepted_dep_dicts: list[dict[str, str]] = []

    for row in raw_deps:
        if not isinstance(row, dict):
            continue
        from_title = str(row.get("from", "")).strip()
        to_title = str(row.get("to", "")).strip()
        from_id = title_to_id.get(from_title.casefold()) or canonical_to_id.get(canonical_topic_title(from_title))
        to_id = title_to_id.get(to_title.casefold()) or canonical_to_id.get(canonical_topic_title(to_title))
        if from_id is None or to_id is None:
            skipped_dependencies.append(
                SkippedProposedDependency(from_title=from_title, to_title=to_title, reason="unknown topic reference"),
            )
            continue
        if from_id == to_id:
            skipped_dependencies.append(
                SkippedProposedDependency(
                    from_title=from_title,
                    to_title=to_title,
                    reason="duplicate resolution would create a self-dependency",
                ),
            )
            continue
        if any(
            dependency["from_topic_id"] == from_id and dependency["to_topic_id"] == to_id
            for dependency in accepted_dep_dicts
        ):
            skipped_dependencies.append(
                SkippedProposedDependency(
                    from_title=from_title,
                    to_title=to_title,
                    reason="duplicate dependency after duplicate resolution",
                ),
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

    return proposed_topics, proposed_dependencies, skipped_dependencies
