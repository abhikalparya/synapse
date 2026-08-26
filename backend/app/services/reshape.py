"""Reshape mode: restructure a SELECTED subgraph (split/merge/reorder), the most invasive
of the four AI operation modes -- always produces a Proposal for review, never applies
directly. Unlike ingest/expand it can propose removing edges, merging topics, and editing
existing summaries, so it builds its Proposal directly rather than through the shared
add-only builder in proposal_common.py.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from app.models.proposal import (
    Proposal,
    ProposedDependency,
    ProposedDependencyRemoval,
    ProposedMerge,
    ProposedTopic,
    ProposedTopicEdit,
    SkippedProposedDependency,
)
from app.prompts.reshape import build_reshape_prompt
from app.services.llm import call_llm, llm_operation
from app.services.proposal_common import parse_llm_json_object, review_confidence_threshold
from app.services.proposal_events import log_proposal_created
from app.services.proposals import save_proposal
from app.services.topics import get_topic_by_id, load_all_topics, load_dependencies, would_create_cycle

logger = logging.getLogger(__name__)


def _resolve_title(title: str, title_to_id: dict[str, str]) -> str | None:
    return title_to_id.get(title.strip().casefold())


def filter_reshape_new_dependencies(
    raw_deps: list[Any],
    *,
    title_to_id: dict[str, str],
    accepted_dep_dicts: list[dict[str, str]],
) -> tuple[list[ProposedDependency], list[SkippedProposedDependency]]:
    """Production reshape edge filter: unknown/out-of-scope titles and cycles are skipped.

    ``accepted_dep_dicts`` is mutated as edges are accepted (same as ``run_reshape``).
    """
    proposed_dependencies: list[ProposedDependency] = []
    skipped_dependencies: list[SkippedProposedDependency] = []
    for row in raw_deps:
        if not isinstance(row, dict):
            continue
        from_title = str(row.get("from", "")).strip()
        to_title = str(row.get("to", "")).strip()
        from_id = _resolve_title(from_title, title_to_id)
        to_id = _resolve_title(to_title, title_to_id)
        if from_id is None or to_id is None:
            skipped_dependencies.append(
                SkippedProposedDependency(
                    from_title=from_title,
                    to_title=to_title,
                    reason="unknown or out-of-scope topic reference",
                ),
            )
            continue
        if would_create_cycle(from_id, to_id, accepted_dep_dicts):
            skipped_dependencies.append(
                SkippedProposedDependency(
                    from_title=from_title,
                    to_title=to_title,
                    reason="would create a cycle",
                ),
            )
            continue
        accepted_dep_dicts.append({"from_topic_id": from_id, "to_topic_id": to_id})
        proposed_dependencies.append(ProposedDependency(from_temp_id=from_id, to_temp_id=to_id))
    return proposed_dependencies, skipped_dependencies


async def run_reshape(*, topic_ids: list[str], instructions: str | None) -> Proposal:
    """
    Propose a restructuring of a selected topic subgraph. The LLM only sees the selected
    topics' own title/summary plus the dependency edges touching them (edges leaving the
    selection are shown as read-only boundary context, not something it can reference by
    title) -- so like expand, this never regenerates or touches the rest of the graph.
    """
    if not topic_ids:
        raise ValueError("Select at least one topic to reshape")

    selected_rows = []
    for tid in topic_ids:
        row = get_topic_by_id(tid)
        if row is None:
            raise ValueError(f"Unknown topic id: {tid}")
        selected_rows.append(row)

    selected_set = set(topic_ids)
    all_deps = load_dependencies()
    title_by_id = {t["id"]: t["title"] for t in load_all_topics()}

    internal_edges = [
        (title_by_id[d["from_topic_id"]], title_by_id[d["to_topic_id"]])
        for d in all_deps
        if d["from_topic_id"] in selected_set and d["to_topic_id"] in selected_set
    ]
    boundary_edges = [
        (title_by_id[d["from_topic_id"]], title_by_id[d["to_topic_id"]])
        for d in all_deps
        if (d["from_topic_id"] in selected_set) != (d["to_topic_id"] in selected_set)
    ]

    prompt = build_reshape_prompt(
        topics=[{"title": r["title"], "summary": r["summary"]} for r in selected_rows],
        internal_edges=internal_edges,
        boundary_edges=boundary_edges,
        instructions=instructions,
    )
    with llm_operation("reshape"):
        raw = await call_llm(prompt)
    data = parse_llm_json_object(raw)

    # Only selected-topic titles resolve -- boundary/outside titles are deliberately never
    # added here, so any attempt to reference one is rejected by the lookups below.
    title_to_id: dict[str, str] = {r["title"].casefold(): r["id"] for r in selected_rows}
    threshold = review_confidence_threshold()

    proposed_topics: list[ProposedTopic] = []
    for row in data.get("new_topics") or []:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        summary = str(row.get("summary", "")).strip()
        try:
            confidence = max(0.0, min(1.0, float(row.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        temp_id = uuid.uuid4().hex
        proposed_topics.append(
            ProposedTopic(temp_id=temp_id, title=title, summary=summary, confidence=confidence, needs_review=confidence <= threshold),
        )
        title_to_id[title.casefold()] = temp_id

    # Seed the in-memory cycle check with the selection's real, current internal edges --
    # unlike ingest/expand, reshape's whole premise is an existing subgraph, so this check
    # can (and should) be fully accurate rather than a best-effort local pre-filter.
    accepted_dep_dicts: list[dict[str, str]] = [
        {"from_topic_id": d["from_topic_id"], "to_topic_id": d["to_topic_id"]}
        for d in all_deps
        if d["from_topic_id"] in selected_set and d["to_topic_id"] in selected_set
    ]

    proposed_dependencies, skipped_dependencies = filter_reshape_new_dependencies(
        list(data.get("new_dependencies") or []),
        title_to_id=title_to_id,
        accepted_dep_dicts=accepted_dep_dicts,
    )

    errors: list[str] = []

    removed_dependencies: list[ProposedDependencyRemoval] = []
    for row in data.get("removed_dependencies") or []:
        if not isinstance(row, dict):
            continue
        from_title = str(row.get("from", "")).strip()
        to_title = str(row.get("to", "")).strip()
        from_id = _resolve_title(from_title, title_to_id)
        to_id = _resolve_title(to_title, title_to_id)
        if from_id is None or to_id is None:
            errors.append(f"removed_dependencies: unresolvable {from_title!r} -> {to_title!r}")
            continue
        removed_dependencies.append(
            ProposedDependencyRemoval(from_topic_id=from_id, to_topic_id=to_id, reason=str(row.get("reason", "")).strip()),
        )

    merges: list[ProposedMerge] = []
    for row in data.get("merges") or []:
        if not isinstance(row, dict):
            continue
        source_title = str(row.get("source", "")).strip()
        target_title = str(row.get("target", "")).strip()
        source_id = _resolve_title(source_title, title_to_id)
        target_id = _resolve_title(target_title, title_to_id)
        if source_id is None or target_id is None or source_id == target_id:
            errors.append(f"merges: unresolvable or invalid {source_title!r} -> {target_title!r}")
            continue
        merges.append(ProposedMerge(source_topic_id=source_id, target_topic_id=target_id, reason=str(row.get("reason", "")).strip()))

    edits: list[ProposedTopicEdit] = []
    for row in data.get("edits") or []:
        if not isinstance(row, dict):
            continue
        topic_title = str(row.get("topic", "")).strip()
        topic_id = _resolve_title(topic_title, title_to_id)
        new_summary = str(row.get("new_summary", "")).strip()
        if topic_id is None or not new_summary:
            errors.append(f"edits: unresolvable topic {topic_title!r} or empty new_summary")
            continue
        edits.append(ProposedTopicEdit(topic_id=topic_id, new_summary=new_summary, reason=str(row.get("reason", "")).strip()))

    if not (proposed_topics or proposed_dependencies or removed_dependencies or merges or edits):
        raise ValueError("LLM did not propose any well-formed restructuring operation for this selection")

    label = f"reshape: {len(topic_ids)} topic(s) selected"
    if instructions and instructions.strip():
        label += f" ({instructions.strip()[:80]!r})"

    proposal = Proposal(
        id=uuid.uuid4().hex,
        status="pending",
        mode="reshape",
        source=label,
        topics=proposed_topics,
        dependencies=proposed_dependencies,
        removed_dependencies=removed_dependencies,
        merges=merges,
        edits=edits,
        skipped_dependencies=skipped_dependencies,
        errors=errors,
        created_at=datetime.now(timezone.utc),
    )
    save_proposal(proposal)
    log_proposal_created(proposal)

    logger.info(
        "Reshape proposal %s built: new_topics=%s new_deps=%s removed=%s merges=%s edits=%s skipped=%s",
        proposal.id,
        len(proposed_topics),
        len(proposed_dependencies),
        len(removed_dependencies),
        len(merges),
        len(edits),
        len(skipped_dependencies),
    )
    return proposal
