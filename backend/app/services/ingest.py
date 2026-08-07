"""Ingest mode: goal / topic-dump / ingested notes -> a reviewable Proposal of new topics
and dependencies. One of four AI operation modes (ingest/expand/audit/reshape) -- this is
the only one that starts from raw external input rather than an existing part of the graph.

Nothing here writes to the graph -- ``run_ingest`` only builds and persists a ``Proposal``
record. Actually creating topics/dependencies happens in ``apply_proposal``
(``services/proposals.py``), which is the only place a graph mutation can originate from.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.models.proposal import Proposal
from app.prompts.ingest import build_ingest_prompt
from app.services.file_handler import read_raw_note, resolve_raw_note_file
from app.services.llm import call_llm
from app.services.proposal_common import build_topics_and_dependencies, parse_llm_json_object, review_confidence_threshold
from app.services.proposals import save_proposal
from app.services.topics import load_all_topics

logger = logging.getLogger(__name__)


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


async def run_ingest(
    *,
    goal: str | None,
    topics: list[str] | None,
    filenames: list[str] | None,
) -> Proposal:
    """
    Call the LLM for a topic + dependency DAG from the given source(s) and build (and
    persist) a pending Proposal -- no topics or dependencies are written to the graph.
    Every candidate dependency is checked against the DAG-cycle invariant in-memory
    against the other proposed edges; edges that would close a cycle (or reference an
    unknown title) are skipped and reported, never silently dropped. Topics below the
    confidence threshold are flagged ``needs_review`` for the reviewer, not auto-rejected.
    """
    source_text, source_errors, source_label = _build_source_text(goal, topics, filenames)
    if not source_text.strip():
        raise ValueError("Provide at least one of: goal, topics, filenames (with resolvable content)")

    known_titles = sorted({str(r.get("title", "")).strip() for r in load_all_topics() if r.get("title")})
    prompt = build_ingest_prompt(source_text, known_topic_titles=known_titles)
    raw = await call_llm(prompt)
    data = parse_llm_json_object(raw)

    raw_topics = data.get("topics")
    raw_deps = data.get("dependencies")
    if not isinstance(raw_topics, list) or not raw_topics:
        raise ValueError("LLM response did not include a non-empty 'topics' list")
    if not isinstance(raw_deps, list):
        raw_deps = []

    proposed_topics, proposed_dependencies, skipped_dependencies = build_topics_and_dependencies(
        raw_topics,
        raw_deps,
        confidence_threshold=review_confidence_threshold(),
    )

    proposal = Proposal(
        id=uuid.uuid4().hex,
        status="pending",
        mode="ingest",
        source=source_label,
        topics=proposed_topics,
        dependencies=proposed_dependencies,
        skipped_dependencies=skipped_dependencies,
        errors=list(source_errors),
        created_at=datetime.now(timezone.utc),
    )
    save_proposal(proposal)

    logger.info(
        "Ingest proposal %s built: topics=%s dependencies=%s skipped=%s needs_review=%s",
        proposal.id,
        len(proposed_topics),
        len(proposed_dependencies),
        len(skipped_dependencies),
        sum(1 for t in proposed_topics if t.needs_review),
    )
    return proposal
