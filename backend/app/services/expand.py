"""Expand mode: deepen ONE existing topic by proposing new sub-topics/prerequisites
beneath it, without touching the rest of the graph. One of four AI operation modes.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.models.proposal import Proposal
from app.prompts.expand import build_expand_prompt
from app.services.llm import call_llm_detailed, llm_operation
from app.services.operation_context import finalize_generation_meta, synapse_operation
from app.services.proposal_common import build_topics_and_dependencies, parse_llm_json_object, review_confidence_threshold
from app.services.proposal_events import log_proposal_created
from app.services.proposals import save_proposal
from app.services.topics import get_topic_by_id, load_all_topics, load_dependencies

logger = logging.getLogger(__name__)


async def run_expand(*, topic_id: str, instructions: str | None) -> Proposal:
    """
    Propose new sub-topics/prerequisites beneath a single existing topic. The LLM only
    sees that one topic (title, summary) and its current direct prerequisite titles --
    never the rest of the graph -- so the operation stays genuinely scoped rather than
    quietly regenerating everything.
    """
    row = get_topic_by_id(topic_id)
    if row is None:
        raise LookupError(f"No topic with id {topic_id!r}")

    all_deps = load_dependencies()
    existing_prereq_ids = {d["to_topic_id"] for d in all_deps if d["from_topic_id"] == topic_id}
    existing_prereq_titles: list[str] = []
    if existing_prereq_ids:
        title_by_id = {t["id"]: t["title"] for t in load_all_topics()}
        existing_prereq_titles = [title_by_id[i] for i in existing_prereq_ids if i in title_by_id]

    prompt = build_expand_prompt(row["title"], row["summary"], existing_prereq_titles, instructions)

    with synapse_operation():
        with llm_operation("expand"):
            record = await call_llm_detailed(prompt)
        data = parse_llm_json_object(record.text)

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
            extra_title_to_id={row["title"].casefold(): topic_id},
        )

        label = f"expand: {row['title']!r}"
        if instructions and instructions.strip():
            label += f" ({instructions.strip()[:80]!r})"

        meta = finalize_generation_meta({"generation_strategy": "expand"})
        proposal = Proposal(
            id=uuid.uuid4().hex,
            status="pending",
            mode="expand",
            source=label,
            topics=proposed_topics,
            dependencies=proposed_dependencies,
            skipped_dependencies=skipped_dependencies,
            generation_meta=meta,
            created_at=datetime.now(timezone.utc),
        )
        save_proposal(proposal)
        log_proposal_created(proposal)

    logger.info(
        "Expand proposal %s built for topic %s: topics=%s dependencies=%s skipped=%s",
        proposal.id,
        topic_id,
        len(proposed_topics),
        len(proposed_dependencies),
        len(skipped_dependencies),
    )
    return proposal
