"""Synapse evaluation path: same ingest prompt JSON as Direct, then real proposal validation.

Does **not** persist a Proposal or touch the user's graph database. Reuses
``build_topics_and_dependencies`` / ``would_create_cycle`` from production code.
"""

from __future__ import annotations

from app.evaluation.failure_analysis import classify_llm_exception
from app.evaluation.schemas import GeneratedGraph
from app.services.proposal_common import build_topics_and_dependencies, review_confidence_threshold


def run_synapse_from_raw(raw: str) -> GeneratedGraph:
    """Apply Synapse's deterministic topic/dependency builder to raw ingest JSON."""
    try:
        from app.services.proposal_common import parse_llm_json_object

        data = parse_llm_json_object(raw)
        raw_topics = data.get("topics")
        raw_deps = data.get("dependencies")
        if not isinstance(raw_topics, list) or not raw_topics:
            raise ValueError("LLM response did not include a non-empty 'topics' list")
        if not isinstance(raw_deps, list):
            raw_deps = []

        proposed_topics, proposed_dependencies, skipped = build_topics_and_dependencies(
            raw_topics,
            raw_deps,
            confidence_threshold=review_confidence_threshold(),
        )
        id_to_title = {t.temp_id: t.title for t in proposed_topics}
        deps: list[tuple[str, str]] = []
        for d in proposed_dependencies:
            frm = id_to_title.get(d.from_temp_id)
            to = id_to_title.get(d.to_temp_id)
            if frm and to:
                deps.append((frm, to))

        return GeneratedGraph(
            topics=[t.title for t in proposed_topics],
            dependencies=deps,
            skipped_dependencies=[
                {"from_title": s.from_title, "to_title": s.to_title, "reason": s.reason}
                for s in skipped
            ],
            topic_confidences=[t.confidence for t in proposed_topics],
            raw_response=raw,
            parse_ok=True,
        )
    except Exception as exc:
        return GeneratedGraph(
            topics=[],
            dependencies=[],
            raw_response=raw,
            parse_ok=False,
            error=str(exc),
            error_category=classify_llm_exception(exc),
        )
