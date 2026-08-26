"""Ingest mode: goal / topic-dump / ingested notes -> a reviewable Proposal of new topics
and dependencies. Among AI operation modes (ingest/expand/audit/reshape), this is
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
from app.services.generation_strategy import resolve_runtime_generation_strategy
from app.services.llm import call_llm_detailed, llm_operation
from app.services.operation_context import finalize_generation_meta, synapse_operation
from app.services.proposal_common import build_topics_and_dependencies, parse_llm_json_object, review_confidence_threshold
from app.services.proposal_events import log_proposal_created
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
    generation_strategy: str | None = None,
    curriculum_domain: str | None = None,
    require_domain_prior: bool = False,
) -> Proposal:
    """
    Call the LLM for a topic + dependency DAG from the given source(s) and build (and
    persist) a pending Proposal -- no topics or dependencies are written to the graph.

    Product strategies:
    - ``baseline`` (default / production)
    - ``domain_curriculum_prior`` (opt-in experimental)
    - ``domain_prior_edge_classifier`` (experimental only)

    Closed experiments (Concept-First, coverage recovery) are evaluation-only and are
    not routed through this product path.
    """
    source_text, source_errors, source_label = _build_source_text(goal, topics, filenames)
    if not source_text.strip():
        raise ValueError("Provide at least one of: goal, topics, filenames (with resolvable content)")

    with synapse_operation():
        strategy = resolve_runtime_generation_strategy(generation_strategy)
        if strategy == "domain_curriculum_prior":
            return await _run_ingest_domain_curriculum_prior(
                source_text=source_text,
                source_errors=source_errors,
                source_label=source_label,
                curriculum_domain=curriculum_domain,
                require_domain_prior=require_domain_prior,
            )
        if strategy == "domain_prior_edge_classifier":
            return await _run_ingest_domain_prior_edge_classifier(
                source_text=source_text,
                source_errors=source_errors,
                source_label=source_label,
                curriculum_domain=curriculum_domain,
                require_domain_prior=require_domain_prior,
            )

        return await _run_ingest_baseline(
            source_text=source_text,
            source_errors=source_errors,
            source_label=source_label,
            generation_meta={"generation_strategy": "baseline"},
        )


async def _run_ingest_baseline(
    *,
    source_text: str,
    source_errors: list[str],
    source_label: str,
    generation_meta: dict | None = None,
) -> Proposal:
    known_titles = sorted({str(r.get("title", "")).strip() for r in load_all_topics() if r.get("title")})
    prompt = build_ingest_prompt(source_text, known_topic_titles=known_titles)
    with llm_operation("ingest"):
        record = await call_llm_detailed(prompt)
    raw = record.text
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

    meta = finalize_generation_meta(generation_meta or {"generation_strategy": "baseline"})
    proposal = Proposal(
        id=uuid.uuid4().hex,
        status="pending",
        mode="ingest",
        source=source_label,
        topics=proposed_topics,
        dependencies=proposed_dependencies,
        skipped_dependencies=skipped_dependencies,
        errors=list(source_errors),
        generation_meta=meta,
        created_at=datetime.now(timezone.utc),
    )
    save_proposal(proposal)
    log_proposal_created(proposal)

    logger.info(
        "Ingest proposal %s built: strategy=%s topics=%s dependencies=%s skipped=%s needs_review=%s",
        proposal.id,
        meta.get("generation_strategy"),
        len(proposed_topics),
        len(proposed_dependencies),
        len(skipped_dependencies),
        sum(1 for t in proposed_topics if t.needs_review),
    )
    return proposal


async def _run_ingest_domain_curriculum_prior(
    *,
    source_text: str,
    source_errors: list[str],
    source_label: str,
    curriculum_domain: str | None = None,
    require_domain_prior: bool = False,
) -> Proposal:
    import os

    from app.curriculum.resolution import resolve_domain, resolution_to_meta
    from app.services.domain_curriculum_prior import run_domain_curriculum_prior_pipeline

    env_domain = (os.environ.get("SYNAPSE_CURRICULUM_DOMAIN") or "").strip()
    resolution = resolve_domain(
        domain_override=curriculum_domain or env_domain or None,
        require_inventory=True,
        on_unresolved="error" if require_domain_prior else "baseline",
        on_unavailable="error" if require_domain_prior else "baseline",
    )
    if not resolution.ok:
        reason = resolution.fallback_reason or resolution.status
        if require_domain_prior or resolution.fallback_action == "error":
            raise ValueError(
                f"{reason}: domain_curriculum_prior requires a resolvable domain "
                "with a frozen inventory (set curriculum_domain / SYNAPSE_CURRICULUM_DOMAIN)"
            )
        logger.info(
            "Domain curriculum prior unavailable (%s); falling back to baseline",
            reason,
        )
        return await _run_ingest_baseline(
            source_text=source_text,
            source_errors=source_errors,
            source_label=source_label,
            generation_meta={
                "generation_strategy": "baseline",
                "fallback_reason": reason,
                **resolution_to_meta(resolution),
                "requested_strategy": "domain_curriculum_prior",
            },
        )

    domain = resolution.domain or ""
    result = await run_domain_curriculum_prior_pipeline(source_text, domain=domain)
    if not result.parse_ok or not result.topics:
        detail = "; ".join(result.errors) or "domain curriculum prior failed"
        raise ValueError(f"Domain curriculum prior ingest failed: {detail}")
    if result.new_concept_count:
        raise ValueError(
            f"Domain curriculum prior invented concepts rejected: new_concept_count="
            f"{result.new_concept_count}"
        )

    raw_topics = [
        {
            "title": t["title"],
            "summary": t.get("summary", ""),
            "confidence": t.get("confidence", 0.7),
        }
        for t in result.topics
    ]
    raw_deps = list(result.dependencies)
    proposed_topics, proposed_dependencies, skipped_dependencies = build_topics_and_dependencies(
        raw_topics,
        raw_deps,
        confidence_threshold=review_confidence_threshold(),
    )
    if result.skipped_dependencies and not skipped_dependencies:
        skipped_dependencies = list(result.skipped_dependencies)

    errors = list(source_errors)
    errors.extend(result.errors)
    meta = finalize_generation_meta(
        {**result.to_meta(), **resolution_to_meta(resolution)},
        generation_strategy="domain_curriculum_prior",
    )
    proposal = Proposal(
        id=uuid.uuid4().hex,
        status="pending",
        mode="ingest",
        source=(
            f"{source_label} [generation_strategy=domain_curriculum_prior "
            f"domain={domain} inventory_version={result.inventory_version}]"
        ),
        topics=proposed_topics,
        dependencies=proposed_dependencies,
        skipped_dependencies=skipped_dependencies,
        errors=errors,
        generation_meta=meta,
        created_at=datetime.now(timezone.utc),
    )
    save_proposal(proposal)
    log_proposal_created(proposal)
    logger.info(
        "Domain-curriculum-prior ingest proposal %s built: domain=%s topics=%s deps=%s",
        proposal.id,
        domain,
        len(proposed_topics),
        len(proposed_dependencies),
    )
    return proposal


async def _run_ingest_domain_prior_edge_classifier(
    *,
    source_text: str,
    source_errors: list[str],
    source_label: str,
    curriculum_domain: str | None = None,
    require_domain_prior: bool = False,
) -> Proposal:
    import os

    from app.curriculum.resolution import resolve_domain, resolution_to_meta
    from app.services.domain_prior_edge_classifier import (
        run_domain_prior_edge_classifier_pipeline,
    )

    env_domain = (os.environ.get("SYNAPSE_CURRICULUM_DOMAIN") or "").strip()
    resolution = resolve_domain(
        domain_override=curriculum_domain or env_domain or None,
        require_inventory=True,
        on_unresolved="error" if require_domain_prior else "baseline",
        on_unavailable="error" if require_domain_prior else "baseline",
    )
    if not resolution.ok:
        reason = resolution.fallback_reason or resolution.status
        if require_domain_prior or resolution.fallback_action == "error":
            raise ValueError(
                f"{reason}: domain_prior_edge_classifier requires a resolvable domain "
                "with a frozen inventory (set curriculum_domain / SYNAPSE_CURRICULUM_DOMAIN)"
            )
        logger.info(
            "Domain prior edge classifier unavailable (%s); falling back to baseline",
            reason,
        )
        return await _run_ingest_baseline(
            source_text=source_text,
            source_errors=source_errors,
            source_label=source_label,
            generation_meta={
                "generation_strategy": "baseline",
                "fallback_reason": reason,
                **resolution_to_meta(resolution),
                "requested_strategy": "domain_prior_edge_classifier",
            },
        )

    domain = resolution.domain or ""
    result = await run_domain_prior_edge_classifier_pipeline(source_text, domain=domain)
    if not result.parse_ok or not result.topics:
        detail = "; ".join(result.errors) or "domain prior edge classifier failed"
        raise ValueError(f"Domain prior edge classifier ingest failed: {detail}")
    if result.new_concept_count:
        raise ValueError(
            f"Domain prior edge classifier invented concepts rejected: new_concept_count="
            f"{result.new_concept_count}"
        )

    raw_topics = [
        {
            "title": t["title"],
            "summary": t.get("summary", ""),
            "confidence": t.get("confidence", 0.7),
        }
        for t in result.topics
    ]
    raw_deps = list(result.dependencies)
    proposed_topics, proposed_dependencies, skipped_dependencies = build_topics_and_dependencies(
        raw_topics,
        raw_deps,
        confidence_threshold=review_confidence_threshold(),
    )
    if result.skipped_dependencies and not skipped_dependencies:
        skipped_dependencies = list(result.skipped_dependencies)

    errors = list(source_errors)
    errors.extend(result.errors)
    meta = finalize_generation_meta(
        {**result.to_meta(), **resolution_to_meta(resolution)},
        generation_strategy="domain_prior_edge_classifier",
    )
    proposal = Proposal(
        id=uuid.uuid4().hex,
        status="pending",
        mode="ingest",
        source=(
            f"{source_label} [generation_strategy=domain_prior_edge_classifier "
            f"domain={domain}]"
        ),
        topics=proposed_topics,
        dependencies=proposed_dependencies,
        skipped_dependencies=skipped_dependencies,
        errors=errors,
        generation_meta=meta,
        created_at=datetime.now(timezone.utc),
    )
    save_proposal(proposal)
    log_proposal_created(proposal)
    logger.info(
        "Domain-prior-edge-classifier ingest proposal %s built: domain=%s topics=%s deps=%s",
        proposal.id,
        domain,
        len(proposed_topics),
        len(proposed_dependencies),
    )
    return proposal
