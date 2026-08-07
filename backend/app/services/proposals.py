"""SQLite-backed persistence for review proposals (pending / applied / discarded), plus
the apply/discard lifecycle -- POST /apply is the only path that turns a Proposal into
real Topic/Dependency records.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.db.models import ProposalRow, ResourceRow
from app.db.session import SessionLocal
from app.models.proposal import (
    ApplyResponse,
    Proposal,
    ProposedDependency,
    ProposedDependencyRemoval,
    ProposedMerge,
    ProposedTopic,
    ProposedTopicEdit,
    SkippedProposedDependency,
)
from app.models.topic import Dependency, DependencyCreate, Resource, Topic, TopicCreate
from app.services.snapshots import snapshot_graph
from app.services.topics import (
    DependencyCycleError,
    _add_dependency_in_session,
    _create_topic_in_session,
    _edit_topic_in_session,
    _merge_topics_in_session,
    _remove_dependency_in_session,
)

logger = logging.getLogger(__name__)


def _proposal_row_to_model(row: ProposalRow) -> Proposal:
    return Proposal(
        id=row.id,
        status=row.status,
        mode=row.mode,
        source=row.source,
        topics=[ProposedTopic.model_validate(t) for t in row.topics],
        dependencies=[ProposedDependency.model_validate(d) for d in row.dependencies],
        removed_dependencies=[ProposedDependencyRemoval.model_validate(d) for d in row.removed_dependencies],
        merges=[ProposedMerge.model_validate(m) for m in row.merges],
        edits=[ProposedTopicEdit.model_validate(e) for e in row.edits],
        skipped_dependencies=[SkippedProposedDependency.model_validate(s) for s in row.skipped_dependencies],
        errors=list(row.errors or []),
        created_at=row.created_at,
        applied_at=row.applied_at,
        snapshot_id=row.snapshot_id,
    )


def save_proposal(proposal: Proposal) -> None:
    with SessionLocal() as session, session.begin():
        row = session.get(ProposalRow, proposal.id)
        if row is None:
            row = ProposalRow(id=proposal.id)
            session.add(row)
        row.status = proposal.status
        row.mode = proposal.mode
        row.source = proposal.source
        row.topics = [t.model_dump(mode="json") for t in proposal.topics]
        row.dependencies = [d.model_dump(mode="json") for d in proposal.dependencies]
        row.removed_dependencies = [d.model_dump(mode="json") for d in proposal.removed_dependencies]
        row.merges = [m.model_dump(mode="json") for m in proposal.merges]
        row.edits = [e.model_dump(mode="json") for e in proposal.edits]
        row.skipped_dependencies = [s.model_dump(mode="json") for s in proposal.skipped_dependencies]
        row.errors = list(proposal.errors)
        row.created_at = proposal.created_at or datetime.now(timezone.utc)
        row.applied_at = proposal.applied_at
        row.snapshot_id = proposal.snapshot_id


def load_proposal(proposal_id: str) -> Proposal | None:
    with SessionLocal() as session:
        row = session.get(ProposalRow, proposal_id)
        if row is None:
            return None
        try:
            return _proposal_row_to_model(row)
        except ValueError as exc:
            logger.warning("Proposal %s failed validation: %s", proposal_id, exc)
            return None


def list_proposals(*, status: str | None = None) -> list[Proposal]:
    with SessionLocal() as session:
        stmt = select(ProposalRow).order_by(ProposalRow.created_at.desc())
        if status is not None:
            stmt = stmt.where(ProposalRow.status == status)
        rows = session.scalars(stmt).all()
        out: list[Proposal] = []
        for row in rows:
            try:
                out.append(_proposal_row_to_model(row))
            except ValueError as exc:
                logger.warning("Skipping unreadable proposal %s: %s", row.id, exc)
        return out


def apply_proposal(proposal_id: str) -> ApplyResponse:
    """
    Commit a pending proposal: snapshot the whole graph first, then apply every proposed
    operation -- new topics, new dependencies, dependency removals, topic edits, and
    merges, in that order -- inside a single transaction. If anything unexpected fails
    partway through, the whole apply rolls back rather than leaving a half-applied graph.

    A proposed *dependency add* that fails its own cycle/uniqueness check is still just
    skipped and reported (an expected per-edge outcome, caught inside the loop) --
    removals, edits, and merges have no equivalent "expected failure" path, since they
    only ever reference topics the proposal's own author (ingest/expand/reshape) already
    confirmed exist at proposal-build time; if one fails here it's a real inconsistency
    (e.g. the topic was deleted by another apply since), and the whole apply aborts.
    """
    proposal = load_proposal(proposal_id)
    if proposal is None:
        raise LookupError(f"No proposal with id {proposal_id!r}")
    if proposal.status != "pending":
        raise ValueError(f"Proposal {proposal_id!r} is already {proposal.status}, not pending")

    snapshot_id = snapshot_graph()

    temp_to_real: dict[str, str] = {}
    created_topics: list[Topic] = []
    created_dependencies: list[Dependency] = []
    skipped_dependencies: list[SkippedProposedDependency] = list(proposal.skipped_dependencies)
    title_by_temp_id = {pt.temp_id: pt.title for pt in proposal.topics}
    removed_count = 0
    edited_count = 0
    merged_count = 0

    with SessionLocal() as session, session.begin():
        for pt in proposal.topics:
            row = _create_topic_in_session(session, TopicCreate(title=pt.title, summary=pt.summary))
            temp_to_real[pt.temp_id] = row.id
            resources: list[Resource] = []
            if pt.source_note_path:
                # Phase 11 vault import: trace the created topic back to its source note.
                resource_row = ResourceRow(
                    topic_id=row.id,
                    type="note",
                    source_ref=pt.source_note_path,
                    title=Path(pt.source_note_path).stem,
                )
                session.add(resource_row)
                session.flush()
                resources.append(
                    Resource(
                        id=resource_row.id,
                        type=resource_row.type,
                        source_ref=resource_row.source_ref,
                        title=resource_row.title,
                    ),
                )
            created_topics.append(
                Topic(
                    id=row.id,
                    title=row.title,
                    summary=row.summary,
                    status=row.status,
                    resources=resources,
                    quiz_passed=row.quiz_passed,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                ),
            )

        for pd in proposal.dependencies:
            # Either side may be a temp_id from this same proposal (new topic, just
            # created above) or -- for expand/reshape -- the real id of a topic that
            # already existed before this proposal; fall back to using it as-is.
            from_id = temp_to_real.get(pd.from_temp_id, pd.from_temp_id)
            to_id = temp_to_real.get(pd.to_temp_id, pd.to_temp_id)
            from_title = title_by_temp_id.get(pd.from_temp_id, pd.from_temp_id)
            to_title = title_by_temp_id.get(pd.to_temp_id, pd.to_temp_id)
            try:
                dep_row = _add_dependency_in_session(
                    session,
                    DependencyCreate(from_topic_id=from_id, to_topic_id=to_id),
                )
            except (DependencyCycleError, ValueError) as exc:
                skipped_dependencies.append(
                    SkippedProposedDependency(from_title=from_title, to_title=to_title, reason=str(exc)),
                )
                continue
            created_dependencies.append(
                Dependency(
                    id=dep_row.id,
                    from_topic_id=dep_row.from_topic_id,
                    to_topic_id=dep_row.to_topic_id,
                    created_at=dep_row.created_at,
                ),
            )

        for rd in proposal.removed_dependencies:
            if _remove_dependency_in_session(session, rd.from_topic_id, rd.to_topic_id):
                removed_count += 1

        for ed in proposal.edits:
            _edit_topic_in_session(session, ed.topic_id, ed.new_summary)
            edited_count += 1

        for m in proposal.merges:
            _merge_topics_in_session(session, m.source_topic_id, m.target_topic_id)
            merged_count += 1

    proposal.status = "applied"
    proposal.applied_at = datetime.now(timezone.utc)
    proposal.snapshot_id = snapshot_id
    save_proposal(proposal)

    logger.info(
        "Applied proposal %s (mode=%s): topics=%s dependencies=%s removed=%s edited=%s merged=%s skipped=%s snapshot=%s",
        proposal_id,
        proposal.mode,
        len(created_topics),
        len(created_dependencies),
        removed_count,
        edited_count,
        merged_count,
        len(skipped_dependencies),
        snapshot_id,
    )
    return ApplyResponse(
        proposal_id=proposal_id,
        snapshot_id=snapshot_id,
        created_topics=created_topics,
        created_dependencies=created_dependencies,
        removed_dependency_count=removed_count,
        merged_topic_count=merged_count,
        edited_topic_count=edited_count,
        skipped_dependencies=skipped_dependencies,
    )


def discard_proposal(proposal_id: str) -> Proposal:
    proposal = load_proposal(proposal_id)
    if proposal is None:
        raise LookupError(f"No proposal with id {proposal_id!r}")
    if proposal.status != "pending":
        raise ValueError(f"Proposal {proposal_id!r} is already {proposal.status}, not pending")
    proposal.status = "discarded"
    save_proposal(proposal)
    logger.info("Discarded proposal %s", proposal_id)
    return proposal
