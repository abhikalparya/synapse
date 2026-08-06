"""SQLite-backed persistence for review proposals (pending / applied / discarded), plus
the apply/discard lifecycle -- POST /apply is the only path that turns a Proposal into
real Topic/Dependency records.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import select

from app.db.models import ProposalRow
from app.db.session import SessionLocal
from app.models.proposal import ApplyResponse, Proposal, ProposedDependency, ProposedTopic, SkippedProposedDependency
from app.models.topic import Dependency, DependencyCreate, Topic, TopicCreate
from app.services.snapshots import snapshot_graph
from app.services.topics import DependencyCycleError, _add_dependency_in_session, _create_topic_in_session

logger = logging.getLogger(__name__)


def _proposal_row_to_model(row: ProposalRow) -> Proposal:
    return Proposal(
        id=row.id,
        status=row.status,
        source=row.source,
        topics=[ProposedTopic.model_validate(t) for t in row.topics],
        dependencies=[ProposedDependency.model_validate(d) for d in row.dependencies],
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
        row.source = proposal.source
        row.topics = [t.model_dump(mode="json") for t in proposal.topics]
        row.dependencies = [d.model_dump(mode="json") for d in proposal.dependencies]
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
    Commit a pending proposal: snapshot the whole graph first, then persist every proposed
    topic and dependency in a single transaction -- if anything unexpected fails partway
    through, the whole apply rolls back rather than leaving a half-applied graph (the flat-
    JSON version couldn't offer this: each topic/dependency was its own separate file write).
    A proposed dependency that fails its own cycle/uniqueness check is still just skipped and
    reported, exactly as before -- that's an expected per-edge outcome, not a transaction
    failure, so it's caught inside the loop rather than aborting the whole apply.
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

    with SessionLocal() as session, session.begin():
        for pt in proposal.topics:
            row = _create_topic_in_session(session, TopicCreate(title=pt.title, summary=pt.summary))
            temp_to_real[pt.temp_id] = row.id
            created_topics.append(
                Topic(
                    id=row.id,
                    title=row.title,
                    summary=row.summary,
                    status=row.status,
                    resources=[],
                    quiz_passed=row.quiz_passed,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                ),
            )

        for pd in proposal.dependencies:
            from_id = temp_to_real.get(pd.from_temp_id)
            to_id = temp_to_real.get(pd.to_temp_id)
            from_title = title_by_temp_id.get(pd.from_temp_id, pd.from_temp_id)
            to_title = title_by_temp_id.get(pd.to_temp_id, pd.to_temp_id)
            if from_id is None or to_id is None:
                skipped_dependencies.append(
                    SkippedProposedDependency(from_title=from_title, to_title=to_title, reason="topic failed to persist"),
                )
                continue
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

    proposal.status = "applied"
    proposal.applied_at = datetime.now(timezone.utc)
    proposal.snapshot_id = snapshot_id
    save_proposal(proposal)

    logger.info(
        "Applied proposal %s: topics=%s dependencies=%s skipped=%s snapshot=%s",
        proposal_id,
        len(created_topics),
        len(created_dependencies),
        len(skipped_dependencies),
        snapshot_id,
    )
    return ApplyResponse(
        proposal_id=proposal_id,
        snapshot_id=snapshot_id,
        created_topics=created_topics,
        created_dependencies=created_dependencies,
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
