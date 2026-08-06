"""Flat-JSON persistence for review proposals (pending / applied / discarded), plus the
apply/discard lifecycle -- POST /apply is the only path that turns a Proposal into real
Topic/Dependency records.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.models.proposal import ApplyResponse, Proposal, SkippedProposedDependency
from app.models.topic import Dependency, DependencyCreate, Topic, TopicCreate
from app.services.snapshots import snapshot_graph
from app.services.topics import DependencyCycleError, add_dependency, save_topic

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROPOSALS_DIR = _PROJECT_ROOT / "proposals"


def _ensure_proposals_dir() -> None:
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)


def _path_for(proposal_id: str) -> Path:
    return PROPOSALS_DIR / f"{proposal_id}.json"


def save_proposal(proposal: Proposal) -> None:
    _ensure_proposals_dir()
    _path_for(proposal.id).write_text(
        json.dumps(proposal.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_proposal(proposal_id: str) -> Proposal | None:
    path = _path_for(proposal_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load proposal %s: %s", proposal_id, exc)
        return None
    try:
        return Proposal.model_validate(data)
    except ValueError as exc:
        logger.warning("Proposal %s failed validation: %s", proposal_id, exc)
        return None


def list_proposals(*, status: str | None = None) -> list[Proposal]:
    _ensure_proposals_dir()
    out: list[Proposal] = []
    for path in sorted(PROPOSALS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            proposal = Proposal.model_validate(data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Skipping unreadable proposal %s: %s", path.name, exc)
            continue
        if status is None or proposal.status == status:
            out.append(proposal)
    out.sort(key=lambda p: p.created_at.isoformat() if p.created_at else p.id, reverse=True)
    return out


def apply_proposal(proposal_id: str) -> ApplyResponse:
    """
    Commit a pending proposal: snapshot the whole graph first, then persist each proposed
    topic (assigning real ids) and, for each proposed dependency, resolve temp ids to real
    ones and run it back through the Phase 1 cycle check -- the in-memory check at proposal
    time is not re-litigated here, but disk state may have moved on since generation, so this
    is the real, final gate before anything lands.
    """
    proposal = load_proposal(proposal_id)
    if proposal is None:
        raise LookupError(f"No proposal with id {proposal_id!r}")
    if proposal.status != "pending":
        raise ValueError(f"Proposal {proposal_id!r} is already {proposal.status}, not pending")

    snapshot_id = snapshot_graph()

    temp_to_real: dict[str, str] = {}
    created_topics: list[Topic] = []
    for pt in proposal.topics:
        stored = save_topic(TopicCreate(title=pt.title, summary=pt.summary))
        topic = Topic.model_validate({k: v for k, v in stored.items() if k != "path"})
        temp_to_real[pt.temp_id] = topic.id
        created_topics.append(topic)

    created_dependencies: list[Dependency] = []
    skipped_dependencies: list[SkippedProposedDependency] = list(proposal.skipped_dependencies)
    title_by_temp_id = {pt.temp_id: pt.title for pt in proposal.topics}

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
            payload = add_dependency(DependencyCreate(from_topic_id=from_id, to_topic_id=to_id))
        except (DependencyCycleError, ValueError) as exc:
            skipped_dependencies.append(
                SkippedProposedDependency(from_title=from_title, to_title=to_title, reason=str(exc)),
            )
            continue
        created_dependencies.append(Dependency.model_validate(payload))

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
