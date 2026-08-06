"""SQLite-backed persistence for Topics, Resources, and directed prerequisite Dependencies.

Public functions preserve the same signatures and plain-dict/list return shapes as the
original flat-JSON version (Phases 1-6) -- every downstream consumer (routes, roadmap
generation, quiz gating, the MCP bridge) already treats topics/dependencies as
storage-agnostic dicts, so none of that code needed to change for this migration.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DependencyRow, ResourceRow, TopicRow
from app.db.session import SessionLocal
from app.models.topic import Dependency, DependencyCreate, ResourceCreate, Topic, TopicCreate


class DependencyCycleError(ValueError):
    """Raised when adding a dependency would introduce a cycle (including self-loops)."""


def _topic_row_to_dict(row: TopicRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "title": row.title,
        "summary": row.summary,
        "status": row.status,
        "resources": [
            {"id": r.id, "type": r.type, "source_ref": r.source_ref, "title": r.title} for r in row.resources
        ],
        "quiz_passed": row.quiz_passed,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _dependency_row_to_dict(row: DependencyRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "from_topic_id": row.from_topic_id,
        "to_topic_id": row.to_topic_id,
        "created_at": row.created_at,
    }


def load_all_topics() -> list[dict]:
    with SessionLocal() as session:
        rows = session.scalars(select(TopicRow).order_by(TopicRow.created_at)).all()
        return [_topic_row_to_dict(r) for r in rows]


def get_topic_by_id(topic_id: str) -> dict | None:
    with SessionLocal() as session:
        row = session.get(TopicRow, topic_id)
        return _topic_row_to_dict(row) if row else None


def _create_topic_in_session(session: Session, data: TopicCreate) -> TopicRow:
    """Insert a topic within an existing transaction; the caller controls commit/rollback."""
    now = datetime.now(timezone.utc)
    topic = Topic(title=data.title.strip(), summary=data.summary.strip(), status=data.status)
    row = TopicRow(
        id=topic.id,
        title=topic.title,
        summary=topic.summary,
        status=topic.status,
        quiz_passed=False,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    session.flush()
    return row


def save_topic(data: TopicCreate) -> dict:
    """Validate and persist a new topic; returns the stored dict."""
    with SessionLocal() as session, session.begin():
        row = _create_topic_in_session(session, data)
        result = _topic_row_to_dict(row)
    return result


def update_topic(topic_id: str, **patch: object) -> dict | None:
    """Apply plain-column ``patch`` fields (e.g. status, quiz_passed) to an existing topic
    and persist; returns the updated dict, or None if no topic with that id exists."""
    with SessionLocal() as session, session.begin():
        row = session.get(TopicRow, topic_id)
        if row is None:
            return None
        for k, v in patch.items():
            setattr(row, k, v)
        row.updated_at = datetime.now(timezone.utc)
        session.flush()
        # re-validate against the same invariants the pre-SQLite version enforced
        Topic.model_validate(_topic_row_to_dict(row))
        result = _topic_row_to_dict(row)
    return result


def attach_resource(topic_id: str, data: ResourceCreate) -> dict | None:
    """Attach a new resource to an existing topic; returns the updated topic dict, or
    None if no topic with that id exists."""
    with SessionLocal() as session, session.begin():
        topic_row = session.get(TopicRow, topic_id)
        if topic_row is None:
            return None
        resource_row = ResourceRow(
            topic_id=topic_id,
            type=data.type,
            source_ref=data.source_ref.strip(),
            title=data.title.strip(),
        )
        session.add(resource_row)
        topic_row.updated_at = datetime.now(timezone.utc)
        session.flush()
        session.refresh(topic_row)
        result = _topic_row_to_dict(topic_row)
    return result


def load_dependencies() -> list[dict]:
    with SessionLocal() as session:
        rows = session.scalars(select(DependencyRow).order_by(DependencyRow.created_at)).all()
        return [_dependency_row_to_dict(r) for r in rows]


def _build_adjacency(deps: list[dict]) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {}
    for d in deps:
        adj.setdefault(d["from_topic_id"], []).append(d["to_topic_id"])
    return adj


def _reachable(start: str, adj: dict[str, list[str]]) -> set[str]:
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def would_create_cycle(from_topic_id: str, to_topic_id: str, deps: list[dict]) -> bool:
    """True if adding edge ``from_topic_id -> to_topic_id`` closes a cycle in the existing graph."""
    if from_topic_id == to_topic_id:
        return True
    adj = _build_adjacency(deps)
    return from_topic_id in _reachable(to_topic_id, adj)


def _add_dependency_in_session(session: Session, data: DependencyCreate) -> DependencyRow:
    """Validate + insert a dependency within an existing transaction; raises DependencyCycleError
    or ValueError on failure (caller decides whether that aborts the whole transaction or is
    caught and treated as a per-edge skip)."""
    from_id = data.from_topic_id.strip()
    to_id = data.to_topic_id.strip()

    if from_id == to_id:
        raise DependencyCycleError("A topic cannot depend on itself")
    if session.get(TopicRow, from_id) is None:
        raise ValueError(f"Unknown from_topic_id: {from_id}")
    if session.get(TopicRow, to_id) is None:
        raise ValueError(f"Unknown to_topic_id: {to_id}")

    existing_rows = session.scalars(select(DependencyRow)).all()
    deps = [_dependency_row_to_dict(r) for r in existing_rows]

    if any(d["from_topic_id"] == from_id and d["to_topic_id"] == to_id for d in deps):
        raise ValueError("This dependency already exists")
    if would_create_cycle(from_id, to_id, deps):
        raise DependencyCycleError(
            f"Adding this dependency would create a cycle: {to_id!r} already (transitively) requires {from_id!r}",
        )

    now = datetime.now(timezone.utc)
    row = DependencyRow(from_topic_id=from_id, to_topic_id=to_id, created_at=now)
    session.add(row)
    session.flush()
    return row


def add_dependency(data: DependencyCreate) -> dict:
    """Validate endpoints exist and the DAG invariant holds, persist, return the dependency dict."""
    with SessionLocal() as session, session.begin():
        row = _add_dependency_in_session(session, data)
        result = _dependency_row_to_dict(row)
    return result
