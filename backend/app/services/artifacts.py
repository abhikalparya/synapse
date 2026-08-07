"""SQLite-backed persistence for Artifacts -- outputs a learner produced while studying a
topic (a note, code snippet, summary, or generated output), distinct from Resources
(inputs they studied from). Create + list-by-topic only; no update/delete for now, since
artifacts are meant to be a lightweight, append-only study log.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.db.models import ArtifactRow, TopicRow
from app.db.session import SessionLocal
from app.models.artifact import ArtifactCreate


def _artifact_row_to_dict(row: ArtifactRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "topic_id": row.topic_id,
        "type": row.type,
        "title": row.title,
        "content": row.content,
        "created_at": row.created_at,
    }


def list_artifacts_for_topic(topic_id: str) -> list[dict]:
    with SessionLocal() as session:
        rows = session.scalars(
            select(ArtifactRow).where(ArtifactRow.topic_id == topic_id).order_by(ArtifactRow.created_at),
        ).all()
        return [_artifact_row_to_dict(r) for r in rows]


def create_artifact(topic_id: str, data: ArtifactCreate) -> dict | None:
    """Returns the created artifact dict, or None if no topic with that id exists."""
    with SessionLocal() as session, session.begin():
        if session.get(TopicRow, topic_id) is None:
            return None
        row = ArtifactRow(
            topic_id=topic_id,
            type=data.type,
            title=data.title.strip(),
            content=data.content,
            created_at=datetime.now(timezone.utc),
        )
        session.add(row)
        session.flush()
        result = _artifact_row_to_dict(row)
    return result
