"""SQLite-backed persistence for Zones -- visual/logical grouping regions. A topic
belongs to at most one zone at a time (TopicRow.zone_id); assignment/unassignment goes
through PATCH /topics/{id}, not a zone-side endpoint (see routes/topics.py).
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.db.models import TopicRow, ZoneRow
from app.db.session import SessionLocal


def _zone_row_to_dict(row: ZoneRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "label": row.label,
        "color": row.color,
        "created_at": row.created_at,
    }


def load_all_zones() -> list[dict]:
    with SessionLocal() as session:
        rows = session.scalars(select(ZoneRow).order_by(ZoneRow.created_at)).all()
        return [_zone_row_to_dict(r) for r in rows]


def get_zone_by_id(zone_id: str) -> dict | None:
    with SessionLocal() as session:
        row = session.get(ZoneRow, zone_id)
        return _zone_row_to_dict(row) if row else None


def create_zone(label: str, color: str | None) -> dict:
    with SessionLocal() as session, session.begin():
        row = ZoneRow(label=label.strip(), color=color, created_at=datetime.now(timezone.utc))
        session.add(row)
        session.flush()
        result = _zone_row_to_dict(row)
    return result


def update_zone(zone_id: str, *, label: str | None = None, color_set: bool = False, color: str | None = None) -> dict | None:
    """``color_set`` distinguishes "don't touch color" from "set color to None" -- same
    ambiguity as TopicUpdate.zone_id, resolved by the route via model_fields_set."""
    with SessionLocal() as session, session.begin():
        row = session.get(ZoneRow, zone_id)
        if row is None:
            return None
        if label is not None:
            row.label = label.strip()
        if color_set:
            row.color = color
        session.flush()
        result = _zone_row_to_dict(row)
    return result


def delete_zone(zone_id: str) -> bool:
    """Deletes the zone and unassigns (not deletes) every topic currently in it."""
    with SessionLocal() as session, session.begin():
        row = session.get(ZoneRow, zone_id)
        if row is None:
            return False
        members = session.scalars(select(TopicRow).where(TopicRow.zone_id == zone_id)).all()
        for topic in members:
            topic.zone_id = None
        session.delete(row)
    return True
