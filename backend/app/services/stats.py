from datetime import datetime, timezone
from typing import Any


def _parse_dt(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, str) and raw.strip():
        try:
            return datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sort_time(topic: dict[str, Any]) -> datetime:
    return _parse_dt(topic.get("created_at")) or datetime.fromtimestamp(0, tz=timezone.utc)


def compute_knowledge_stats(
    topics: list[dict[str, Any]],
    dependencies: list[dict[str, Any]],
) -> dict[str, Any]:
    total_nodes = len(topics)
    total_edges = len(dependencies)

    sorted_topics = sorted(topics, key=_sort_time, reverse=True)
    recent_nodes: list[dict[str, Any]] = []
    for t in sorted_topics[:5]:
        recent_nodes.append(
            {
                "id": str(t.get("id", "")),
                "title": str(t.get("title", "")).strip() or str(t.get("id", "")),
                "status": str(t.get("status", "not_started")),
                "created_at": _iso_utc(_parse_dt(t.get("created_at"))),
                "updated_at": _iso_utc(_parse_dt(t.get("updated_at"))),
            },
        )

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "recent_nodes": recent_nodes,
    }
