import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.tags import normalize_tags_list


def _parse_created_value(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=timezone.utc)
        return raw.astimezone(timezone.utc)
    if isinstance(raw, str) and raw.strip():
        try:
            return datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _fallback_created(path: Path) -> datetime:
    m = re.search(r"_(\d{8})_(\d{6})_", path.name)
    if m:
        d, t = m.groups()
        return datetime(
            int(d[:4]),
            int(d[4:6]),
            int(d[6:8]),
            int(t[:2]),
            int(t[2:4]),
            int(t[4:6]),
            tzinfo=timezone.utc,
        )
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _sort_time(page: dict[str, Any]) -> datetime:
    path = page["path"]
    assert isinstance(path, Path)
    return _parse_created_value(page.get("created_at")) or _fallback_created(path)


def _iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def compute_knowledge_stats(pages: list[dict[str, Any]]) -> dict[str, Any]:
    total_nodes = len(pages)
    total_edges = sum(len(p.get("related_topics") or []) for p in pages)

    tag_counter: Counter[str] = Counter()
    for p in pages:
        for t in normalize_tags_list(p.get("tags")):
            tag_counter[t] += 1

    top_tags = [{"tag": tag, "count": n} for tag, n in tag_counter.most_common(50)]

    sorted_pages = sorted(pages, key=_sort_time, reverse=True)
    recent_nodes: list[dict[str, Any]] = []
    for p in sorted_pages[:5]:
        path = p["path"]
        assert isinstance(path, Path)
        created = _parse_created_value(p.get("created_at")) or _fallback_created(path)
        updated = _parse_created_value(p.get("updated_at"))
        tags = normalize_tags_list(p.get("tags"))
        recent_nodes.append(
            {
                "title": str(p.get("title", "")).strip() or path.stem,
                "filename": path.name,
                "created_at": _iso_utc(created),
                "updated_at": _iso_utc(updated),
                "tags": tags,
            }
        )

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "recent_nodes": recent_nodes,
        "top_tags": top_tags,
    }
