"""Flat-JSON persistence for Topics and directed prerequisite Dependencies."""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.models.topic import Dependency, DependencyCreate, Topic, TopicCreate

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TOPICS_DIR = _PROJECT_ROOT / "topics"
DEPENDENCIES_PATH = TOPICS_DIR / "_dependencies.json"


class DependencyCycleError(ValueError):
    """Raised when adding a dependency would introduce a cycle (including self-loops)."""


def _ensure_topics_dir() -> None:
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)


def _slugify_title(title: str) -> str:
    base = (title or "untitled").lower().strip()
    base = re.sub(r"[^\w\s-]", "", base, flags=re.UNICODE)
    base = re.sub(r"[-\s]+", "-", base).strip("-")
    return base[:80] if base else "untitled"


def load_all_topics() -> list[dict]:
    """Load every topic ``*.json`` under topics/ (excludes the ``_dependencies.json`` edge list)."""
    _ensure_topics_dir()
    records: list[dict] = []
    for path in sorted(TOPICS_DIR.glob("*.json")):
        if not path.is_file() or path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping topic %s: %s", path.name, exc)
            continue
        if not isinstance(data, dict):
            logger.warning("Skipping non-object topic JSON: %s", path.name)
            continue
        row = dict(data)
        row["path"] = path
        records.append(row)
    return records


def get_topic_by_id(topic_id: str) -> dict | None:
    for row in load_all_topics():
        if row.get("id") == topic_id:
            return row
    return None


def update_topic(topic_id: str, **patch: object) -> dict | None:
    """Apply ``patch`` fields to an existing topic and persist in place; returns the updated
    dict (with ``path``), or None if no topic with that id exists."""
    row = get_topic_by_id(topic_id)
    if row is None:
        return None
    path = row["path"]
    body = {k: v for k, v in row.items() if k != "path"}
    body.update(patch)
    body["updated_at"] = datetime.now(timezone.utc)
    topic = Topic.model_validate(body)
    payload = topic.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out = dict(payload)
    out["path"] = path
    return out


def save_topic(data: TopicCreate) -> dict:
    """Validate and persist a new topic; returns the stored dict (with a ``path`` key)."""
    _ensure_topics_dir()
    now = datetime.now(timezone.utc)
    topic = Topic(
        title=data.title.strip(),
        summary=data.summary.strip(),
        status=data.status,
        created_at=now,
        updated_at=now,
    )
    payload = topic.model_dump(mode="json")
    stamp = now.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    slug = _slugify_title(topic.title)
    path = TOPICS_DIR / f"{slug}_{stamp}_{suffix}.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Created topic: %s (%s)", topic.title, path.name)
    out = dict(payload)
    out["path"] = path
    return out


def load_dependencies() -> list[dict]:
    _ensure_topics_dir()
    if not DEPENDENCIES_PATH.is_file():
        return []
    try:
        data = json.loads(DEPENDENCIES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load dependencies: %s", exc)
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _save_dependencies(deps: list[dict]) -> None:
    _ensure_topics_dir()
    DEPENDENCIES_PATH.write_text(
        json.dumps(deps, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


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


def add_dependency(data: DependencyCreate) -> dict:
    """Validate endpoints exist and the DAG invariant holds, persist, return the dependency dict."""
    from_id = data.from_topic_id.strip()
    to_id = data.to_topic_id.strip()

    if from_id == to_id:
        raise DependencyCycleError("A topic cannot depend on itself")
    if get_topic_by_id(from_id) is None:
        raise ValueError(f"Unknown from_topic_id: {from_id}")
    if get_topic_by_id(to_id) is None:
        raise ValueError(f"Unknown to_topic_id: {to_id}")

    deps = load_dependencies()
    if any(d.get("from_topic_id") == from_id and d.get("to_topic_id") == to_id for d in deps):
        raise ValueError("This dependency already exists")
    if would_create_cycle(from_id, to_id, deps):
        raise DependencyCycleError(
            f"Adding this dependency would create a cycle: {to_id!r} already (transitively) requires {from_id!r}",
        )

    now = datetime.now(timezone.utc)
    dependency = Dependency(from_topic_id=from_id, to_topic_id=to_id, created_at=now)
    payload = dependency.model_dump(mode="json")
    deps.append(payload)
    _save_dependencies(deps)
    logger.info("Created dependency: %s requires %s", from_id, to_id)
    return payload
