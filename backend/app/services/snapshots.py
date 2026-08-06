"""Whole-graph snapshot/restore for Topics + Dependencies, backing POST /rollback.

Unlike the old per-page version snapshots in ``services/rewrite.py``, an apply can create
many topics and edges at once, so rollback needs to restore the *entire* topics/ directory
(every topic file's exact prior bytes, plus the dependency edge list) to a single point in time.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.topics import DEPENDENCIES_PATH, TOPICS_DIR

logger = logging.getLogger(__name__)

SNAPSHOTS_DIR = TOPICS_DIR / "_snapshots"


def _ensure_snapshots_dir() -> None:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def _topic_files() -> dict[str, dict[str, Any]]:
    """Every topic file's raw parsed JSON, keyed by filename (excludes ``_``-prefixed files)."""
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(TOPICS_DIR.glob("*.json")):
        if not path.is_file() or path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable topic file in snapshot: %s (%s)", path.name, exc)
            continue
        if isinstance(data, dict):
            out[path.name] = data
    return out


def _load_dependencies_raw() -> list[dict[str, Any]]:
    if not DEPENDENCIES_PATH.is_file():
        return []
    try:
        data = json.loads(DEPENDENCIES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def snapshot_graph() -> str:
    """Capture every topic file + the dependency edge list as they exist right now; returns the snapshot id."""
    _ensure_snapshots_dir()
    now = datetime.now(timezone.utc)
    snapshot_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    payload = {
        "id": snapshot_id,
        "created_at": now.isoformat(),
        "topic_files": _topic_files(),
        "dependencies": _load_dependencies_raw(),
    }
    out_path = SNAPSHOTS_DIR / f"{snapshot_id}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(
        "Saved graph snapshot %s (topics=%s, dependencies=%s)",
        snapshot_id,
        len(payload["topic_files"]),
        len(payload["dependencies"]),
    )
    return snapshot_id


def _latest_snapshot_id() -> str | None:
    _ensure_snapshots_dir()
    names = sorted(p.stem for p in SNAPSHOTS_DIR.glob("*.json") if p.is_file())
    return names[-1] if names else None


def restore_snapshot(snapshot_id: str | None = None) -> dict[str, Any]:
    """
    Restore topics/ + dependencies to exactly the state captured in ``snapshot_id``
    (or the most recently taken snapshot if omitted). Raises ``LookupError`` if no
    matching snapshot exists.
    """
    _ensure_snapshots_dir()
    resolved_id = snapshot_id or _latest_snapshot_id()
    if resolved_id is None:
        raise LookupError("No snapshots exist to roll back to")

    snap_path = SNAPSHOTS_DIR / f"{resolved_id}.json"
    if not snap_path.is_file():
        raise LookupError(f"No snapshot with id {resolved_id!r}")

    try:
        payload = json.loads(snap_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LookupError(f"Snapshot {resolved_id!r} is corrupt: {exc}") from exc

    topic_files: dict[str, Any] = payload.get("topic_files") or {}
    dependencies: list[Any] = payload.get("dependencies") or []

    for path in TOPICS_DIR.glob("*.json"):
        if path.is_file() and not path.name.startswith("_"):
            path.unlink()

    for name, data in topic_files.items():
        safe_name = Path(name).name
        (TOPICS_DIR / safe_name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    DEPENDENCIES_PATH.write_text(
        json.dumps(dependencies, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Restored graph snapshot %s (topics=%s, dependencies=%s)",
        resolved_id,
        len(topic_files),
        len(dependencies),
    )
    return {
        "snapshot_id": resolved_id,
        "restored_topics": len(topic_files),
        "restored_dependencies": len(dependencies),
    }
