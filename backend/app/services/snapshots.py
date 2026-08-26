"""Whole-graph snapshot/restore backing POST /rollback, via SQLite's own online backup API.

An apply can create many topics and edges at once, so "undo" needs to restore the entire
database to a single point in time, not just the last uncommitted write (that's what
apply's own transaction already guarantees, atomically, on its own -- see
services/proposals.py). This is the separate, coarser mechanism for reverting an apply
that already committed: snapshot_graph() copies the live .db file via sqlite3's built-in
backup() (a consistent, live copy, not a raw file-byte copy); restore_snapshot() copies
it back the same way.
"""

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from app.db.models import DependencyRow, TopicRow
from app.db.session import DB_PATH, SessionLocal, engine
from app.services.proposal_events import log_rollback

logger = logging.getLogger(__name__)

# Keep snapshots beside the live DB so SYNAPSE_DB_PATH isolation also isolates rollback files.
SNAPSHOTS_DIR = DB_PATH.parent / "_snapshots"


def _ensure_snapshots_dir() -> None:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def snapshot_graph() -> str:
    """Copy the live database to a timestamped snapshot file; returns the snapshot id."""
    _ensure_snapshots_dir()
    now = datetime.now(timezone.utc)
    snapshot_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    snap_path = SNAPSHOTS_DIR / f"{snapshot_id}.db"

    src = sqlite3.connect(str(DB_PATH))
    try:
        dst = sqlite3.connect(str(snap_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()

    logger.info("Saved graph snapshot %s", snapshot_id)
    return snapshot_id


def _latest_snapshot_id() -> str | None:
    _ensure_snapshots_dir()
    names = sorted(p.stem for p in SNAPSHOTS_DIR.glob("*.db") if p.is_file())
    return names[-1] if names else None


def restore_snapshot(snapshot_id: str | None = None) -> dict[str, Any]:
    """
    Restore the database to exactly the state captured in ``snapshot_id`` (or the most
    recently taken snapshot if omitted). Raises ``LookupError`` if no matching snapshot
    exists.
    """
    _ensure_snapshots_dir()
    resolved_id = snapshot_id or _latest_snapshot_id()
    if resolved_id is None:
        raise LookupError("No snapshots exist to roll back to")

    snap_path = SNAPSHOTS_DIR / f"{Path(resolved_id).name}.db"
    if not snap_path.is_file():
        raise LookupError(f"No snapshot with id {resolved_id!r}")

    # Drop pooled connections first so nothing holds a stale handle to the file we're
    # about to overwrite via a fresh live-backup copy.
    engine.dispose()

    src = sqlite3.connect(str(snap_path))
    try:
        dst = sqlite3.connect(str(DB_PATH))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()

    with SessionLocal() as session:
        n_topics = session.scalar(select(func.count()).select_from(TopicRow)) or 0
        n_deps = session.scalar(select(func.count()).select_from(DependencyRow)) or 0

    logger.info("Restored graph snapshot %s (topics=%s, dependencies=%s)", resolved_id, n_topics, n_deps)
    log_rollback(resolved_id)
    return {
        "snapshot_id": resolved_id,
        "restored_topics": n_topics,
        "restored_dependencies": n_deps,
    }
