#!/usr/bin/env python3
"""
One-time migration: read the pre-SQLite flat-JSON topics/dependencies/quizzes/proposals
store (Phases 1-6) and populate the SQLite database (data/synapse.db) introduced in
Phase 7, preserving every id and timestamp exactly.

Usage (from anywhere -- the backend/ package root is resolved from this file's location):
    python scripts/migrate_json_to_sqlite.py [--force]

Idempotency: refuses to run if the database already has topics or proposals, unless
--force is passed (and even then, it does not deduplicate -- re-running against a
non-empty database will produce duplicate rows).

This does NOT delete the original JSON files under topics/ or proposals/ -- keep them
around until you've confirmed the migrated data looks right, then remove them yourself.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db.models import DependencyRow, ProposalRow, QuizRow, ResourceRow, TopicRow  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402

TOPICS_DIR = _BACKEND_DIR / "topics"
DEPENDENCIES_PATH = TOPICS_DIR / "_dependencies.json"
QUIZZES_DIR = TOPICS_DIR / "_quizzes"
PROPOSALS_DIR = _BACKEND_DIR / "proposals"


def _parse_dt(raw: object) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ! skipping unreadable {path.name}: {exc}")
        return None


def migrate_topics(session: Session) -> set[str]:
    if not TOPICS_DIR.is_dir():
        print("No topics/ directory found -- nothing to migrate for topics.")
        return set()

    migrated = 0
    topic_ids: set[str] = set()
    for path in sorted(TOPICS_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        topic_id = str(data.get("id", "")).strip()
        if not topic_id:
            print(f"  ! skipping {path.name}: no id field")
            continue

        now = datetime.now(timezone.utc)
        session.add(
            TopicRow(
                id=topic_id,
                title=str(data.get("title", "")),
                summary=str(data.get("summary", "")),
                status=str(data.get("status", "not_started")),
                quiz_passed=bool(data.get("quiz_passed", False)),
                created_at=_parse_dt(data.get("created_at")) or now,
                updated_at=_parse_dt(data.get("updated_at")) or now,
            ),
        )
        for r in data.get("resources") or []:
            if not isinstance(r, dict):
                continue
            kwargs = {
                "topic_id": topic_id,
                "type": str(r.get("type", "link")),
                "source_ref": str(r.get("source_ref", "")),
                "title": str(r.get("title", "")),
            }
            rid = str(r.get("id", "")).strip()
            if rid:
                kwargs["id"] = rid
            session.add(ResourceRow(**kwargs))

        topic_ids.add(topic_id)
        migrated += 1

    session.flush()
    print(f"Migrated {migrated} topic(s).")
    return topic_ids


def migrate_dependencies(session: Session, known_topic_ids: set[str]) -> None:
    if not DEPENDENCIES_PATH.is_file():
        print("No _dependencies.json found -- nothing to migrate for dependencies.")
        return
    data = _load_json(DEPENDENCIES_PATH)
    if not isinstance(data, list):
        print("  ! _dependencies.json is not a list, skipping")
        return

    migrated = 0
    skipped = 0
    now = datetime.now(timezone.utc)
    for d in data:
        if not isinstance(d, dict):
            continue
        from_id = str(d.get("from_topic_id", "")).strip()
        to_id = str(d.get("to_topic_id", "")).strip()
        if from_id not in known_topic_ids or to_id not in known_topic_ids:
            print(f"  ! skipping dependency {from_id} -> {to_id}: references an unmigrated topic")
            skipped += 1
            continue
        kwargs = {
            "from_topic_id": from_id,
            "to_topic_id": to_id,
            "created_at": _parse_dt(d.get("created_at")) or now,
        }
        did = str(d.get("id", "")).strip()
        if did:
            kwargs["id"] = did
        session.add(DependencyRow(**kwargs))
        migrated += 1

    session.flush()
    suffix = f", skipped {skipped}" if skipped else ""
    print(f"Migrated {migrated} dependency(ies){suffix}.")


def migrate_quizzes(session: Session, known_topic_ids: set[str]) -> None:
    if not QUIZZES_DIR.is_dir():
        print("No topics/_quizzes/ directory found -- nothing to migrate for quizzes.")
        return

    migrated = 0
    for path in sorted(QUIZZES_DIR.glob("*.json")):
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        topic_id = str(data.get("topic_id", "")).strip()
        if topic_id not in known_topic_ids:
            print(f"  ! skipping quiz {path.name}: topic {topic_id!r} was not migrated")
            continue
        session.add(
            QuizRow(
                topic_id=topic_id,
                questions=data.get("questions") or [],
                created_at=_parse_dt(data.get("created_at")) or datetime.now(timezone.utc),
            ),
        )
        migrated += 1

    session.flush()
    print(f"Migrated {migrated} quiz(zes).")


def migrate_proposals(session: Session) -> None:
    if not PROPOSALS_DIR.is_dir():
        print("No proposals/ directory found -- nothing to migrate for proposals.")
        return

    migrated = 0
    for path in sorted(PROPOSALS_DIR.glob("*.json")):
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        proposal_id = str(data.get("id", "")).strip()
        if not proposal_id:
            print(f"  ! skipping {path.name}: no id field")
            continue
        session.add(
            ProposalRow(
                id=proposal_id,
                status=str(data.get("status", "pending")),
                source=str(data.get("source", "")),
                topics=data.get("topics") or [],
                dependencies=data.get("dependencies") or [],
                skipped_dependencies=data.get("skipped_dependencies") or [],
                errors=data.get("errors") or [],
                created_at=_parse_dt(data.get("created_at")) or datetime.now(timezone.utc),
                applied_at=_parse_dt(data.get("applied_at")),
                snapshot_id=data.get("snapshot_id"),
            ),
        )
        migrated += 1

    session.flush()
    print(f"Migrated {migrated} proposal(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Proceed even if the database already has rows")
    args = parser.parse_args()

    with SessionLocal() as precheck:
        has_topics = precheck.scalar(select(TopicRow.id).limit(1)) is not None
        has_proposals = precheck.scalar(select(ProposalRow.id).limit(1)) is not None
    if (has_topics or has_proposals) and not args.force:
        print(
            "Refusing to migrate: the SQLite database already has data. Re-run with "
            "--force if you really want to migrate on top of it (this will NOT "
            "deduplicate -- you may end up with duplicate rows).",
        )
        raise SystemExit(1)

    with SessionLocal() as session, session.begin():
        topic_ids = migrate_topics(session)
        migrate_dependencies(session, topic_ids)
        migrate_quizzes(session, topic_ids)
        migrate_proposals(session)

    print()
    print("Migration complete. Original JSON files were left untouched under topics/ and")
    print("proposals/ -- remove them yourself once you've confirmed the migrated data looks right.")


if __name__ == "__main__":
    main()
