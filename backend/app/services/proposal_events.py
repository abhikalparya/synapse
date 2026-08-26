"""Append-only proposal lifecycle events.

Stored as JSONL under ``backend/data/`` (outside the SQLite file) so ``POST /rollback``
snapshots cannot erase operational evaluation history.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models.proposal import Proposal

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = _BACKEND_DIR / "data"

PROPOSAL_EVENTS_PATH = DATA_DIR / "proposal_events.jsonl"


def events_log_path() -> Path:
    """Event log location. ``SYNAPSE_PROPOSAL_EVENTS_PATH`` isolates evaluation runs."""
    override = (os.environ.get("SYNAPSE_PROPOSAL_EVENTS_PATH") or "").strip()
    if override:
        return Path(override)
    return PROPOSAL_EVENTS_PATH


def classify_skip_reason(reason: str) -> str:
    """Map a skipped-dependency reason string to a stable failure category."""
    r = (reason or "").casefold()
    if "cycle" in r or "depend on itself" in r or "cannot depend on itself" in r:
        return "CYCLE_ATTEMPT"
    if "out-of-scope" in r or "out of scope" in r:
        return "OUT_OF_SCOPE_REFERENCE"
    if "unknown" in r:
        return "INVALID_TOPIC_REFERENCE"
    return "INVALID_TOPIC_REFERENCE"


def proposal_fingerprint(proposal: Proposal) -> str:
    """Stable hash of the reviewable proposal content (topics/deps/removals/merges/edits)."""
    payload = {
        "topics": [(t.title, t.summary, round(t.confidence, 4)) for t in proposal.topics],
        "dependencies": [(d.from_temp_id, d.to_temp_id) for d in proposal.dependencies],
        "removed_dependencies": [(d.from_topic_id, d.to_topic_id) for d in proposal.removed_dependencies],
        "merges": [(m.source_topic_id, m.target_topic_id) for m in proposal.merges],
        "edits": [(e.topic_id, e.new_summary) for e in proposal.edits],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def append_proposal_event(event: dict[str, Any]) -> None:
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    target = events_log_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Failed to append proposal event: %s", exc)


def log_proposal_created(proposal: Proposal) -> None:
    confidences = [t.confidence for t in proposal.topics]
    skipped = [
        {
            "from_title": s.from_title,
            "to_title": s.to_title,
            "reason": s.reason,
            "category": classify_skip_reason(s.reason),
        }
        for s in proposal.skipped_dependencies
    ]
    meta = dict(proposal.generation_meta or {})
    append_proposal_event(
        {
            "event": "proposal_created",
            "stage": "presented_for_review",
            "proposal_id": proposal.id,
            "mode": proposal.mode,
            "source": proposal.source,
            "topic_count": len(proposal.topics),
            "dependency_count": len(proposal.dependencies),
            "topic_confidences": confidences,
            "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
            "fingerprint": proposal_fingerprint(proposal),
            "skipped_dependencies": skipped,
            "invalid_edges_caught": len(skipped),
            "cycle_causing_edges_rejected": sum(1 for s in skipped if s["category"] == "CYCLE_ATTEMPT"),
            "out_of_scope_references_rejected": sum(
                1 for s in skipped if s["category"] in ("OUT_OF_SCOPE_REFERENCE", "INVALID_TOPIC_REFERENCE")
            ),
            "generation_strategy": meta.get("generation_strategy"),
            "domain": meta.get("domain"),
            "inventory_version": meta.get("inventory_version"),
            "inventory_hash": meta.get("inventory_hash"),
            "fallback_reason": meta.get("fallback_reason"),
            "generation_meta": {
                k: meta[k]
                for k in (
                    "generation_strategy",
                    "domain",
                    "inventory_version",
                    "inventory_hash",
                    "selected_concept_count",
                    "fallback_reason",
                    "domain_resolution_status",
                )
                if k in meta
            },
        },
    )
    for s in skipped:
        append_proposal_event(
            {
                "event": "deterministically_rejected",
                "stage": "deterministically_rejected",
                "proposal_id": proposal.id,
                "mode": proposal.mode,
                "category": s["category"],
                "from_title": s["from_title"],
                "to_title": s["to_title"],
                "reason": s["reason"],
            },
        )


def log_proposal_applied(proposal: Proposal, *, fingerprint_at_apply: str | None = None) -> None:
    created_fp = None
    for ev in iter_proposal_events():
        if ev.get("event") == "proposal_created" and ev.get("proposal_id") == proposal.id:
            created_fp = ev.get("fingerprint")
    applied_fp = fingerprint_at_apply or proposal_fingerprint(proposal)
    modified = bool(created_fp and applied_fp and created_fp != applied_fp)
    confidences = [t.confidence for t in proposal.topics]
    append_proposal_event(
        {
            "event": "proposal_applied",
            "stage": "modified" if modified else "accepted_unchanged",
            "proposal_id": proposal.id,
            "mode": proposal.mode,
            "outcome": "modified" if modified else "accepted_unchanged",
            "modified": modified,
            "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
            "topic_confidences": confidences,
            "fingerprint": applied_fp,
            "created_fingerprint": created_fp,
        },
    )


def log_proposal_discarded(proposal: Proposal) -> None:
    confidences = [t.confidence for t in proposal.topics]
    append_proposal_event(
        {
            "event": "proposal_discarded",
            "stage": "rejected",
            "proposal_id": proposal.id,
            "mode": proposal.mode,
            "outcome": "rejected",
            "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
            "topic_confidences": confidences,
            "fingerprint": proposal_fingerprint(proposal),
        },
    )


def log_proposal_modified(proposal: Proposal, *, edits_count: int = 1) -> None:
    """Future hook: call when a reviewer edits a pending proposal before apply."""
    append_proposal_event(
        {
            "event": "proposal_modified",
            "stage": "modified",
            "proposal_id": proposal.id,
            "mode": proposal.mode,
            "edits_count": edits_count,
            "fingerprint": proposal_fingerprint(proposal),
        },
    )


def log_rollback(snapshot_id: str) -> None:
    append_proposal_event(
        {
            "event": "rollback",
            "stage": "rollback",
            "snapshot_id": snapshot_id,
        },
    )


def iter_proposal_events(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or events_log_path()
    if not target.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read proposal events: %s", exc)
        return []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed proposal event at %s:%s", target, line_no)
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events
