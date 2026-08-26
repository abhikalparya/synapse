"""Aggregate proposal lifecycle + confidence-calibration metrics from the event log.

Reports only what has actually been recorded. Modification rate is 0 until a reviewer
edit path exists; apply currently always compares as accepted_unchanged when fingerprints match.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from app.services.proposal_events import PROPOSAL_EVENTS_PATH, iter_proposal_events

CONFIDENCE_BUCKETS = (
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0001),
)


def _bucket_label(lo: float, hi: float) -> str:
    if hi >= 1.0:
        return f"{lo:.1f}–1.0"
    return f"{lo:.1f}–{hi:.1f}"


def _bucket_for(confidence: float) -> str:
    for lo, hi in CONFIDENCE_BUCKETS:
        if lo <= confidence < hi:
            return _bucket_label(lo, hi)
    return _bucket_label(*CONFIDENCE_BUCKETS[-1])


def collect_proposal_metrics(path: Path | None = None) -> dict[str, Any]:
    events = iter_proposal_events(path or PROPOSAL_EVENTS_PATH)
    if not events:
        return {
            "available": False,
            "note": "No proposal events recorded yet. Metrics populate as proposals are created/applied/discarded.",
            "counts": {},
            "rates": {},
            "confidence_calibration": {},
            "deterministic_rejections": {},
        }

    created = [e for e in events if e.get("event") == "proposal_created"]
    applied = [e for e in events if e.get("event") == "proposal_applied"]
    discarded = [e for e in events if e.get("event") == "proposal_discarded"]
    modified = [e for e in events if e.get("event") == "proposal_modified"]
    rollbacks = [e for e in events if e.get("event") == "rollback"]
    rejected_edges = [e for e in events if e.get("event") == "deterministically_rejected"]

    decided = len(applied) + len(discarded)
    accepted_unchanged = sum(1 for e in applied if e.get("outcome") == "accepted_unchanged")
    accepted_modified = sum(1 for e in applied if e.get("outcome") == "modified")

    rates = {
        "acceptance_rate": (len(applied) / decided) if decided else None,
        "rejection_rate": (len(discarded) / decided) if decided else None,
        "modification_rate": (accepted_modified / decided) if decided else None,
        "accepted_unchanged_rate": (accepted_unchanged / decided) if decided else None,
    }

    # Average edits before application: count proposal_modified events per applied proposal.
    edits_before_apply: list[int] = []
    for a in applied:
        pid = a.get("proposal_id")
        edits_before_apply.append(sum(1 for m in modified if m.get("proposal_id") == pid))
    avg_edits = (sum(edits_before_apply) / len(edits_before_apply)) if edits_before_apply else None

    det_counts: dict[str, int] = defaultdict(int)
    for e in rejected_edges:
        det_counts[str(e.get("category") or "UNKNOWN")] += 1

    # Confidence calibration: mean confidence at decision time vs outcome.
    bucket_stats: dict[str, dict[str, int]] = {
        _bucket_label(lo, hi): {"accepted_unchanged": 0, "modified": 0, "rejected": 0, "n": 0}
        for lo, hi in CONFIDENCE_BUCKETS
    }
    for e in applied + discarded:
        mean_c = e.get("mean_confidence")
        if mean_c is None:
            continue
        try:
            conf = float(mean_c)
        except (TypeError, ValueError):
            continue
        label = _bucket_for(conf)
        bucket_stats[label]["n"] += 1
        outcome = e.get("outcome")
        if outcome == "accepted_unchanged":
            bucket_stats[label]["accepted_unchanged"] += 1
        elif outcome == "modified":
            bucket_stats[label]["modified"] += 1
        elif outcome == "rejected" or e.get("event") == "proposal_discarded":
            bucket_stats[label]["rejected"] += 1

    calibration: dict[str, Any] = {}
    for label, st in bucket_stats.items():
        n = st["n"]
        if n == 0:
            calibration[label] = {
                "n": 0,
                "acceptance_rate": None,
                "modification_rate": None,
                "rejection_rate": None,
                "note": "insufficient data",
            }
            continue
        calibration[label] = {
            "n": n,
            "acceptance_rate": st["accepted_unchanged"] / n,
            "modification_rate": st["modified"] / n,
            "rejection_rate": st["rejected"] / n,
        }

    return {
        "available": True,
        "note": (
            "Rates use recorded apply/discard events only. "
            "Proposals cannot currently be edited in-app, so modification_rate reflects "
            "fingerprint mismatches (normally 0)."
        ),
        "counts": {
            "proposals_created": len(created),
            "proposals_applied": len(applied),
            "proposals_discarded": len(discarded),
            "proposals_modified_events": len(modified),
            "rollbacks": len(rollbacks),
            "invalid_edges_caught": len(rejected_edges),
            "cycle_causing_edges_rejected": det_counts.get("CYCLE_ATTEMPT", 0),
            "out_of_scope_or_invalid_refs_rejected": det_counts.get("OUT_OF_SCOPE_REFERENCE", 0)
            + det_counts.get("INVALID_TOPIC_REFERENCE", 0),
        },
        "rates": rates,
        "average_edits_before_application": avg_edits,
        "deterministic_rejections_by_category": dict(det_counts),
        "confidence_calibration": calibration,
        "claim": (
            "Do not treat LLM self-reported confidence as calibrated probability "
            "unless bucket outcomes demonstrate correlation."
        ),
    }
