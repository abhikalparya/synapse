"""Deterministic candidate edge-pair construction for closed-world classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from app.curriculum.selection import SelectedConcept
from app.services.proposal_common import parse_llm_json_object

ALLOWED_DECISIONS = frozenset({"REQUIRED", "NOT_REQUIRED", "UNCERTAIN"})


@dataclass(frozen=True)
class CandidatePair:
    from_id: str
    to_id: str
    from_title: str
    to_title: str

    def key(self) -> tuple[str, str]:
        return (self.from_id, self.to_id)


@dataclass
class EdgeDecision:
    from_id: str
    to_id: str
    decision: str
    reason: str = ""
    from_candidate: bool = True


@dataclass
class ClassificationParseResult:
    decisions: list[EdgeDecision] = field(default_factory=list)
    required_edges: list[tuple[str, str]] = field(default_factory=list)  # titles
    rejected_unknown_ids: list[str] = field(default_factory=list)
    rejected_non_candidate: list[tuple[str, str]] = field(default_factory=list)
    rejected_invalid_decision: list[str] = field(default_factory=list)
    uncertain_count: int = 0
    duplicate_decision_count: int = 0
    pairs_evaluated: int = 0


def generate_candidate_pairs(
    selected: Iterable[SelectedConcept],
    *,
    max_candidate_pairs: int | None = None,
) -> tuple[list[CandidatePair], dict[str, Any]]:
    """All directed pairs among selected concepts, excluding self-loops.

    Ordering is deterministic (sorted by concept_id). Truncation (if any) is reported
    explicitly and never silent.
    """
    concepts = sorted({(s.concept_id, s.title) for s in selected}, key=lambda x: x[0])
    pairs: list[CandidatePair] = []
    for i, (fid, ftitle) in enumerate(concepts):
        for j, (tid, ttitle) in enumerate(concepts):
            if i == j or fid == tid:
                continue
            pairs.append(
                CandidatePair(from_id=fid, to_id=tid, from_title=ftitle, to_title=ttitle)
            )
    total = len(pairs)
    omitted = 0
    if max_candidate_pairs is not None and max_candidate_pairs >= 0 and total > max_candidate_pairs:
        omitted = total - max_candidate_pairs
        pairs = pairs[:max_candidate_pairs]
    meta = {
        "candidate_space_size": total,
        "candidate_pairs_evaluated": len(pairs),
        "candidate_pairs_omitted": omitted,
        "truncated": omitted > 0,
        "selected_concept_count": len(concepts),
    }
    return pairs, meta


def batch_candidate_pairs(
    pairs: list[CandidatePair],
    *,
    pairs_per_batch: int,
) -> list[list[CandidatePair]]:
    """Deterministic contiguous batching."""
    if pairs_per_batch <= 0:
        raise ValueError("pairs_per_batch must be positive")
    return [pairs[i : i + pairs_per_batch] for i in range(0, len(pairs), pairs_per_batch)]


def parse_classification_response(
    raw_text: str,
    candidates: list[CandidatePair],
    *,
    id_to_title: dict[str, str],
) -> ClassificationParseResult:
    """Parse model JSON; keep only valid candidate REQUIRED decisions."""
    result = ClassificationParseResult()
    cand_keys = {p.key(): p for p in candidates}
    result.pairs_evaluated = len(candidates)
    try:
        payload = parse_llm_json_object(raw_text)
    except Exception:
        return result
    rows = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return result

    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        fid = str(row.get("from_id") or row.get("from") or "").strip()
        tid = str(row.get("to_id") or row.get("to") or "").strip()
        decision = str(row.get("decision") or "").strip().upper()
        reason = str(row.get("reason") or "").strip()

        if fid not in id_to_title or tid not in id_to_title:
            if fid and fid not in id_to_title:
                result.rejected_unknown_ids.append(fid)
            if tid and tid not in id_to_title:
                result.rejected_unknown_ids.append(tid)
            continue
        key = (fid, tid)
        if key not in cand_keys:
            result.rejected_non_candidate.append(key)
            continue
        if decision not in ALLOWED_DECISIONS:
            result.rejected_invalid_decision.append(decision or "<missing>")
            continue
        if key in seen:
            result.duplicate_decision_count += 1
            continue
        seen.add(key)
        ed = EdgeDecision(from_id=fid, to_id=tid, decision=decision, reason=reason)
        result.decisions.append(ed)
        if decision == "UNCERTAIN":
            result.uncertain_count += 1
        elif decision == "REQUIRED":
            pair = cand_keys[key]
            result.required_edges.append((pair.from_title, pair.to_title))
    return result


def merge_classification_results(
    parts: list[ClassificationParseResult],
) -> ClassificationParseResult:
    merged = ClassificationParseResult()
    seen_req: set[tuple[str, str]] = set()
    seen_dec: set[tuple[str, str]] = set()
    for part in parts:
        merged.pairs_evaluated += part.pairs_evaluated
        merged.uncertain_count += part.uncertain_count
        merged.duplicate_decision_count += part.duplicate_decision_count
        merged.rejected_unknown_ids.extend(part.rejected_unknown_ids)
        merged.rejected_non_candidate.extend(part.rejected_non_candidate)
        merged.rejected_invalid_decision.extend(part.rejected_invalid_decision)
        for d in part.decisions:
            key = (d.from_id, d.to_id)
            if key in seen_dec:
                merged.duplicate_decision_count += 1
                continue
            seen_dec.add(key)
            merged.decisions.append(d)
        for edge in part.required_edges:
            if edge in seen_req:
                continue
            seen_req.add(edge)
            merged.required_edges.append(edge)
    return merged
