"""Targeted missing-prerequisite coverage recovery (EVALUATION-ONLY / closed experiment).

Runs AFTER baseline joint generation in historical evaluation adapters. Does not redesign
the graph, does not use gold data, and does not write the live graph. Product ingest no
longer routes this strategy — use evaluation systems for reproducibility only.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.models.proposal import SkippedProposedDependency
from app.prompts.coverage_recovery import build_coverage_recovery_prompt
from app.services.llm import call_llm_detailed, llm_operation
from app.services.proposal_common import parse_llm_json_object
from app.services.topics import would_create_cycle

CandidateCategory = Literal[
    "REQUIRED_MISSING_PREREQUISITE",
    "OPTIONAL_NICE_TO_HAVE",
    "RELATED_BUT_NOT_REQUIRED",
    "OUT_OF_SCOPE",
    "UNKNOWN",
]

REQUIRED_CATEGORY = "REQUIRED_MISSING_PREREQUISITE"
REJECT_CATEGORIES = frozenset(
    {"OPTIONAL_NICE_TO_HAVE", "RELATED_BUT_NOT_REQUIRED", "OUT_OF_SCOPE", "UNKNOWN"}
)

DEFAULT_MAX_RECOVERY_CANDIDATES = 5


def max_recovery_candidates() -> int:
    raw = os.environ.get("SYNAPSE_MAX_RECOVERY_CANDIDATES", str(DEFAULT_MAX_RECOVERY_CANDIDATES)).strip()
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_MAX_RECOVERY_CANDIDATES
    return max(0, min(20, n))


def _norm_title(title: str) -> str:
    s = title.casefold().strip()
    s = re.sub(r"[^\w\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass
class CoverageCandidate:
    category: str
    title: str
    summary: str = ""
    reason: str = ""
    target_topics: list[str] = field(default_factory=list)
    relationships: list[tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.5
    operation: str = "NEW_TOPIC_AND_EDGES"  # or NEW_EDGE_ONLY / NEW_TOPIC_ONLY
    rejection_reason: str | None = None
    accepted: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["relationships"] = [list(e) for e in self.relationships]
        return d


@dataclass
class CoverageRecoveryResult:
    """Outcome of one coverage-recovery pass (at most one pass per request)."""

    recovery_enabled: bool = True
    parse_ok: bool = True
    error: str | None = None
    raw_response: str | None = None
    all_candidates: list[CoverageCandidate] = field(default_factory=list)
    retained_candidates: list[CoverageCandidate] = field(default_factory=list)
    accepted_candidates: list[CoverageCandidate] = field(default_factory=list)
    rejected_candidates: list[CoverageCandidate] = field(default_factory=list)
    topics_after: list[dict[str, Any]] = field(default_factory=list)
    dependencies_after: list[dict[str, str]] = field(default_factory=list)
    skipped_dependencies: list[SkippedProposedDependency] = field(default_factory=list)
    new_topic_titles: list[str] = field(default_factory=list)
    new_edges: list[tuple[str, str]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    llm_latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    tokens_estimated: bool = True
    cost_usd: float | None = None
    max_candidates: int = DEFAULT_MAX_RECOVERY_CANDIDATES

    def to_meta(self) -> dict[str, Any]:
        return {
            "generation_strategy": "baseline_coverage_recovery",
            "recovery_enabled": self.recovery_enabled,
            "recovery_parse_ok": self.parse_ok,
            "recovery_error": self.error,
            "recovery_max_candidates": self.max_candidates,
            "recovery_candidate_count": self.counts.get("candidate_count", 0),
            "recovery_applied_count": self.counts.get("applied_count", 0),
            "recovery_rejected_count": self.counts.get("rejected_count", 0),
            "recovery_duplicate_count": self.counts.get("duplicate_count", 0),
            "recovery_cycle_rejected_count": self.counts.get("cycle_rejected_count", 0),
            "recovery_out_of_scope_count": self.counts.get("out_of_scope_count", 0),
            "recovery_optional_rejected_count": self.counts.get("optional_rejected_count", 0),
            "recovery_related_rejected_count": self.counts.get("related_rejected_count", 0),
            "recovery_invalid_target_count": self.counts.get("invalid_target_count", 0),
            "recovery_truncated_count": self.counts.get("truncated_count", 0),
            "recovery_new_topics": list(self.new_topic_titles),
            "recovery_new_edges": [list(e) for e in self.new_edges],
            "recovery_all_candidates": [c.to_dict() for c in self.all_candidates],
            "recovery_accepted": [c.to_dict() for c in self.accepted_candidates],
            "recovery_rejected": [c.to_dict() for c in self.rejected_candidates],
            "recovery_llm_latency_ms": self.llm_latency_ms,
            "recovery_input_tokens": self.input_tokens,
            "recovery_output_tokens": self.output_tokens,
            "recovery_tokens_estimated": self.tokens_estimated,
            "recovery_cost_usd": self.cost_usd,
        }


def parse_coverage_candidates(data: dict[str, Any]) -> list[CoverageCandidate]:
    raw = data.get("candidates")
    if not isinstance(raw, list):
        return []
    out: list[CoverageCandidate] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        cat = str(row.get("category") or "UNKNOWN").strip().upper().replace("-", "_").replace(" ", "_")
        # Normalize common variants
        aliases = {
            "REQUIRED": REQUIRED_CATEGORY,
            "REQUIRED_MISSING": REQUIRED_CATEGORY,
            "REQUIRED_PREREQUISITE": REQUIRED_CATEGORY,
            "OPTIONAL": "OPTIONAL_NICE_TO_HAVE",
            "OPTIONAL_NICE_TO_HAVE": "OPTIONAL_NICE_TO_HAVE",
            "RELATED": "RELATED_BUT_NOT_REQUIRED",
            "RELATED_BUT_NOT_REQUIRED": "RELATED_BUT_NOT_REQUIRED",
            "OUT_OF_SCOPE": "OUT_OF_SCOPE",
            "OUTOFSCOPE": "OUT_OF_SCOPE",
        }
        category = aliases.get(cat, cat if cat in REJECT_CATEGORIES or cat == REQUIRED_CATEGORY else "UNKNOWN")
        try:
            confidence = float(row.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        targets = [str(t).strip() for t in (row.get("target_topics") or []) if str(t).strip()]
        rels: list[tuple[str, str]] = []
        for r in row.get("relationships") or []:
            if isinstance(r, dict):
                frm = str(r.get("from") or "").strip()
                to = str(r.get("to") or "").strip()
                if frm and to:
                    rels.append((frm, to))
            elif isinstance(r, (list, tuple)) and len(r) == 2:
                frm, to = str(r[0]).strip(), str(r[1]).strip()
                if frm and to:
                    rels.append((frm, to))
        # Default relationships: each target requires the new title
        if not rels and targets:
            rels = [(t, title) for t in targets]
        out.append(
            CoverageCandidate(
                category=category,
                title=title,
                summary=str(row.get("summary") or "").strip(),
                reason=str(row.get("reason") or "").strip(),
                target_topics=targets,
                relationships=rels,
                confidence=confidence,
            )
        )
    return out


def rank_candidates(candidates: list[CoverageCandidate]) -> list[CoverageCandidate]:
    """Higher confidence + more dependent targets first (no gold)."""

    def key(c: CoverageCandidate) -> tuple[float, int, str]:
        return (c.confidence, len(c.target_topics) + len(c.relationships), c.title.casefold())

    return sorted(candidates, key=key, reverse=True)


def truncate_candidates(
    candidates: list[CoverageCandidate],
    *,
    max_n: int | None = None,
) -> tuple[list[CoverageCandidate], list[CoverageCandidate]]:
    limit = max_recovery_candidates() if max_n is None else max_n
    required = [c for c in candidates if c.category == REQUIRED_CATEGORY]
    ranked = rank_candidates(required)
    retained = ranked[:limit]
    truncated = ranked[limit:]
    for c in truncated:
        c.rejection_reason = "truncated_by_max_candidates"
        c.accepted = False
    return retained, truncated


def _existing_title_set(topics: list[dict[str, Any]]) -> dict[str, str]:
    """norm -> canonical title."""
    out: dict[str, str] = {}
    for t in topics:
        title = str(t.get("title") or "").strip()
        if title:
            out[_norm_title(title)] = title
    return out


def validate_and_merge_recovery(
    *,
    baseline_topics: list[dict[str, Any]],
    baseline_dependencies: list[dict[str, str]],
    retained: list[CoverageCandidate],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    list[CoverageCandidate],
    list[CoverageCandidate],
    list[SkippedProposedDependency],
    dict[str, int],
]:
    """Deterministically validate retained candidates against the baseline graph.

    Returns topics_after, deps_after, accepted, rejected, skipped, counts.
    """
    topics = [
        {
            "title": str(t.get("title") or "").strip(),
            "summary": str(t.get("summary") or "").strip(),
            "confidence": float(t.get("confidence", 0.7) or 0.7),
        }
        for t in baseline_topics
        if str(t.get("title") or "").strip()
    ]
    deps = [
        {"from": str(d.get("from") or "").strip(), "to": str(d.get("to") or "").strip()}
        for d in baseline_dependencies
        if str(d.get("from") or "").strip() and str(d.get("to") or "").strip()
    ]
    title_map = _existing_title_set(topics)
    # Stable temp ids for cycle check (proposal-local)
    title_to_id = {t["title"]: f"t{i}" for i, t in enumerate(topics)}
    accepted_dep_dicts = [
        {"from_topic_id": title_to_id[d["from"]], "to_topic_id": title_to_id[d["to"]]}
        for d in deps
        if d["from"] in title_to_id and d["to"] in title_to_id
    ]
    existing_edge_norms = {
        (_norm_title(d["from"]), _norm_title(d["to"])) for d in deps
    }

    accepted: list[CoverageCandidate] = []
    rejected: list[CoverageCandidate] = []
    skipped: list[SkippedProposedDependency] = []
    counts = {
        "candidate_count": 0,
        "applied_count": 0,
        "rejected_count": 0,
        "duplicate_count": 0,
        "cycle_rejected_count": 0,
        "out_of_scope_count": 0,
        "optional_rejected_count": 0,
        "related_rejected_count": 0,
        "invalid_target_count": 0,
        "truncated_count": 0,
        "empty_title_count": 0,
        "no_relationship_count": 0,
    }

    for cand in retained:
        counts["candidate_count"] += 1
        if cand.category != REQUIRED_CATEGORY:
            if cand.category == "OPTIONAL_NICE_TO_HAVE":
                counts["optional_rejected_count"] += 1
                cand.rejection_reason = "optional_nice_to_have"
            elif cand.category == "RELATED_BUT_NOT_REQUIRED":
                counts["related_rejected_count"] += 1
                cand.rejection_reason = "related_but_not_required"
            elif cand.category == "OUT_OF_SCOPE":
                counts["out_of_scope_count"] += 1
                cand.rejection_reason = "out_of_scope"
            else:
                cand.rejection_reason = "unknown_category"
            cand.accepted = False
            rejected.append(cand)
            counts["rejected_count"] += 1
            continue

        title = cand.title.strip()
        if not title:
            cand.rejection_reason = "empty_title"
            counts["empty_title_count"] += 1
            cand.accepted = False
            rejected.append(cand)
            counts["rejected_count"] += 1
            continue

        ntitle = _norm_title(title)
        topic_exists = ntitle in title_map
        if topic_exists:
            cand.operation = "NEW_EDGE_ONLY"
            canonical_title = title_map[ntitle]
        else:
            cand.operation = "NEW_TOPIC_AND_EDGES"
            canonical_title = title

        if not cand.relationships:
            cand.rejection_reason = "no_relationships"
            counts["no_relationship_count"] += 1
            cand.accepted = False
            rejected.append(cand)
            counts["rejected_count"] += 1
            continue

        # Resolve relationship endpoints
        pending_edges: list[tuple[str, str]] = []
        invalid = False
        for frm, to in cand.relationships:
            # Prefer attaching: from = existing dependent, to = candidate prerequisite
            frm_n, to_n = _norm_title(frm), _norm_title(to)
            # Map candidate title variants to canonical
            if to_n == ntitle:
                to_resolved = canonical_title
            elif to_n in title_map:
                to_resolved = title_map[to_n]
            else:
                # Unknown target that isn't the candidate
                cand.rejection_reason = "invalid_target"
                counts["invalid_target_count"] += 1
                invalid = True
                break
            if frm_n == ntitle:
                frm_resolved = canonical_title
            elif frm_n in title_map:
                frm_resolved = title_map[frm_n]
            else:
                cand.rejection_reason = "invalid_target"
                counts["invalid_target_count"] += 1
                invalid = True
                break
            if frm_resolved == to_resolved:
                cand.rejection_reason = "self_loop"
                counts["cycle_rejected_count"] += 1
                invalid = True
                break
            pending_edges.append((frm_resolved, to_resolved))

        if invalid:
            cand.accepted = False
            rejected.append(cand)
            counts["rejected_count"] += 1
            continue

        # Duplicate topic (edge-only is OK)
        if not topic_exists:
            # Add topic tentatively for cycle checks
            topics.append(
                {
                    "title": canonical_title,
                    "summary": cand.summary or cand.reason or f"Prerequisite concept: {canonical_title}",
                    "confidence": cand.confidence,
                }
            )
            title_map[ntitle] = canonical_title
            title_to_id[canonical_title] = f"t{len(title_to_id)}"
        else:
            # Edge-only: if all edges already exist, duplicate
            if all((_norm_title(a), _norm_title(b)) in existing_edge_norms for a, b in pending_edges):
                cand.rejection_reason = "duplicate_edges"
                counts["duplicate_count"] += 1
                cand.accepted = False
                rejected.append(cand)
                counts["rejected_count"] += 1
                continue

        # Cycle check each new edge against accumulating accepted set
        added_any = False
        for frm, to in pending_edges:
            en = (_norm_title(frm), _norm_title(to))
            if en in existing_edge_norms:
                skipped.append(
                    SkippedProposedDependency(from_title=frm, to_title=to, reason="duplicate dependency")
                )
                continue
            from_id = title_to_id.get(frm)
            to_id = title_to_id.get(to)
            if not from_id or not to_id:
                skipped.append(
                    SkippedProposedDependency(from_title=frm, to_title=to, reason="unknown topic reference")
                )
                counts["invalid_target_count"] += 1
                continue
            if would_create_cycle(from_id, to_id, accepted_dep_dicts):
                skipped.append(
                    SkippedProposedDependency(
                        from_title=frm,
                        to_title=to,
                        reason="would create a cycle with other proposed dependencies",
                    )
                )
                counts["cycle_rejected_count"] += 1
                continue
            deps.append({"from": frm, "to": to})
            accepted_dep_dicts.append({"from_topic_id": from_id, "to_topic_id": to_id})
            existing_edge_norms.add(en)
            added_any = True

        if not added_any and not topic_exists:
            # Rolled back empty topic add
            topics.pop()
            title_map.pop(ntitle, None)
            title_to_id.pop(canonical_title, None)
            cand.rejection_reason = cand.rejection_reason or "no_valid_edges"
            cand.accepted = False
            rejected.append(cand)
            counts["rejected_count"] += 1
            continue

        if not added_any and topic_exists:
            cand.rejection_reason = "no_valid_edges"
            cand.accepted = False
            rejected.append(cand)
            counts["rejected_count"] += 1
            continue

        if topic_exists and added_any:
            cand.operation = "NEW_EDGE_ONLY"
        elif not topic_exists and added_any:
            cand.operation = "NEW_TOPIC_AND_EDGES"

        cand.accepted = True
        cand.rejection_reason = None
        accepted.append(cand)
        counts["applied_count"] += 1

    return topics, deps, accepted, rejected, skipped, counts


def apply_parsed_coverage_recovery(
    *,
    baseline_topics: list[dict[str, Any]],
    baseline_dependencies: list[dict[str, str]],
    raw_llm_text: str | None,
    parse_error: str | None = None,
    max_candidates: int | None = None,
    llm_meta: dict[str, Any] | None = None,
) -> CoverageRecoveryResult:
    """Pure merge path used by tests and by the live LLM wrapper."""
    limit = max_recovery_candidates() if max_candidates is None else max_candidates
    result = CoverageRecoveryResult(max_candidates=limit)
    if llm_meta:
        result.llm_latency_ms = float(llm_meta.get("llm_latency_ms") or 0.0)
        result.input_tokens = llm_meta.get("input_tokens")
        result.output_tokens = llm_meta.get("output_tokens")
        result.tokens_estimated = bool(llm_meta.get("tokens_estimated", True))
        result.cost_usd = llm_meta.get("cost_usd")
    result.raw_response = raw_llm_text

    if parse_error:
        result.parse_ok = False
        result.error = parse_error
        result.topics_after = list(baseline_topics)
        result.dependencies_after = list(baseline_dependencies)
        return result

    try:
        data = parse_llm_json_object(raw_llm_text or "{}")
    except Exception as exc:
        result.parse_ok = False
        result.error = str(exc)
        result.topics_after = list(baseline_topics)
        result.dependencies_after = list(baseline_dependencies)
        return result

    all_cands = parse_coverage_candidates(data)
    # Non-required still recorded then rejected
    required = [c for c in all_cands if c.category == REQUIRED_CATEGORY]
    non_required = [c for c in all_cands if c.category != REQUIRED_CATEGORY]
    retained, truncated = truncate_candidates(required, max_n=limit)

    topics_after, deps_after, accepted, rejected, skipped, counts = validate_and_merge_recovery(
        baseline_topics=baseline_topics,
        baseline_dependencies=baseline_dependencies,
        retained=retained,
    )
    for c in truncated:
        counts["truncated_count"] += 1
        counts["rejected_count"] += 1
        rejected.append(c)
    for c in non_required:
        if c.category == "OPTIONAL_NICE_TO_HAVE":
            counts["optional_rejected_count"] += 1
            c.rejection_reason = "optional_nice_to_have"
        elif c.category == "RELATED_BUT_NOT_REQUIRED":
            counts["related_rejected_count"] += 1
            c.rejection_reason = "related_but_not_required"
        elif c.category == "OUT_OF_SCOPE":
            counts["out_of_scope_count"] += 1
            c.rejection_reason = "out_of_scope"
        else:
            c.rejection_reason = "unknown_category"
        c.accepted = False
        rejected.append(c)
        counts["rejected_count"] += 1
        counts["candidate_count"] += 1

    counts["candidate_count"] = len(all_cands)
    baseline_title_norms = {_norm_title(str(t.get("title") or "")) for t in baseline_topics}
    baseline_edge_norms = {
        (_norm_title(str(d.get("from") or "")), _norm_title(str(d.get("to") or "")))
        for d in baseline_dependencies
    }
    new_topics = [
        str(t["title"])
        for t in topics_after
        if _norm_title(str(t["title"])) not in baseline_title_norms
    ]
    new_edges = [
        (str(d["from"]), str(d["to"]))
        for d in deps_after
        if (_norm_title(str(d["from"])), _norm_title(str(d["to"]))) not in baseline_edge_norms
    ]

    result.all_candidates = all_cands
    result.retained_candidates = retained
    result.accepted_candidates = accepted
    result.rejected_candidates = rejected
    result.topics_after = topics_after
    result.dependencies_after = deps_after
    result.skipped_dependencies = skipped
    result.new_topic_titles = new_topics
    result.new_edges = new_edges
    result.counts = counts
    return result


async def run_coverage_recovery_pass(
    *,
    learning_objective: str,
    baseline_topics: list[dict[str, Any]],
    baseline_dependencies: list[dict[str, str]],
    temperature: float = 0.0,
    seed: int | None = None,
    max_candidates: int | None = None,
) -> CoverageRecoveryResult:
    """One LLM coverage audit + deterministic merge. Never recursive."""
    prompt = build_coverage_recovery_prompt(
        learning_objective=learning_objective,
        topics=[{"title": str(t.get("title") or ""), "summary": str(t.get("summary") or "")} for t in baseline_topics],
        edges=[
            (str(d.get("from") or ""), str(d.get("to") or ""))
            for d in baseline_dependencies
            if d.get("from") and d.get("to")
        ],
    )
    t0 = time.perf_counter()
    try:
        with llm_operation("coverage_recovery"):
            record = await call_llm_detailed(prompt, temperature=temperature, seed=seed)
        raw = record.text
        llm_meta = {
            "llm_latency_ms": (time.perf_counter() - t0) * 1000.0,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "tokens_estimated": record.tokens_estimated,
            "cost_usd": record.estimated_cost_usd,
        }
        return apply_parsed_coverage_recovery(
            baseline_topics=baseline_topics,
            baseline_dependencies=baseline_dependencies,
            raw_llm_text=raw,
            max_candidates=max_candidates,
            llm_meta=llm_meta,
        )
    except Exception as exc:
        return apply_parsed_coverage_recovery(
            baseline_topics=baseline_topics,
            baseline_dependencies=baseline_dependencies,
            raw_llm_text=None,
            parse_error=str(exc),
            max_candidates=max_candidates,
            llm_meta={"llm_latency_ms": (time.perf_counter() - t0) * 1000.0},
        )
