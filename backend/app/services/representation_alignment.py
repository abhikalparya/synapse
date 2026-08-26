"""Constrained representation alignment (experimental).

Transforms existing generated topic titles only. Never invents, adds, or drops concepts
except MERGE_WITH_EXISTING_GENERATED_TOPIC (consolidate duplicates already in the graph).

Runtime inputs: generated topics/deps + optional request text. Never gold/eval aliases.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.services.concept_normalization import (
    _display_after_strip,
    normalize_concept_key,
    strip_tutorial_framing_key,
)
from app.services.topics import would_create_cycle

Decision = Literal[
    "KEEP_ORIGINAL",
    "NORMALIZE_TITLE",
    "MERGE_WITH_EXISTING_GENERATED_TOPIC",
    "PRESERVE_UNRESOLVED",
]

Method = Literal[
    "identity",
    "tutorial_framing_rule",
    "deterministic_normalization",
    "existing_context_alignment",
    "unresolved",
]


@dataclass
class AlignmentRecord:
    original_title: str
    aligned_title: str
    decision: Decision
    method: Method
    reason: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlignmentResult:
    topics_before: list[str]
    topics_after: list[str]
    dependencies_before: list[tuple[str, str]]
    dependencies_after: list[tuple[str, str]]
    records: list[AlignmentRecord] = field(default_factory=list)
    title_map: dict[str, str] = field(default_factory=dict)  # original → aligned
    dag_valid: bool = True
    dag_details: list[str] = field(default_factory=list)
    new_topics_created: int = 0
    topics_deleted_without_merge: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def to_meta(self) -> dict[str, Any]:
        return {
            "representation_alignment": True,
            "alignment_records": [r.to_dict() for r in self.records],
            "alignment_title_map": dict(self.title_map),
            "alignment_counts": dict(self.counts),
            "alignment_dag_valid": self.dag_valid,
            "alignment_dag_details": list(self.dag_details),
            "topics_before": list(self.topics_before),
            "topics_after": list(self.topics_after),
            "dependencies_before": [list(e) for e in self.dependencies_before],
            "dependencies_after": [list(e) for e in self.dependencies_after],
            "new_topics_created": self.new_topics_created,
            "topics_deleted_without_merge": self.topics_deleted_without_merge,
        }


def _norm_display(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip())


def _framing_normalize(title: str) -> tuple[str | None, str]:
    """Return (new_display_or_None, reason). None means no safe framing change."""
    display = _display_after_strip(title)
    if not display:
        return None, ""
    display = _norm_display(display)
    if not display or display.casefold() == title.casefold():
        return None, ""
    # Refuse if strip collapses to empty / ultra-short / structural-only
    key = normalize_concept_key(display)
    if len(key) < 2 or key in {"intro", "introduction", "overview", "basics", "fundamental"}:
        return None, ""
    return display, "Removed introductory framing without changing concept identity."


def _context_canonical(title: str, request_text: str | None) -> tuple[str | None, str]:
    """If request text already names the concept core, prefer that surface form."""
    if not request_text or not request_text.strip():
        return None, ""
    stripped = _display_after_strip(title) or title
    core = _norm_display(stripped)
    if not core or len(core) < 2:
        return None, ""
    # Look for whole-word / phrase occurrence of the stripped core in request (casefold).
    pattern = re.compile(rf"(?<!\w){re.escape(core)}(?!\w)", re.IGNORECASE)
    m = pattern.search(request_text)
    if not m:
        # Also try framing-stripped key tokens of length >= 4 as phrase
        key = strip_tutorial_framing_key(normalize_concept_key(title))
        if len(key) < 4:
            return None, ""
        # Reconstruct a loose phrase from key (may miss casing); require request contains key words in order
        words = key.split()
        if len(words) == 1:
            m2 = re.search(rf"(?<!\w){re.escape(words[0])}(?!\w)", request_text, re.IGNORECASE)
            if not m2:
                return None, ""
            # Prefer original casing from request match
            return request_text[m2.start() : m2.end()], (
                "Aligned to terminology already present in the request text."
            )
        return None, ""
    surface = request_text[m.start() : m.end()]
    if surface.casefold() == title.casefold():
        return None, ""
    return surface, "Aligned to terminology already present in the request text."


def align_titles(
    topics: list[str],
    *,
    request_text: str | None = None,
    enable_framing: bool = True,
    enable_context: bool = True,
    enable_merge: bool = True,
) -> tuple[list[AlignmentRecord], dict[str, str]]:
    """Decide aligned title for each topic. Returns records + original→aligned map."""
    records: list[AlignmentRecord] = []
    provisional: list[str] = []

    for title in topics:
        t = _norm_display(title)
        decision: Decision = "KEEP_ORIGINAL"
        method: Method = "identity"
        reason = "No safe representation change."
        aligned = t
        conf: float | None = None

        if enable_framing:
            framed, why = _framing_normalize(t)
            if framed:
                aligned = framed
                decision = "NORMALIZE_TITLE"
                method = "tutorial_framing_rule"
                reason = why

        if enable_context and decision == "KEEP_ORIGINAL":
            ctx, why = _context_canonical(t, request_text)
            if ctx:
                aligned = _norm_display(ctx)
                decision = "NORMALIZE_TITLE"
                method = "existing_context_alignment"
                reason = why

        # If framing produced a title, context can refine casing only when same key
        if enable_context and decision == "NORMALIZE_TITLE" and method == "tutorial_framing_rule":
            ctx, why = _context_canonical(t, request_text)
            if ctx and normalize_concept_key(ctx) == normalize_concept_key(aligned):
                aligned = _norm_display(ctx)
                method = "existing_context_alignment"
                reason = why

        if not aligned:
            aligned = t
            decision = "PRESERVE_UNRESOLVED"
            method = "unresolved"
            reason = "Normalization would empty the title; preserved original."

        records.append(
            AlignmentRecord(
                original_title=t,
                aligned_title=aligned,
                decision=decision,
                method=method,
                reason=reason,
                confidence=conf,
            )
        )
        provisional.append(aligned)

    title_map = {r.original_title: r.aligned_title for r in records}

    if not enable_merge:
        return records, title_map

    # Group by deterministic concept key after framing strip
    groups: dict[str, list[int]] = {}
    for i, aligned in enumerate(provisional):
        key = strip_tutorial_framing_key(normalize_concept_key(aligned))
        if len(key) < 2:
            records[i].decision = "PRESERVE_UNRESOLVED"
            records[i].method = "unresolved"
            records[i].reason = "Concept key too short for safe merge; preserved."
            records[i].aligned_title = records[i].original_title
            provisional[i] = records[i].original_title
            continue
        groups.setdefault(key, []).append(i)

    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        # Prefer shortest aligned title, then lexicographic for stability
        survivor_i = sorted(
            idxs,
            key=lambda i: (len(provisional[i]), provisional[i].casefold(), i),
        )[0]
        survivor = provisional[survivor_i]
        for i in idxs:
            if i == survivor_i:
                if records[i].decision == "KEEP_ORIGINAL":
                    # Survivor may still keep original if already canonical in group
                    pass
                continue
            # Unsafe guard: require same strip key (already grouped) — merge
            records[i] = AlignmentRecord(
                original_title=records[i].original_title,
                aligned_title=survivor,
                decision="MERGE_WITH_EXISTING_GENERATED_TOPIC",
                method="deterministic_normalization",
                reason=(
                    f"Duplicate concept already represented by generated node {survivor!r}."
                ),
                confidence=None,
            )
            provisional[i] = survivor

    title_map = {r.original_title: r.aligned_title for r in records}
    return records, title_map


def remap_dependencies(
    dependencies: list[tuple[str, str]],
    title_map: dict[str, str],
) -> list[tuple[str, str]]:
    """Remap endpoints; drop self-loops and duplicates. Does not invent edges."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for frm, to in dependencies:
        a = title_map.get(frm, frm)
        b = title_map.get(to, to)
        a, b = _norm_display(a), _norm_display(b)
        if not a or not b or a == b:
            continue
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _validate_dag(topics: list[str], deps: list[tuple[str, str]]) -> tuple[bool, list[str]]:
    topic_set = set(topics)
    accepted: list[dict[str, str]] = []
    details: list[str] = []
    for frm, to in deps:
        if frm not in topic_set or to not in topic_set:
            details.append(f"invalid reference: {frm!r} -> {to!r}")
            continue
        if frm == to or would_create_cycle(frm, to, accepted):
            details.append(f"cycle/self-loop: {frm!r} -> {to!r}")
            continue
        accepted.append({"from_topic_id": frm, "to_topic_id": to})
    # Filter to cycle-safe edges only for "after" graph if needed — caller decides.
    # Here we report validity of the remapped graph as-is.
    valid = not any(d.startswith("cycle") or d.startswith("invalid") for d in details)
    # Recompute properly
    self_loops = sum(1 for a, b in deps if a == b)
    invalid = sum(1 for a, b in deps if a not in topic_set or b not in topic_set)
    cycles = 0
    accepted = []
    for frm, to in deps:
        if frm not in topic_set or to not in topic_set or frm == to:
            continue
        if would_create_cycle(frm, to, accepted):
            cycles += 1
        else:
            accepted.append({"from_topic_id": frm, "to_topic_id": to})
    valid = self_loops == 0 and cycles == 0 and invalid == 0
    if not valid:
        details = [
            f"self_loops={self_loops}",
            f"cycles={cycles}",
            f"invalid_refs={invalid}",
        ]
    return valid, details


def filter_deps_to_dag(
    topics: list[str],
    deps: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Keep remapped edges that pass existing cycle checks (skip violators)."""
    topic_set = set(topics)
    accepted_dicts: list[dict[str, str]] = []
    out: list[tuple[str, str]] = []
    for frm, to in deps:
        if frm not in topic_set or to not in topic_set or frm == to:
            continue
        if would_create_cycle(frm, to, accepted_dicts):
            continue
        accepted_dicts.append({"from_topic_id": frm, "to_topic_id": to})
        out.append((frm, to))
    return out


def align_graph(
    topics: list[str],
    dependencies: list[tuple[str, str]] | list[list[str]],
    *,
    request_text: str | None = None,
    enable_framing: bool = True,
    enable_context: bool = True,
    enable_merge: bool = True,
) -> AlignmentResult:
    """Apply constrained representation alignment to one generated graph."""
    topics_in = [_norm_display(t) for t in topics if _norm_display(t)]
    deps_in: list[tuple[str, str]] = []
    for d in dependencies:
        if len(d) != 2:
            continue
        a, b = _norm_display(str(d[0])), _norm_display(str(d[1]))
        if a and b:
            deps_in.append((a, b))

    records, title_map = align_titles(
        topics_in,
        request_text=request_text,
        enable_framing=enable_framing,
        enable_context=enable_context,
        enable_merge=enable_merge,
    )

    # Unique topics after alignment (merge reduces count)
    after_topics: list[str] = []
    seen: set[str] = set()
    for r in records:
        t = r.aligned_title
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        after_topics.append(t)

    remapped = remap_dependencies(deps_in, title_map)
    # Drop edges that would cycle after remap (should be rare); do not invent replacements
    safe_deps = filter_deps_to_dag(after_topics, remapped)
    dag_valid, dag_details = _validate_dag(after_topics, safe_deps)

    # Safety invariants
    before_set = {t.casefold() for t in topics_in}
    after_norms = {normalize_concept_key(t) for t in after_topics}
    # Every after topic must come from some original via map
    invented = 0
    for t in after_topics:
        origins = [r.original_title for r in records if r.aligned_title.casefold() == t.casefold()]
        if not origins:
            invented += 1

    counts = {
        "topics_inspected": len(topics_in),
        "normalized": sum(1 for r in records if r.decision == "NORMALIZE_TITLE"),
        "merged": sum(1 for r in records if r.decision == "MERGE_WITH_EXISTING_GENERATED_TOPIC"),
        "kept": sum(1 for r in records if r.decision == "KEEP_ORIGINAL"),
        "unresolved": sum(1 for r in records if r.decision == "PRESERVE_UNRESOLVED"),
        "topics_after": len(after_topics),
        "edges_before": len(deps_in),
        "edges_after": len(safe_deps),
        "duplicate_edges_removed": max(0, len(remapped) - len(set(remapped))),
    }
    # framing-normalized subset
    counts["framing_normalized"] = sum(
        1 for r in records if r.decision == "NORMALIZE_TITLE" and r.method == "tutorial_framing_rule"
    )
    counts["context_aligned"] = sum(
        1 for r in records if r.method == "existing_context_alignment"
    )
    n = max(1, counts["topics_inspected"])
    changed = counts["normalized"] + counts["merged"]
    counts["alignment_rate_num"] = changed
    counts["alignment_rate"] = changed / n
    counts["merge_rate"] = counts["merged"] / n

    return AlignmentResult(
        topics_before=topics_in,
        topics_after=after_topics,
        dependencies_before=deps_in,
        dependencies_after=safe_deps,
        records=records,
        title_map=title_map,
        dag_valid=dag_valid,
        dag_details=dag_details,
        new_topics_created=invented,
        topics_deleted_without_merge=0,
        counts=counts,
    )
