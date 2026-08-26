"""Deterministic per-request concept inventory normalization (Concept-First Stage 2).

No uncontrolled fuzzy semantic merges. When uncertain, prefer PRESERVE_UNRESOLVED.
Granularity policy is consistency-within-graph, not a universal ontology level.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

Decision = Literal[
    "ACCEPT",
    "MERGE",
    "REJECT_DUPLICATE",
    "REJECT_OUT_OF_SCOPE",
    "SPLIT",
    "PRESERVE_UNRESOLVED",
]

DetectedCondition = Literal[
    "EXACT_DUPLICATE",
    "NORMALIZED_DUPLICATE",
    "CLEAR_ALIAS",
    "GRANULARITY_CONFLICT",
    "ABSTRACTION_CONFLICT",
    "DECOMPOSITION_CONFLICT",
    "OUT_OF_SCOPE_CANDIDATE",
    "UNKNOWN",
]

# Documented granularity policy (within one request inventory):
# 1. A concept should be a learnable unit.
# 2. Specific enough to participate in prerequisite relationships.
# 3. Not unnecessarily fragmented.
# 4. Comparable in abstraction level to neighboring concepts.
# Do NOT enforce a universal global granularity; detect conflicts and record them.
GRANULARITY_POLICY = (
    "Within a single request inventory, prefer learnable units that are specific enough "
    "for prerequisite edges, not over-fragmented, and comparable in abstraction to neighbors. "
    "Conflicts are recorded; uncertain cases are preserved unresolved rather than auto-split."
)

_BOILERPLATE_PREFIXES = (
    "introduction to ",
    "intro to ",
    "basics of ",
    "basics ",
    "overview of ",
    "overview ",
    "understanding ",
    "getting started with ",
    "getting started ",
    "advanced ",
    "fundamentals of ",
    "fundamentals ",
)

_BOILERPLATE_SUFFIXES = (
    " basics",
    " fundamentals",
    " overview",
    " introduction",
)

# Structural / tutorial scaffolding — reject unless they somehow survive other checks.
_OUT_OF_SCOPE_EXACT = frozenset(
    {
        "introduction",
        "intro",
        "overview",
        "miscellaneous",
        "misc",
        "advanced topics",
        "getting started",
        "conclusion",
        "summary",
        "appendix",
        "resources",
        "further reading",
    }
)

_STRUCTURAL_RE = re.compile(
    r"^(module|lesson|chapter|unit|part|section|week|day)\s*\d+\b",
    re.IGNORECASE,
)

# Broader umbrellas vs typical finer peers (normalized forms). Used only for conflict detection.
_ABSTRACTION_UMBRELLAS = frozenset(
    {
        "programming fundamental",
        "programming fundamentals",
        "computer science",
        "mathematics",
        "machine learning",
        "software engineering",
        "cloud computing basic",
        "web application basic",
        "distributed systems fundamental",
        "distributed systems basic",
    }
)

_DECOMPOSITION_PARTS: dict[str, frozenset[str]] = {
    "programming fundamental": frozenset(
        {"variable", "data type", "control flow", "function", "loop", "conditional"}
    ),
    "programming fundamentals": frozenset(
        {"variable", "data type", "control flow", "function", "loop", "conditional"}
    ),
    "linear algebra": frozenset(
        {"vector", "matrix", "determinant", "matrix operation"}
    ),
    "probability": frozenset({"statistic", "combinatoric", "independence of event"}),
}

# Exact normalized synonym → canonical display preference (not fuzzy).
_KNOWN_ALIAS_NORMS: dict[str, str] = {
    "control structures": "Control Flow",
    "control structure": "Control Flow",
    "hash functions": "Hashing",
    "hash function": "Hashing",
}


def normalize_concept_key(title: str) -> str:
    s = unicodedata.normalize("NFKD", title or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold().strip()
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    tokens = []
    for tok in s.split():
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
            tok = tok[:-1]
        tokens.append(tok)
    return " ".join(tokens)


def strip_tutorial_framing_key(norm: str) -> str:
    s = norm
    changed = True
    while changed:
        changed = False
        for p in _BOILERPLATE_PREFIXES:
            if s.startswith(p):
                s = s[len(p) :].strip()
                changed = True
        for suf in _BOILERPLATE_SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf):
                s = s[: -len(suf)].strip()
                changed = True
    return s


def _display_after_strip(original: str) -> str | None:
    """Best-effort display title after removing tutorial framing from the original string."""
    s = original.strip()
    low = s.casefold()
    for p in _BOILERPLATE_PREFIXES:
        if low.startswith(p):
            rest = s[len(p) :].strip(" -:–—")
            return rest if rest else None
    for suf in (" Basics", " Fundamentals", " Overview", " Introduction"):
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)].strip()
    return None


def _is_out_of_scope(title: str, key: str) -> bool:
    if key in _OUT_OF_SCOPE_EXACT:
        return True
    if _STRUCTURAL_RE.match(title.strip()):
        return True
    if re.match(r"^(advanced|basic)\s+topics?$", key):
        return True
    return False


@dataclass
class CandidateConcept:
    title: str
    description: str = ""
    reason: str = ""


@dataclass
class NormalizationDecision:
    original_title: str
    normalized_title: str | None
    decision: Decision
    detected_condition: DetectedCondition
    decision_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizationResult:
    inventory: list[CandidateConcept] = field(default_factory=list)
    decisions: list[NormalizationDecision] = field(default_factory=list)
    accepted_count: int = 0
    merged_count: int = 0
    duplicate_rejection_count: int = 0
    out_of_scope_rejection_count: int = 0
    granularity_conflict_count: int = 0
    abstraction_conflict_count: int = 0
    decomposition_conflict_count: int = 0
    unresolved_count: int = 0
    granularity_policy: str = GRANULARITY_POLICY

    def to_dict(self) -> dict[str, Any]:
        return {
            "granularity_policy": self.granularity_policy,
            "inventory": [
                {"title": c.title, "description": c.description, "reason": c.reason}
                for c in self.inventory
            ],
            "decisions": [d.to_dict() for d in self.decisions],
            "accepted_count": self.accepted_count,
            "merged_count": self.merged_count,
            "duplicate_rejection_count": self.duplicate_rejection_count,
            "out_of_scope_rejection_count": self.out_of_scope_rejection_count,
            "granularity_conflict_count": self.granularity_conflict_count,
            "abstraction_conflict_count": self.abstraction_conflict_count,
            "decomposition_conflict_count": self.decomposition_conflict_count,
            "unresolved_count": self.unresolved_count,
        }


def _inventory_keys(inventory: Sequence[CandidateConcept]) -> dict[str, str]:
    """Map normalized key → display title currently in inventory."""
    return {normalize_concept_key(c.title): c.title for c in inventory}


def _count_finer_neighbors(key: str, inventory_keys: dict[str, str]) -> int:
    parts = _DECOMPOSITION_PARTS.get(key)
    if not parts:
        # Also check if this umbrella's stem matches a known map key
        for umbrella, part_set in _DECOMPOSITION_PARTS.items():
            if key == umbrella or key.startswith(umbrella):
                parts = part_set
                break
    if not parts:
        return 0
    return sum(1 for k in inventory_keys if k in parts or any(k.startswith(p + " ") for p in parts))


def _has_umbrella_neighbor(key: str, inventory_keys: dict[str, str]) -> str | None:
    for umbrella, parts in _DECOMPOSITION_PARTS.items():
        if key in parts or any(key == p or key.startswith(p + " ") for p in parts):
            if umbrella in inventory_keys:
                return inventory_keys[umbrella]
    return None


def normalize_concepts(candidates: Sequence[CandidateConcept | dict[str, Any]]) -> NormalizationResult:
    """Normalize candidate concepts into a consistent per-request inventory."""
    result = NormalizationResult()
    parsed: list[CandidateConcept] = []
    for raw in candidates:
        if isinstance(raw, CandidateConcept):
            parsed.append(raw)
        elif isinstance(raw, dict):
            title = str(raw.get("title", "")).strip()
            if not title:
                continue
            parsed.append(
                CandidateConcept(
                    title=title,
                    description=str(raw.get("description") or raw.get("summary") or "").strip(),
                    reason=str(raw.get("reason") or "").strip(),
                )
            )

    for cand in parsed:
        original_title = cand.title
        working = cand
        key = normalize_concept_key(working.title)
        inv_keys = _inventory_keys(result.inventory)
        stripped_key = strip_tutorial_framing_key(key)
        framing_rewritten = False

        # 1) Out of scope structural labels
        if _is_out_of_scope(working.title, key):
            result.decisions.append(
                NormalizationDecision(
                    original_title=original_title,
                    normalized_title=None,
                    decision="REJECT_OUT_OF_SCOPE",
                    detected_condition="OUT_OF_SCOPE_CANDIDATE",
                    decision_reason="Structural or tutorial scaffolding label, not a learnable concept.",
                )
            )
            result.out_of_scope_rejection_count += 1
            continue

        # 2) Exact duplicate against current inventory
        if key in inv_keys:
            result.decisions.append(
                NormalizationDecision(
                    original_title=original_title,
                    normalized_title=inv_keys[key],
                    decision="REJECT_DUPLICATE",
                    detected_condition="EXACT_DUPLICATE",
                    decision_reason=f"Exact normalized duplicate of existing inventory concept {inv_keys[key]!r}.",
                )
            )
            result.duplicate_rejection_count += 1
            continue

        # 3) Tutorial framing → merge into existing canonical, or rewrite title
        if stripped_key and stripped_key != key:
            if stripped_key in inv_keys:
                result.decisions.append(
                    NormalizationDecision(
                        original_title=original_title,
                        normalized_title=inv_keys[stripped_key],
                        decision="MERGE",
                        detected_condition="CLEAR_ALIAS",
                        decision_reason=(
                            "Tutorial framing removed; canonical concept already exists in inventory."
                        ),
                    )
                )
                result.merged_count += 1
                continue
            display = _display_after_strip(working.title) or stripped_key.title()
            if stripped_key in _KNOWN_ALIAS_NORMS:
                display = _KNOWN_ALIAS_NORMS[stripped_key]
                alias_key = normalize_concept_key(display)
                if alias_key in inv_keys:
                    result.decisions.append(
                        NormalizationDecision(
                            original_title=original_title,
                            normalized_title=inv_keys[alias_key],
                            decision="MERGE",
                            detected_condition="CLEAR_ALIAS",
                            decision_reason="Tutorial framing + known alias map to existing inventory concept.",
                        )
                    )
                    result.merged_count += 1
                    continue
            working = CandidateConcept(
                title=display,
                description=working.description,
                reason=working.reason,
            )
            key = normalize_concept_key(working.title)
            stripped_key = strip_tutorial_framing_key(key)
            inv_keys = _inventory_keys(result.inventory)
            framing_rewritten = True
            if key in inv_keys:
                result.decisions.append(
                    NormalizationDecision(
                        original_title=original_title,
                        normalized_title=inv_keys[key],
                        decision="MERGE",
                        detected_condition="NORMALIZED_DUPLICATE",
                        decision_reason="After tutorial framing removal, matches existing inventory concept.",
                    )
                )
                result.merged_count += 1
                continue

        # 4) Known exact alias map
        if key in _KNOWN_ALIAS_NORMS:
            canonical = _KNOWN_ALIAS_NORMS[key]
            ckey = normalize_concept_key(canonical)
            if ckey in inv_keys:
                result.decisions.append(
                    NormalizationDecision(
                        original_title=original_title,
                        normalized_title=inv_keys[ckey],
                        decision="MERGE",
                        detected_condition="CLEAR_ALIAS",
                        decision_reason=f"Known exact alias of inventory concept {inv_keys[ckey]!r}.",
                    )
                )
                result.merged_count += 1
                continue
            working = CandidateConcept(
                title=canonical,
                description=working.description,
                reason=working.reason,
            )
            key = ckey
            framing_rewritten = True

        # 5) Normalized duplicate via stripped key against inventory
        if stripped_key and stripped_key in inv_keys and stripped_key != key:
            result.decisions.append(
                NormalizationDecision(
                    original_title=original_title,
                    normalized_title=inv_keys[stripped_key],
                    decision="MERGE",
                    detected_condition="NORMALIZED_DUPLICATE",
                    decision_reason="Normalized form matches an existing inventory concept.",
                )
            )
            result.merged_count += 1
            continue

        # 6) Granularity / abstraction / decomposition conflicts (do not auto-split)
        finer = _count_finer_neighbors(key, inv_keys)
        if (key in _ABSTRACTION_UMBRELLAS or key in _DECOMPOSITION_PARTS) and finer >= 2:
            result.inventory.append(working)
            result.decisions.append(
                NormalizationDecision(
                    original_title=original_title,
                    normalized_title=working.title,
                    decision="PRESERVE_UNRESOLVED",
                    detected_condition="GRANULARITY_CONFLICT",
                    decision_reason=(
                        "Candidate is broader than multiple existing finer concepts; "
                        "cannot deterministically split or merge."
                    ),
                )
            )
            result.granularity_conflict_count += 1
            result.unresolved_count += 1
            continue

        if key in _ABSTRACTION_UMBRELLAS and finer >= 1:
            result.inventory.append(working)
            result.decisions.append(
                NormalizationDecision(
                    original_title=original_title,
                    normalized_title=working.title,
                    decision="PRESERVE_UNRESOLVED",
                    detected_condition="ABSTRACTION_CONFLICT",
                    decision_reason=(
                        "Candidate abstraction level conflicts with finer neighboring concepts."
                    ),
                )
            )
            result.abstraction_conflict_count += 1
            result.unresolved_count += 1
            continue

        umbrella_title = _has_umbrella_neighbor(key, inv_keys)
        if umbrella_title:
            result.inventory.append(working)
            result.decisions.append(
                NormalizationDecision(
                    original_title=original_title,
                    normalized_title=working.title,
                    decision="PRESERVE_UNRESOLVED",
                    detected_condition="DECOMPOSITION_CONFLICT",
                    decision_reason=(
                        f"Candidate appears to decompose broader inventory concept {umbrella_title!r}; "
                        "left unresolved rather than auto-merging."
                    ),
                )
            )
            result.decomposition_conflict_count += 1
            result.unresolved_count += 1
            continue

        # 7) Accept clean (or framing-rewritten) concept
        result.inventory.append(working)
        condition: DetectedCondition = "CLEAR_ALIAS" if framing_rewritten else "UNKNOWN"
        reason = (
            "Tutorial framing removed; accepted rewritten canonical title."
            if framing_rewritten
            else "Accepted into per-request inventory."
        )
        result.decisions.append(
            NormalizationDecision(
                original_title=original_title,
                normalized_title=working.title,
                decision="ACCEPT",
                detected_condition=condition,
                decision_reason=reason,
            )
        )
        result.accepted_count += 1

    return result
