"""Deterministic Stage-1 inventory pruning (Concept-First experimental add-on).

Runtime rules use only the learning objective and candidate concept titles.
They never inspect gold topics, aliases, or evaluation labels.

Pruning is opt-in via generation strategy ``concept_first_pruned``; default
``concept_first`` leaves the inventory unchanged after normalization.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

from app.evaluation.metrics import normalize_topic, topic_similarity, topic_tokens

PruneDecision = Literal["KEEP", "PRUNE"]
PruneReason = Literal[
    "DUPLICATE",
    "NEAR_DUPLICATE",
    "GENERIC_FILLER",
    "LOW_INFORMATION",
    "OBJECTIVE_MISMATCH",
    "REDUNDANT_CONCEPT",
    "MALFORMED",
    "UNKNOWN",
]

# Function words only — never domain vocabulary. Used for objective-mismatch signal.
_OBJECTIVE_STOPWORDS = frozenset(
    {
        "learn",
        "learning",
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "or",
        "for",
        "in",
        "on",
        "with",
        "used",
        "use",
        "how",
        "what",
        "basics",
        "basic",
        "introduction",
        "intro",
        "overview",
        "fundamentals",
        "fundamental",
        "about",
        "into",
        "from",
        "by",
        "as",
        "is",
        "are",
        "be",
        "this",
        "that",
        "their",
        "they",
        "you",
        "your",
        "using",
        "via",
        "an",
    }
)

_GENERIC_EXACT = frozenset(
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
        "basics",
        "fundamentals",
    }
)

_STRUCTURAL_RE = re.compile(
    r"^(module|lesson|chapter|unit|part|section|week|day)\s*\d+\b",
    re.IGNORECASE,
)

_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)

# Near-duplicate threshold: high to avoid collapsing distinct short concepts.
NEAR_DUPLICATE_SIMILARITY = 0.85

# Objective-mismatch: no content-token overlap with the goal AND weakly related to
# inventory peers (conservative peer gate so isolated but goal-linked peers can survive
# via other concepts — peer gate alone is not enough when all peers are also mismatch).
OBJECTIVE_MISMATCH_PEER_SIM_MAX = 0.25


def content_tokens(text: str) -> set[str]:
    """Deterministic content tokens (stopwords stripped). Shared by runtime + offline analysis."""
    return {t for t in topic_tokens(text) if t not in _OBJECTIVE_STOPWORDS and len(t) > 2}


@dataclass
class PruneAuditEntry:
    original_title: str
    normalized_title: str
    decision: PruneDecision
    reason: PruneReason
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PruneResult:
    input_concepts: list[str] = field(default_factory=list)
    kept_concepts: list[str] = field(default_factory=list)
    pruned_concepts: list[str] = field(default_factory=list)
    decisions: list[PruneAuditEntry] = field(default_factory=list)
    input_concept_count: int = 0
    kept_concept_count: int = 0
    pruned_concept_count: int = 0
    retention_rate: float = 1.0
    fallback_to_original_inventory: bool = False
    config_name: str = "combined_conservative"

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "input_concepts": list(self.input_concepts),
            "kept_concepts": list(self.kept_concepts),
            "pruned_concepts": list(self.pruned_concepts),
            "decisions": [d.to_dict() for d in self.decisions],
            "input_concept_count": self.input_concept_count,
            "kept_concept_count": self.kept_concept_count,
            "pruned_concept_count": self.pruned_concept_count,
            "retention_rate": self.retention_rate,
            "fallback_to_original_inventory": self.fallback_to_original_inventory,
        }


def is_malformed(title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return True
    if _PUNCT_ONLY_RE.match(t):
        return True
    if len(normalize_topic(t)) < 2:
        return True
    if not re.search(r"[A-Za-z0-9]", t):
        return True
    return False


def is_generic_filler(title: str) -> bool:
    """Structural/tutorial scaffolding — small explicit set, not a giant blacklist."""
    t = (title or "").strip()
    if not t:
        return False
    key = normalize_topic(t)
    if key in _GENERIC_EXACT:
        return True
    if _STRUCTURAL_RE.match(t):
        return True
    if re.match(r"^(advanced|basic)\s+topics?$", key):
        return True
    return False


def has_objective_content_overlap(title: str, objective: str) -> bool:
    return bool(content_tokens(objective) & content_tokens(title))


PRUNE_CONFIGS: dict[str, dict[str, Any]] = {
    "no_pruning": {
        "exact_duplicate": False,
        "near_duplicate": False,
        "malformed": False,
        "generic_filler": False,
        "objective_mismatch": False,
        "redundant_containment": False,
    },
    "exact_duplicate": {
        "exact_duplicate": True,
        "near_duplicate": False,
        "malformed": False,
        "generic_filler": False,
        "objective_mismatch": False,
        "redundant_containment": False,
    },
    "near_duplicate": {
        "exact_duplicate": True,
        "near_duplicate": True,
        "malformed": False,
        "generic_filler": False,
        "objective_mismatch": False,
        "redundant_containment": False,
    },
    "malformed_and_filler": {
        "exact_duplicate": True,
        "near_duplicate": False,
        "malformed": True,
        "generic_filler": True,
        "objective_mismatch": False,
        "redundant_containment": False,
    },
    "objective_mismatch": {
        "exact_duplicate": True,
        "near_duplicate": False,
        "malformed": True,
        "generic_filler": True,
        "objective_mismatch": True,
        "redundant_containment": False,
    },
    "combined_conservative": {
        "exact_duplicate": True,
        "near_duplicate": True,
        "malformed": True,
        "generic_filler": True,
        "objective_mismatch": True,
        "redundant_containment": True,
    },
}


def prune_inventory(
    concepts: Sequence[str],
    objective: str,
    *,
    config_name: str = "combined_conservative",
    near_duplicate_similarity: float = NEAR_DUPLICATE_SIMILARITY,
    objective_mismatch_peer_sim_max: float = OBJECTIVE_MISMATCH_PEER_SIM_MAX,
) -> PruneResult:
    """Prune a normalized concept inventory. Never uses gold/eval data.

    Ordering of kept concepts is stable (first-seen order of survivors).
    If every concept would be pruned, fall back to the original inventory.
    """
    if config_name not in PRUNE_CONFIGS:
        raise ValueError(f"Unknown prune config {config_name!r}; choose one of {list(PRUNE_CONFIGS)}")
    flags = PRUNE_CONFIGS[config_name]

    titles = [str(t).strip() for t in concepts if str(t).strip()]
    result = PruneResult(
        input_concepts=list(titles),
        input_concept_count=len(titles),
        config_name=config_name,
    )
    if not titles:
        result.kept_concepts = []
        result.kept_concept_count = 0
        result.retention_rate = 1.0
        return result

    if config_name == "no_pruning" or not any(flags.values()):
        for t in titles:
            result.decisions.append(
                PruneAuditEntry(
                    original_title=t,
                    normalized_title=normalize_topic(t),
                    decision="KEEP",
                    reason="UNKNOWN",
                    detail="Pruning disabled for this configuration.",
                )
            )
        result.kept_concepts = list(titles)
        result.kept_concept_count = len(titles)
        result.retention_rate = 1.0
        return result

    kept: list[str] = []
    decisions: list[PruneAuditEntry] = []
    seen_norms: set[str] = set()

    for title in titles:
        norm = normalize_topic(title)

        if flags["malformed"] and is_malformed(title):
            decisions.append(
                PruneAuditEntry(title, norm, "PRUNE", "MALFORMED", "Empty, punctuation-only, or non-informative title.")
            )
            continue

        if flags["generic_filler"] and is_generic_filler(title):
            decisions.append(
                PruneAuditEntry(
                    title,
                    norm,
                    "PRUNE",
                    "GENERIC_FILLER",
                    "Structural/tutorial scaffolding label, not a learnable concept.",
                )
            )
            continue

        if flags["exact_duplicate"] and norm in seen_norms:
            decisions.append(
                PruneAuditEntry(title, norm, "PRUNE", "DUPLICATE", "Exact normalized duplicate of an earlier kept concept.")
            )
            continue

        if flags["near_duplicate"]:
            near_hit = None
            for k in kept:
                if topic_similarity(title, k) >= near_duplicate_similarity and normalize_topic(k) != norm:
                    near_hit = k
                    break
            if near_hit is not None:
                decisions.append(
                    PruneAuditEntry(
                        title,
                        norm,
                        "PRUNE",
                        "NEAR_DUPLICATE",
                        f"High similarity (≥{near_duplicate_similarity}) to earlier kept concept {near_hit!r}.",
                    )
                )
                continue

        if flags["redundant_containment"]:
            tt = topic_tokens(title)
            redundant_of = None
            for k in kept:
                kt = topic_tokens(k)
                # Strict token subset of a longer kept concept → redundant fragment.
                if tt and kt and tt < kt and len(kt) - len(tt) >= 1:
                    redundant_of = k
                    break
            if redundant_of is not None:
                decisions.append(
                    PruneAuditEntry(
                        title,
                        norm,
                        "PRUNE",
                        "REDUNDANT_CONCEPT",
                        f"Token set strictly contained in kept concept {redundant_of!r}.",
                    )
                )
                continue

        if flags["objective_mismatch"]:
            if not has_objective_content_overlap(title, objective):
                peers = [topic_similarity(title, k) for k in titles if k != title]
                best_peer = max(peers) if peers else 0.0
                if best_peer < objective_mismatch_peer_sim_max:
                    decisions.append(
                        PruneAuditEntry(
                            title,
                            norm,
                            "PRUNE",
                            "OBJECTIVE_MISMATCH",
                            (
                                "No content-token overlap with the learning objective and weak "
                                f"similarity to inventory peers (best_peer={best_peer:.3f} "
                                f"< {objective_mismatch_peer_sim_max})."
                            ),
                        )
                    )
                    continue

        # KEEP
        decisions.append(
            PruneAuditEntry(title, norm, "KEEP", "UNKNOWN", "Retained by deterministic pruning rules.")
        )
        kept.append(title)
        seen_norms.add(norm)

    if not kept and titles:
        # Retention guard: never empty unless input empty.
        result.fallback_to_original_inventory = True
        result.kept_concepts = list(titles)
        result.pruned_concepts = []
        result.decisions = [
            PruneAuditEntry(
                t,
                normalize_topic(t),
                "KEEP",
                "UNKNOWN",
                "FALLBACK_TO_ORIGINAL_INVENTORY: all concepts would have been pruned.",
            )
            for t in titles
        ]
        result.kept_concept_count = len(titles)
        result.pruned_concept_count = 0
        result.retention_rate = 1.0
        return result

    result.decisions = decisions
    result.kept_concepts = kept
    result.pruned_concepts = [d.original_title for d in decisions if d.decision == "PRUNE"]
    result.kept_concept_count = len(kept)
    result.pruned_concept_count = len(result.pruned_concepts)
    result.retention_rate = (len(kept) / len(titles)) if titles else 1.0
    return result
