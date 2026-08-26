"""Structured failure categories for prerequisite-graph evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

FAILURE_CATEGORIES = (
    "MISSING_PREREQUISITE",
    "MISSING_TOPIC",
    "INCORRECT_DEPENDENCY",
    "EXTRA_DEPENDENCY",
    "REDUNDANT_TRANSITIVE_EDGE",
    "EXTRA_TOPIC",
    "WRONG_DEPENDENCY_DIRECTION",
    "TITLE_PARAPHRASE",
    "ALIAS_MISMATCH",
    "GRANULARITY_MISMATCH",
    "GOLD_GRAPH_AMBIGUITY",
    "HALLUCINATED_TOPIC",
    "DUPLICATE_TOPIC",
    "SELF_LOOP",
    "CYCLE_ATTEMPT",
    "INVALID_TOPIC_REFERENCE",
    "OUT_OF_SCOPE_REFERENCE",
    "LLM_PARSE_FAILURE",
    "LLM_TIMEOUT",
    "LLM_PROVIDER_FAILURE",
    "SEMANTIC_AUDIT_UNAVAILABLE",
)

_DISPLAY = {
    "MISSING_PREREQUISITE": "Missing prerequisite",
    "MISSING_TOPIC": "Missing topic",
    "INCORRECT_DEPENDENCY": "Incorrect dependency",
    "EXTRA_DEPENDENCY": "Extra dependency",
    "REDUNDANT_TRANSITIVE_EDGE": "Redundant transitive edge",
    "EXTRA_TOPIC": "Extra topic",
    "WRONG_DEPENDENCY_DIRECTION": "Wrong dependency direction",
    "TITLE_PARAPHRASE": "Title paraphrase",
    "ALIAS_MISMATCH": "Alias mismatch",
    "GRANULARITY_MISMATCH": "Granularity mismatch",
    "GOLD_GRAPH_AMBIGUITY": "Gold graph ambiguity",
    "HALLUCINATED_TOPIC": "Hallucinated topic",
    "DUPLICATE_TOPIC": "Duplicate topic",
    "SELF_LOOP": "Self-loop",
    "CYCLE_ATTEMPT": "Cycle attempt",
    "INVALID_TOPIC_REFERENCE": "Invalid topic reference",
    "OUT_OF_SCOPE_REFERENCE": "Out-of-scope reference",
    "LLM_PARSE_FAILURE": "LLM parse failure",
    "LLM_TIMEOUT": "LLM timeout",
    "LLM_PROVIDER_FAILURE": "LLM provider failure",
    "SEMANTIC_AUDIT_UNAVAILABLE": "Semantic audit unavailable",
}


def classify_llm_exception(exc: BaseException) -> str:
    msg = str(exc).casefold()
    if "timeout" in msg or "timed out" in msg:
        return "LLM_TIMEOUT"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        # JSON/shape failures from parse paths
        if "json" in msg or "llm" in msg or "expect" in msg:
            return "LLM_PARSE_FAILURE"
        return "LLM_PARSE_FAILURE"
    return "LLM_PROVIDER_FAILURE"


def summarize_failures(failure_lists: Iterable[Iterable[str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for failures in failure_lists:
        for f in failures:
            if f in FAILURE_CATEGORIES:
                counts[f] += 1
            elif f:
                counts[f] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def format_failure_table(counts: dict[str, int]) -> str:
    if not counts:
        return "Failure Type                 Count\n----------------------------------\n(none)"
    lines = ["Failure Type                 Count", "----------------------------------"]
    for key, n in counts.items():
        label = _DISPLAY.get(key, key)
        lines.append(f"{label:<28} {n:>5}")
    return "\n".join(lines)
