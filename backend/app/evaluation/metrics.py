"""Deterministic topic/dependency matching and graph quality metrics.

Matching strategy (no LLM judge):
1. Unicode NFKD, casefold, strip punctuation to spaces, collapse whitespace
2. Drop leading articles (a/an/the)
3. Light English plural stemming on tokens
4. Optional per-example ``topic_aliases`` (gold title -> acceptable synonyms) —
   exact normalized identity only (aliases never enter fuzzy matching)
5. Optional ``allowed_extra_topics`` (not required for recall; not counted as hallucinations)
6. Token containment (shorter title's tokens ⊆ longer) or Jaccard ≥ 0.5 against
   **canonical** gold titles only; 1–3 letter acronyms do not match via containment
   (so ``SQL`` ≠ ``SQL Injection``)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from app.evaluation.schemas import EvalExample, GeneratedGraph, GraphQualityScores, GraphValidity
from app.services.topics import would_create_cycle

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")


def normalize_topic(title: str) -> str:
    s = unicodedata.normalize("NFKD", title or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    s = _ARTICLE_RE.sub("", s)
    tokens = [_light_stem(tok) for tok in s.split() if tok]
    return " ".join(tokens)


def _light_stem(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "xes", "zes")):
        return token[:-2]
    if token.endswith("es") and len(token) > 4 and not token.endswith(("ses", "nes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 4:
        return token[:-1]
    return token


def topic_tokens(title: str) -> set[str]:
    return {t for t in normalize_topic(title).split() if t}


def topic_similarity(a: str, b: str) -> float:
    """Deterministic overlap in [0, 1]. 1.0 is exact normalized equality."""
    na, nb = normalize_topic(a), normalize_topic(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ta, tb = topic_tokens(a), topic_tokens(b)
    if not ta or not tb:
        return 0.0
    smaller, larger = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if smaller <= larger:
        only = next(iter(smaller))
        if len(smaller) >= 2 or (len(smaller) == 1 and len(only) >= 4):
            return 0.75
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


_SIMILARITY_THRESHOLD = 0.5


def match_topic(title: str, example: EvalExample) -> str | None:
    """Return the canonical required/optional title if ``title`` matches, else None.

    Order:
    1. Exact normalized equality against gold titles and explicit aliases
    2. Token containment / Jaccard ≥ 0.5 against **canonical gold titles only**

    Aliases never participate in fuzzy matching — only exact normalized identity —
    so a curated alias cannot silently accept granularity variants or related phrases.
    """
    index = _alias_index(example)
    hit = index.get(normalize_topic(title))
    if hit is not None:
        return hit

    best: str | None = None
    best_score = 0.0
    # Fuzzy candidates: canonical gold surfaces only (not alias strings).
    seen_canon: set[str] = set()
    for gold in [*example.required_topic_list(), *example.gold_topics, *example.optional_topic_list()]:
        key = normalize_topic(gold)
        if not key or key in seen_canon:
            continue
        seen_canon.add(key)
        score = topic_similarity(title, gold)
        if score > best_score:
            best_score = score
            best = gold
    if best is not None and best_score >= _SIMILARITY_THRESHOLD:
        return best
    return None


def _alias_index(example: EvalExample) -> dict[str, str]:
    """Map normalized alias/title -> canonical title (original casing)."""
    index: dict[str, str] = {}
    for gold in [*example.required_topic_list(), *example.gold_topics, *example.optional_topic_list()]:
        index[normalize_topic(gold)] = gold
        for alias in example.topic_aliases.get(gold, []):
            index[normalize_topic(alias)] = gold
    return index


def is_required_topic(title: str, example: EvalExample) -> bool:
    matched = match_topic(title, example)
    if matched is None:
        return False
    required_norms = {normalize_topic(t) for t in example.required_topic_list()}
    return normalize_topic(matched) in required_norms


def is_gold_topic(title: str, example: EvalExample) -> bool:
    """Backward-compatible alias: a required (must-recall) topic."""
    return is_required_topic(title, example)


def is_optional_topic(title: str, example: EvalExample) -> bool:
    matched = match_topic(title, example)
    if matched is None:
        return False
    optional_norms = {normalize_topic(t) for t in example.optional_topic_list()}
    return normalize_topic(matched) in optional_norms


def is_in_scope(title: str, example: EvalExample) -> bool:
    return match_topic(title, example) is not None


def dedupe_topics(topics: Iterable[str]) -> tuple[list[str], int]:
    """Keep first occurrence of each normalized title; return (unique, duplicate_count)."""
    seen: set[str] = set()
    unique: list[str] = []
    duplicates = 0
    for t in topics:
        key = normalize_topic(t)
        if not key:
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(t)
    return unique, duplicates


def assess_graph_validity(topics: list[str], dependencies: list[tuple[str, str]]) -> GraphValidity:
    """Reuse Synapse's ``would_create_cycle`` for DAG checks; also count self-loops / bad refs."""
    topic_set = {t for t in topics if t}
    # Title identity for cycle checks: use the topic string as the id.
    accepted: list[dict[str, str]] = []
    self_loops = 0
    cycles = 0
    invalid_refs = 0
    details: list[str] = []

    for frm, to in dependencies:
        if frm not in topic_set or to not in topic_set:
            invalid_refs += 1
            details.append(f"invalid reference: {frm!r} -> {to!r}")
            continue
        if frm == to or would_create_cycle(frm, to, accepted):
            if frm == to:
                self_loops += 1
                details.append(f"self-loop: {frm!r}")
            else:
                cycles += 1
                details.append(f"cycle attempt: {frm!r} -> {to!r}")
            continue
        accepted.append({"from_topic_id": frm, "to_topic_id": to})

    is_valid = self_loops == 0 and cycles == 0 and invalid_refs == 0
    return GraphValidity(
        is_valid=is_valid,
        self_loops=self_loops,
        cycles=cycles,
        invalid_references=invalid_refs,
        details=details,
    )


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def find_redundant_transitive_edges(dependencies: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return generated edges that are implied by a longer prerequisite path.

    Synapse edge ``(from, to)`` means *from requires to* (follow ``from → to`` toward
    foundations). ``(A, C)`` is redundant when there is a path ``A → … → C`` of length ≥ 2
    that does not use ``(A, C)`` itself. Necessary direct edges with no alternate path
    are not classified as redundant.
    """
    # Preserve first occurrence; skip self-loops for path search.
    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for frm, to in dependencies:
        if not frm or not to or frm == to:
            continue
        key = (frm, to)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)

    redundant: list[tuple[str, str]] = []
    for u, v in unique:
        adj: dict[str, list[str]] = {}
        for a, b in unique:
            if (a, b) == (u, v):
                continue
            adj.setdefault(a, []).append(b)
        stack = list(adj.get(u, []))
        visited: set[str] = set()
        found = False
        while stack:
            node = stack.pop()
            if node == v:
                found = True
                break
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adj.get(node, []))
        if found:
            redundant.append((u, v))
    return redundant


def _norm_edge(frm: str, to: str) -> tuple[str, str]:
    return (normalize_topic(frm), normalize_topic(to))


def _canon_edge(frm: str, to: str, example: EvalExample) -> tuple[str, str] | None:
    mf = match_topic(frm, example)
    mt = match_topic(to, example)
    if mf is None or mt is None:
        return None
    return _norm_edge(mf, mt)


def compare_graphs(example: EvalExample, generated: GeneratedGraph) -> dict[str, Any]:
    """Human-inspectable gold vs generated alignment. No LLM judge."""

    unique_topics, _dupes = dedupe_topics(generated.topics)
    required = example.required_topic_list()
    optional = example.optional_topic_list()
    required_deps = example.required_dependency_list()
    acceptable = list(example.acceptable_dependencies)

    required_norms = {normalize_topic(t) for t in required}
    optional_norms = {normalize_topic(t) for t in optional}

    gen_to_canon: dict[str, str] = {}
    matched_required: set[str] = set()
    matched_optional: set[str] = set()
    unmatched_generated: list[str] = []
    for t in unique_topics:
        canon = match_topic(t, example)
        if canon is None:
            unmatched_generated.append(t)
            continue
        gen_to_canon[t] = canon
        cn = normalize_topic(canon)
        if cn in required_norms:
            matched_required.add(cn)
        elif cn in optional_norms:
            matched_optional.add(cn)

    missing_topics = [t for t in required if normalize_topic(t) not in matched_required]

    required_dep_set = {_norm_edge(a, b) for a, b in required_deps}
    acceptable_dep_set = {_norm_edge(a, b) for a, b in acceptable}
    ambiguous_dep_set = {_norm_edge(a, b) for a, b in example.ambiguous_dependencies}
    reverse_required = {(b, a) for a, b in required_dep_set}
    original_required = {_norm_edge(a, b): (a, b) for a, b in required_deps}

    matched_deps: list[tuple[str, str]] = []
    extra_deps: list[tuple[str, str]] = []
    ambiguous_deps: list[tuple[str, str]] = []
    reversed_deps: list[tuple[list[str], list[str]]] = []
    used_acceptable: list[tuple[str, str]] = []
    reversed_required_norms: set[tuple[str, str]] = set()
    matched_required_norms: set[tuple[str, str]] = set()
    gen_canon_edges: list[tuple[str, str]] = []

    for frm, to in generated.dependencies:
        mapped = _canon_edge(frm, to, example)
        if mapped is None:
            extra_deps.append((frm, to))
            continue
        gen_canon_edges.append(mapped)
        if mapped in required_dep_set:
            # One-to-one accounting: each unique required edge matches at most once.
            # Duplicate generated edges that map to an already-matched required edge
            # are extras (not additional matches). Scoring correctness fix — not a
            # calibration / matching-rule change.
            if mapped in matched_required_norms:
                extra_deps.append((frm, to))
            else:
                matched_deps.append((frm, to))
                matched_required_norms.add(mapped)
        elif mapped in reverse_required:
            gold_norm = (mapped[1], mapped[0])
            if gold_norm in reversed_required_norms:
                # Duplicate reverse of an already-recorded direction error.
                extra_deps.append((frm, to))
            else:
                # Record reverse even if the forward required edge was already matched:
                # emitting both directions is a direction error, not a second match.
                gold = original_required.get(gold_norm, gold_norm)
                reversed_deps.append(([frm, to], [gold[0], gold[1]]))
                reversed_required_norms.add(gold_norm)
        elif mapped in acceptable_dep_set:
            used_acceptable.append((frm, to))
        elif mapped in ambiguous_dep_set:
            ambiguous_deps.append((frm, to))
        else:
            extra_deps.append((frm, to))

    missing_deps = [
        (a, b)
        for a, b in required_deps
        if _norm_edge(a, b) not in matched_required_norms
        and _norm_edge(a, b) not in reversed_required_norms
    ]

    closest: list[dict[str, Any]] = []
    for t in unmatched_generated + missing_topics:
        best_title = ""
        best = 0.0
        pool = unique_topics if t in missing_topics else required
        for other in pool:
            score = topic_similarity(t, other)
            if score > best:
                best = score
                best_title = other
        closest.append({"title": t, "closest": best_title, "similarity": round(best, 3)})

    redundant = find_redundant_transitive_edges(list(generated.dependencies))
    redundant_set = set(redundant)
    extra_redundant = [e for e in extra_deps if e in redundant_set]
    extra_non_redundant = [e for e in extra_deps if e not in redundant_set]

    return {
        "required_topics": required,
        "optional_topics": optional,
        "generated_topics": unique_topics,
        "matched_required_topics": sorted(matched_required),
        "matched_optional_topics": sorted(matched_optional),
        "missing_topics": missing_topics,
        "extra_topics": unmatched_generated,
        "required_dependencies": [list(d) for d in required_deps],
        "acceptable_dependencies": [list(d) for d in acceptable],
        "ambiguous_dependencies": [list(d) for d in example.ambiguous_dependencies],
        "generated_dependencies": [list(d) for d in generated.dependencies],
        "matched_dependencies": [list(d) for d in matched_deps],
        "matched_required_edge_count": len(matched_required_norms),
        "missing_dependencies": [list(d) for d in missing_deps],
        "extra_dependencies": [list(d) for d in extra_deps],
        "extra_dependencies_non_redundant": [list(d) for d in extra_non_redundant],
        "ambiguous_dependencies_used": [list(d) for d in ambiguous_deps],
        "reversed_dependencies": reversed_deps,
        "reversed_required_edge_count": len(reversed_required_norms),
        "acceptable_dependencies_used": [list(d) for d in used_acceptable],
        "redundant_transitive_edges": [list(d) for d in redundant],
        "redundant_transitive_among_extra": [list(d) for d in extra_redundant],
        "closest_unmatched": closest,
        "skipped_dependencies": list(generated.skipped_dependencies),
    }


def score_graph(example: EvalExample, generated: GeneratedGraph) -> GraphQualityScores:
    unique_topics, duplicate_count = dedupe_topics(generated.topics)
    required = example.required_topic_list()
    required_deps = example.required_dependency_list()
    comparison = compare_graphs(example, generated)

    matched_topics = len(comparison["matched_required_topics"])
    in_scope = matched_topics + len(comparison["matched_optional_topics"])
    hallucinated = len(comparison["extra_topics"])
    gen_n = len(unique_topics)
    gold_n = len(required)
    topic_precision = (in_scope / gen_n) if gen_n else 1.0
    topic_recall = (matched_topics / gold_n) if gold_n else 1.0

    matched_deps = int(
        comparison.get("matched_required_edge_count", len(comparison["matched_dependencies"]))
    )
    gen_dep_n = len(generated.dependencies)
    gold_dep_n = len(required_deps)
    acceptable_n = len(comparison["acceptable_dependencies_used"])
    ambiguous_n = len(comparison.get("ambiguous_dependencies_used") or [])
    # Legacy precision: required matches + acceptable alternatives count as correct generated edges.
    # Acceptable alternatives do NOT inflate required-edge recall.
    correct_gen_deps = matched_deps + acceptable_n
    dependency_precision = (correct_gen_deps / gen_dep_n) if gen_dep_n else 1.0
    dependency_recall = (matched_deps / gold_dep_n) if gold_dep_n else 1.0

    reversed_n = int(
        comparison.get("reversed_required_edge_count", len(comparison["reversed_dependencies"]))
    )
    # Invalid extras = unresolved structural extras (ambiguous edges are tracked separately
    # and are NOT counted as correct).
    invalid_extra_n = len(comparison["extra_dependencies"])
    # Legacy extra rate includes invalid extras only (ambiguous pulled out of extras in compare).
    extra_n = invalid_extra_n + ambiguous_n
    extra_rate = (extra_n / gen_dep_n) if gen_dep_n else 0.0
    invalid_extra_rate = (invalid_extra_n / gen_dep_n) if gen_dep_n else 0.0
    acceptable_rate = (acceptable_n / gen_dep_n) if gen_dep_n else 0.0
    ambiguous_rate = (ambiguous_n / gen_dep_n) if gen_dep_n else 0.0
    direction_rate = (reversed_n / gold_dep_n) if gold_dep_n else 0.0
    redundant_n = len(comparison["redundant_transitive_edges"])
    redundant_rate = (redundant_n / gen_dep_n) if gen_dep_n else 0.0

    validity = assess_graph_validity(unique_topics, list(generated.dependencies))
    cycle_attempt = validity.cycles > 0 or validity.self_loops > 0
    if any("cycle" in (s.get("reason") or "").casefold() for s in generated.skipped_dependencies):
        cycle_attempt = True

    # Unique required edges that are either correctly matched or reversed (direction error).
    # An edge present as both forward and reverse counts once toward coverage so
    # missing_required_edge_rate stays in [0, 1].
    if matched_deps > gold_dep_n or reversed_n > gold_dep_n:
        raise AssertionError(
            f"unique matched ({matched_deps}) or reversed ({reversed_n}) exceed required ({gold_dep_n})"
        )
    # Coverage uses set semantics: missing list already excludes matched∪reversed.
    missing_n = len(comparison["missing_dependencies"])
    missing = (missing_n / gold_dep_n) if gold_dep_n else 0.0
    if not (0.0 <= missing <= 1.0):
        raise AssertionError(f"missing_required_edge_rate out of range: {missing}")
    halluc_rate = (hallucinated / gen_n) if gen_n else 0.0

    # Required-edge metrics: alternatives do not count toward precision numerator.
    req_denom = matched_deps + invalid_extra_n + reversed_n
    required_edge_precision = (matched_deps / req_denom) if req_denom else 1.0
    required_edge_recall = dependency_recall
    required_edge_f1 = _f1(required_edge_precision, required_edge_recall)

    failures: list[str] = []
    if generated.error_category:
        failures.append(generated.error_category)
    if duplicate_count:
        failures.append("DUPLICATE_TOPIC")
    if hallucinated:
        failures.append("HALLUCINATED_TOPIC")
    if comparison["missing_topics"]:
        failures.append("MISSING_TOPIC")
    if missing > 0:
        failures.append("MISSING_PREREQUISITE")
    if comparison["extra_dependencies_non_redundant"]:
        failures.append("EXTRA_DEPENDENCY")
        failures.append("INCORRECT_DEPENDENCY")
        failures.append("INVALID_EXTRA_EDGE")
    if ambiguous_n:
        failures.append("AMBIGUOUS_EDGE")
    if acceptable_n:
        failures.append("MATCHED_ACCEPTABLE_ALTERNATIVE_EDGE")
    if redundant_n:
        failures.append("REDUNDANT_TRANSITIVE_EDGE")
    if reversed_n:
        failures.append("WRONG_DEPENDENCY_DIRECTION")
    if validity.self_loops:
        failures.append("SELF_LOOP")
    if validity.cycles or any("cycle" in (s.get("reason") or "").casefold() for s in generated.skipped_dependencies):
        failures.append("CYCLE_ATTEMPT")
    if validity.invalid_references:
        failures.append("INVALID_TOPIC_REFERENCE")
    for s in generated.skipped_dependencies:
        reason = (s.get("reason") or "").casefold()
        if "out-of-scope" in reason or "out of scope" in reason:
            failures.append("OUT_OF_SCOPE_REFERENCE")

    seen_f: set[str] = set()
    ordered_failures: list[str] = []
    for f in failures:
        if f not in seen_f:
            seen_f.add(f)
            ordered_failures.append(f)

    return GraphQualityScores(
        topic_precision=topic_precision,
        topic_recall=topic_recall,
        topic_f1=_f1(topic_precision, topic_recall),
        dependency_precision=dependency_precision,
        dependency_recall=dependency_recall,
        dependency_f1=_f1(dependency_precision, dependency_recall),
        graph_valid=validity.is_valid,
        cycle_attempt=cycle_attempt,
        missing_prerequisite_rate=missing,
        hallucinated_topic_rate=halluc_rate,
        extra_dependency_rate=extra_rate,
        dependency_direction_error_rate=direction_rate,
        reversed_dependencies=reversed_n,
        redundant_transitive_edge_count=redundant_n,
        redundant_transitive_edge_rate=redundant_rate,
        required_edge_precision=required_edge_precision,
        required_edge_recall=required_edge_recall,
        required_edge_f1=required_edge_f1,
        missing_required_edge_rate=missing,
        acceptable_alternative_edge_count=acceptable_n,
        acceptable_alternative_rate=acceptable_rate,
        invalid_extra_edge_count=invalid_extra_n,
        invalid_extra_edge_rate=invalid_extra_rate,
        ambiguous_edge_count=ambiguous_n,
        ambiguous_edge_rate=ambiguous_rate,
        duplicate_topics=duplicate_count,
        matched_topics=matched_topics,
        generated_topics=gen_n,
        gold_topics=gold_n,
        matched_dependencies=matched_deps,
        generated_dependencies=gen_dep_n,
        gold_dependencies=gold_dep_n,
        failures=ordered_failures,
    )


def aggregate_scores(scores: list[GraphQualityScores]) -> dict[str, float]:
    """Macro-average of per-example scores (each learning goal weighted equally)."""
    if not scores:
        return {
            "topic_precision": 0.0,
            "topic_recall": 0.0,
            "topic_f1": 0.0,
            "dependency_precision": 0.0,
            "dependency_recall": 0.0,
            "dependency_f1": 0.0,
            "graph_validity_rate": 0.0,
            "cycle_attempt_rate": 0.0,
            "missing_prerequisite_rate": 0.0,
            "hallucinated_topic_rate": 0.0,
            "extra_dependency_rate": 0.0,
            "dependency_direction_error_rate": 0.0,
            "redundant_transitive_edge_rate": 0.0,
            "redundant_transitive_edge_count": 0.0,
            "required_edge_precision": 0.0,
            "required_edge_recall": 0.0,
            "required_edge_f1": 0.0,
            "missing_required_edge_rate": 0.0,
            "acceptable_alternative_rate": 0.0,
            "acceptable_alternative_edge_count": 0.0,
            "invalid_extra_edge_rate": 0.0,
            "invalid_extra_edge_count": 0.0,
            "ambiguous_edge_rate": 0.0,
            "ambiguous_edge_count": 0.0,
            "n": 0.0,
        }

    def mean(attr: str) -> float:
        return sum(getattr(s, attr) for s in scores) / len(scores)

    return {
        "topic_precision": mean("topic_precision"),
        "topic_recall": mean("topic_recall"),
        "topic_f1": mean("topic_f1"),
        "dependency_precision": mean("dependency_precision"),
        "dependency_recall": mean("dependency_recall"),
        "dependency_f1": mean("dependency_f1"),
        "graph_validity_rate": sum(1.0 for s in scores if s.graph_valid) / len(scores),
        "cycle_attempt_rate": sum(1.0 for s in scores if s.cycle_attempt) / len(scores),
        "missing_prerequisite_rate": mean("missing_prerequisite_rate"),
        "hallucinated_topic_rate": mean("hallucinated_topic_rate"),
        "extra_dependency_rate": mean("extra_dependency_rate"),
        "dependency_direction_error_rate": mean("dependency_direction_error_rate"),
        "redundant_transitive_edge_rate": mean("redundant_transitive_edge_rate"),
        "redundant_transitive_edge_count": mean("redundant_transitive_edge_count"),
        "required_edge_precision": mean("required_edge_precision"),
        "required_edge_recall": mean("required_edge_recall"),
        "required_edge_f1": mean("required_edge_f1"),
        "missing_required_edge_rate": mean("missing_required_edge_rate"),
        "acceptable_alternative_rate": mean("acceptable_alternative_rate"),
        "acceptable_alternative_edge_count": mean("acceptable_alternative_edge_count"),
        "invalid_extra_edge_rate": mean("invalid_extra_edge_rate"),
        "invalid_extra_edge_count": mean("invalid_extra_edge_count"),
        "ambiguous_edge_rate": mean("ambiguous_edge_rate"),
        "ambiguous_edge_count": mean("ambiguous_edge_count"),
        "n": float(len(scores)),
    }
