"""Pure relationship failure analysis (offline, evaluation only).

Isolates required edges where both endpoints are consistently present under
curated_alias matching with acceptable representation (EXACT/ALIAS only), yet
the gold direct edge is never generated. Classifies omission modes.

Makes no LLM calls and does not change generation, gold, aliases, or metrics.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.inspect import _graph_from_row
from app.evaluation.metrics import compare_graphs, match_topic, normalize_topic
from app.evaluation.node_edge_attribution import (
    classify_gold_topic_representation,
    load_node_representation_map,
)
from app.evaluation.persistent_failure_attribution import (
    FREQ_CONSISTENT,
    _edge_key,
    _edge_matched,
    _group_rows,
    _norm_edge,
    _topic_present,
    find_latest_stability_artifact,
    has_prerequisite_path,
)
from app.evaluation.schemas import EvalExample, GeneratedGraph

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "failure_analysis"
DEFAULT_BENCH = _REPO_ROOT / "results" / "benchmarks"

ACCEPTABLE_REP = frozenset({"EXACT_MATCH", "ALIAS_MATCH"})
UNACCEPTABLE_REP = frozenset(
    {
        "GRANULARITY_VARIANT",
        "DECOMPOSED",
        "ABSTRACTED",
        "RELATED_BUT_DISTINCT",
        "MISSING",
        "UNKNOWN",
    }
)

FAILURE_CATEGORIES = (
    "REVERSED_DIRECTION",
    "ALTERNATE_DIRECT_RELATIONSHIP",
    "ALTERNATIVE_PATH",
    "MISSING_DIRECT_PREREQUISITE",
    "SCOPE_OR_DIRECTNESS_ERROR",
    "UNKNOWN",
)


def _safe_rate(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def _gen_to_gold_map(graph: GeneratedGraph, example: EvalExample) -> dict[str, str]:
    """Map generated title → matched gold title (normalized keys)."""
    out: dict[str, str] = {}
    for t in graph.topics:
        m = match_topic(t, example)
        if m is not None:
            out[normalize_topic(t)] = normalize_topic(m)
    return out


def remap_dependencies_to_gold(
    graph: GeneratedGraph,
    example: EvalExample,
) -> list[tuple[str, str]]:
    """Remap generated deps into gold title space where endpoints match."""
    mapping = _gen_to_gold_map(graph, example)
    remapped: list[tuple[str, str]] = []
    for a, b in graph.dependencies:
        na, nb = normalize_topic(str(a)), normalize_topic(str(b))
        remapped.append((mapping.get(na, na), mapping.get(nb, nb)))
    return remapped


def edge_present_in_remapped(
    frm: str,
    to: str,
    remapped: list[tuple[str, str]],
) -> bool:
    key = _norm_edge(frm, to)
    return any(_norm_edge(a, b) == key for a, b in remapped)


def shortest_prerequisite_path_length(
    remapped: list[tuple[str, str]],
    source: str,
    target: str,
) -> int | None:
    """BFS length of shortest directed path source → … → target, or None."""
    src_n, tgt_n = normalize_topic(source), normalize_topic(target)
    if src_n == tgt_n:
        return 0
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in remapped:
        adj[normalize_topic(a)].add(normalize_topic(b))
    queue: list[tuple[str, int]] = [(src_n, 0)]
    seen = {src_n}
    while queue:
        cur, dist = queue.pop(0)
        for nxt in adj.get(cur, ()):
            if nxt == tgt_n:
                return dist + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def endpoints_acceptably_present(
    source: str,
    target: str,
    example: EvalExample,
    graph: GeneratedGraph,
    *,
    rep_map: dict[str, Any],
    cmp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict purity gate: curated_alias match + EXACT/ALIAS representation."""
    cmp = cmp if cmp is not None else compare_graphs(example, graph)
    src_matched = _topic_present(source, cmp)
    tgt_matched = _topic_present(target, cmp)
    src_lab = classify_gold_topic_representation(source, example, graph, rep_map=rep_map)
    tgt_lab = classify_gold_topic_representation(target, example, graph, rep_map=rep_map)
    src_ok = src_matched and src_lab.get("status") in ACCEPTABLE_REP
    tgt_ok = tgt_matched and tgt_lab.get("status") in ACCEPTABLE_REP
    reasons: list[str] = []
    if not src_matched:
        reasons.append("source_not_matched")
    elif src_lab.get("status") not in ACCEPTABLE_REP:
        reasons.append(f"source_rep:{src_lab.get('status')}")
    if not tgt_matched:
        reasons.append("target_not_matched")
    elif tgt_lab.get("status") not in ACCEPTABLE_REP:
        reasons.append(f"target_rep:{tgt_lab.get('status')}")
    return {
        "ok": src_ok and tgt_ok,
        "source_matched": src_matched,
        "target_matched": tgt_matched,
        "source_status": src_lab.get("status"),
        "target_status": tgt_lab.get("status"),
        "exclude_reasons": reasons,
    }


def classify_relationship_failure(
    *,
    gold_edge_freq: float,
    reverse_freq: float,
    alt_path_freq: float,
    alt_direct_from_source_freq: float,
    no_relationship_freq: float,
    median_path_length: float | None,
    source_gold_target_recall: float,
    source_has_any_outgoing_freq: float,
) -> str:
    """Assign exactly one primary failure category (conservative priority).

    ALTERNATE_DIRECT_RELATIONSHIP requires the *source* to emit other prerequisite
    edges (wrong target selection). Incoming edges to the gold target from unrelated
    sources are diagnostic only and do not count as alternate-direct.
    """
    # Direction confusion dominates when reverse is common and gold never appears.
    if reverse_freq >= 0.5 and gold_edge_freq == 0.0:
        return "REVERSED_DIRECTION"
    # Reachability without the direct edge.
    if alt_path_freq >= 0.5:
        if median_path_length is not None and median_path_length >= 3:
            return "SCOPE_OR_DIRECTNESS_ERROR"
        return "ALTERNATIVE_PATH"
    # Source repeatedly wires to other prerequisites instead of the gold target.
    if (
        alt_direct_from_source_freq >= 0.5
        and source_gold_target_recall < 0.5
        and source_has_any_outgoing_freq >= 0.5
    ):
        return "ALTERNATE_DIRECT_RELATIONSHIP"
    # Clean omission: no reverse, no path, source does not select alternate prereqs.
    if (
        gold_edge_freq == 0.0
        and reverse_freq == 0.0
        and alt_path_freq == 0.0
        and alt_direct_from_source_freq < 0.5
        and no_relationship_freq >= 0.5
    ):
        return "MISSING_DIRECT_PREREQUISITE"
    if reverse_freq > 0.0 and reverse_freq < 0.5 and gold_edge_freq == 0.0:
        if alt_direct_from_source_freq >= 0.5:
            return "ALTERNATE_DIRECT_RELATIONSHIP"
        return "MISSING_DIRECT_PREREQUISITE"
    if alt_direct_from_source_freq >= 0.5:
        return "ALTERNATE_DIRECT_RELATIONSHIP"
    if no_relationship_freq >= 0.5:
        return "MISSING_DIRECT_PREREQUISITE"
    return "UNKNOWN"


def source_centered_stats(
    source: str,
    gold_target: str,
    remapped_by_gen: list[list[tuple[str, str]]],
    gold_targets_for_source: set[str],
) -> dict[str, Any]:
    """Diagnostic source→prerequisite selection stats across generations."""
    src_n = normalize_topic(source)
    gold_tgt_n = normalize_topic(gold_target)
    gold_tgts_n = {normalize_topic(t) for t in gold_targets_for_source}
    per_gen_targets: list[list[str]] = []
    edge_counts: list[int] = []
    hit_gold = 0
    for remapped in remapped_by_gen:
        tgts = sorted({b for a, b in remapped if normalize_topic(a) == src_n})
        per_gen_targets.append(tgts)
        edge_counts.append(len(tgts))
        if gold_tgt_n in {normalize_topic(t) for t in tgts}:
            hit_gold += 1
    all_alts = [t for gen in per_gen_targets for t in gen if normalize_topic(t) != gold_tgt_n]
    alt_unique = sorted({normalize_topic(t) for t in all_alts})
    # Precision: among generated targets from source, fraction that are gold prereqs of source
    correct = 0
    total = 0
    for gen in per_gen_targets:
        for t in gen:
            total += 1
            if normalize_topic(t) in gold_tgts_n:
                correct += 1
    return {
        "SOURCE_EDGE_COUNT": _safe_rate(sum(edge_counts), len(edge_counts)) if edge_counts else 0.0,
        "SOURCE_ALTERNATIVE_TARGETS": alt_unique,
        "SOURCE_TARGET_DIVERSITY": len(alt_unique),
        "SOURCE_GOLD_TARGET_RECALL": _safe_rate(hit_gold, len(remapped_by_gen)),
        "TARGET_SELECTION_PRECISION": _safe_rate(correct, total),
        "per_generation_targets": per_gen_targets,
    }


def target_centered_stats(
    target: str,
    gold_sources: set[str],
    remapped_by_gen: list[list[tuple[str, str]]],
) -> dict[str, Any]:
    """Diagnostic who points at this prerequisite across generations."""
    tgt_n = normalize_topic(target)
    gold_src_n = {normalize_topic(s) for s in gold_sources}
    covered_gens = 0
    alt_prereq_sources: list[str] = []
    required_hits = 0
    required_slots = 0
    for remapped in remapped_by_gen:
        incoming = sorted({a for a, b in remapped if normalize_topic(b) == tgt_n})
        if incoming:
            covered_gens += 1
        for s in gold_src_n:
            required_slots += 1
            if any(normalize_topic(a) == s for a in incoming):
                required_hits += 1
        for a in incoming:
            if normalize_topic(a) not in gold_src_n:
                alt_prereq_sources.append(a)
    return {
        "TARGET_COVERAGE": _safe_rate(covered_gens, len(remapped_by_gen)),
        "ALTERNATIVE_PREREQUISITE_COUNT": len({normalize_topic(x) for x in alt_prereq_sources}),
        "REQUIRED_EDGE_RECALL_PER_TARGET": _safe_rate(required_hits, required_slots),
        "alternative_sources": sorted({normalize_topic(x) for x in alt_prereq_sources}),
    }


def analyze_pure_edge(
    *,
    case_id: str,
    example: EvalExample,
    source: str,
    target: str,
    graphs: list[GeneratedGraph],
    cmps: list[dict[str, Any]],
    remapped_by_gen: list[list[tuple[str, str]]],
    rep_map: dict[str, Any],
) -> dict[str, Any] | None:
    """Build one pure-relationship failure record, or None if not pure."""
    n = len(graphs)
    if n == 0:
        return None

    # Endpoints must be acceptably present in every generation (conservative).
    endpoint_ok_gens = 0
    exclude_votes: Counter[str] = Counter()
    per_gen_endpoint: list[dict[str, Any]] = []
    for g, cmp in zip(graphs, cmps):
        gate = endpoints_acceptably_present(
            source, target, example, g, rep_map=rep_map, cmp=cmp
        )
        per_gen_endpoint.append(gate)
        if gate["ok"]:
            endpoint_ok_gens += 1
        else:
            for r in gate["exclude_reasons"]:
                exclude_votes[r] += 1

    endpoint_presence = endpoint_ok_gens / n
    if endpoint_presence < FREQ_CONSISTENT:
        return None  # caller may log exclusion separately

    # Gold edge must be missing in all gens for the pure stable set.
    gold_hits = sum(1 for cmp in cmps if _edge_matched(source, target, cmp))
    gold_edge_freq = gold_hits / n
    if gold_edge_freq > 0.0:
        return None

    reverse_hits = 0
    path_hits = 0
    path_lengths: list[int] = []
    alt_direct_from_source_hits = 0
    source_outgoing_hits = 0
    no_rel_hits = 0
    generated_direction_counts: Counter[str] = Counter()
    alt_from_source: Counter[str] = Counter()
    alt_to_target: Counter[str] = Counter()

    src_n, tgt_n = normalize_topic(source), normalize_topic(target)
    for remapped in remapped_by_gen:
        has_gold = edge_present_in_remapped(source, target, remapped)
        has_rev = edge_present_in_remapped(target, source, remapped)
        plen = shortest_prerequisite_path_length(remapped, source, target)
        has_path = plen is not None and plen >= 1
        if has_path:
            path_hits += 1
            path_lengths.append(int(plen))
        if has_rev:
            reverse_hits += 1
            generated_direction_counts["reverse"] += 1
        if has_gold:
            generated_direction_counts["correct"] += 1
        if has_gold and has_rev:
            generated_direction_counts["both"] += 1

        src_outs = [
            b
            for a, b in remapped
            if normalize_topic(a) == src_n and normalize_topic(b) != src_n
        ]
        tgt_ins = [a for a, b in remapped if normalize_topic(b) == tgt_n]
        if src_outs:
            source_outgoing_hits += 1
        for b in src_outs:
            if normalize_topic(b) != tgt_n:
                alt_from_source[normalize_topic(b)] += 1
        for a in tgt_ins:
            if normalize_topic(a) != src_n:
                alt_to_target[normalize_topic(a)] += 1

        has_alt_from_source = any(normalize_topic(b) != tgt_n for b in src_outs)
        if has_alt_from_source:
            alt_direct_from_source_hits += 1

        if not has_gold and not has_rev and not has_path:
            no_rel_hits += 1
            generated_direction_counts["none"] += 1

    reverse_freq = reverse_hits / n
    alt_path_freq = path_hits / n
    alt_direct_freq = alt_direct_from_source_hits / n
    source_outgoing_freq = source_outgoing_hits / n
    no_relationship_freq = no_rel_hits / n
    median_path_length = None
    if path_lengths:
        ordered = sorted(path_lengths)
        median_path_length = float(ordered[len(ordered) // 2])

    gold_targets_for_source = {
        to for frm, to in example.required_dependency_list() if normalize_topic(frm) == src_n
    }
    gold_sources_for_target = {
        frm for frm, to in example.required_dependency_list() if normalize_topic(to) == tgt_n
    }
    src_stats = source_centered_stats(
        source, target, remapped_by_gen, gold_targets_for_source
    )
    tgt_stats = target_centered_stats(target, gold_sources_for_target, remapped_by_gen)

    category = classify_relationship_failure(
        gold_edge_freq=gold_edge_freq,
        reverse_freq=reverse_freq,
        alt_path_freq=alt_path_freq,
        alt_direct_from_source_freq=alt_direct_freq,
        no_relationship_freq=no_relationship_freq,
        median_path_length=median_path_length,
        source_gold_target_recall=float(src_stats["SOURCE_GOLD_TARGET_RECALL"]),
        source_has_any_outgoing_freq=source_outgoing_freq,
    )

    persistence = 1.0 - gold_edge_freq  # 1.0 when never generated
    impact = persistence  # unit impact per stable missing edge

    return {
        "case_id": case_id,
        "category": example.category,
        "difficulty": example.difficulty,
        "goal": example.goal,
        "source_topic": source,
        "target_topic": target,
        "gold_direction": [source, target],
        "generation_count": n,
        "generated_direction_counts": dict(generated_direction_counts),
        "endpoint_presence": endpoint_presence,
        "endpoint_evidence": per_gen_endpoint,
        "edge_generation_frequency": gold_edge_freq,
        "reverse_edge_frequency": reverse_freq,
        "alternative_direct_frequency": alt_direct_freq,
        "alternative_path_frequency": alt_path_freq,
        "no_relationship_frequency": no_relationship_freq,
        "median_path_length": median_path_length,
        "alternative_edges_from_source": [
            {"target": t, "gens": c} for t, c in alt_from_source.most_common()
        ],
        "alternative_edges_to_target": [
            {"source": s, "gens": c} for s, c in alt_to_target.most_common()
        ],
        "alternative_paths": {
            "frequency": alt_path_freq,
            "median_length": median_path_length,
            "present_majority": alt_path_freq >= 0.5,
        },
        "source_centered": src_stats,
        "target_centered": tgt_stats,
        "failure_category": category,
        "FAILURE_PERSISTENCE": persistence,
        "IMPACT": impact,
        "matching_mode": "curated_alias",
        "why_pure_relationship": (
            "Both endpoints consistently curated_alias-matched with EXACT/ALIAS "
            "representation; required direct edge never generated across runs."
        ),
    }


def run_pure_relationship_analysis(
    artifact_path: str | Path | None = None,
    *,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
    output_dir: str | Path | None = None,
    aligned_artifact_path: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Offline pure-relationship failure analysis. Returns (analysis_md, failures_json, pareto_md)."""
    target = Path(artifact_path) if artifact_path else find_latest_stability_artifact()
    if not target.is_file():
        raise FileNotFoundError(f"Artifact not found: {target}")

    payload = json.loads(target.read_text(encoding="utf-8"))
    systems = payload.get("systems") or {}
    if system not in systems:
        # Allow aligned replay systems
        if "representation_alignment" in systems and system == "synapse":
            pass
        elif system not in systems:
            raise ValueError(f"System {system!r} not in artifact; found {list(systems)}")

    system_block = systems.get(system) or systems.get("representation_alignment")
    if system_block is None and "synapse_baseline_replay" in systems:
        system_block = systems["synapse_baseline_replay"]
    if system_block is None:
        raise ValueError(f"No usable system block in {target}")

    rows_by_case = _group_rows(system_block if isinstance(system_block, dict) else {})
    # Prefer multi-gen stability under synapse
    if system in systems:
        rows_by_case = _group_rows(systems[system])

    n_gens = max((len(v) for v in rows_by_case.values()), default=0)
    if n_gens < 1:
        raise ValueError(f"No generations found in {target}")

    ds_stem = payload.get("dataset") or payload.get("dataset_version") or "learning_graph_quality_v1"
    ds_path = Path(dataset_path) if dataset_path else _REPO_ROOT / "data" / "eval" / f"{ds_stem}.jsonl"
    if not ds_path.is_file():
        ds_path = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    examples = {ex.id: ex for ex in load_dataset(ds_path)}
    rep_map = load_node_representation_map()

    # Optional aligned artifact (first-rep) for cross-check note only
    aligned_note: dict[str, Any] | None = None
    if aligned_artifact_path:
        ap = Path(aligned_artifact_path)
        if ap.is_file():
            aligned_note = {"path": str(ap), "used_for": "reference_only_not_scoring"}

    pure_failures: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    total_stable_missing = 0
    case_signatures: list[dict[str, Any]] = []

    for eid, rows in rows_by_case.items():
        ex = examples.get(eid)
        if not ex:
            continue
        adapted = adapt_example_for_edge_mode(
            ex, "edge_calibrated", topic_matching_mode="curated_alias"
        )
        graphs = [_graph_from_row(r) for r in rows]
        cmps = [compare_graphs(adapted, g) for g in graphs]
        remapped_by_gen = [remap_dependencies_to_gold(g, adapted) for g in graphs]
        n = len(graphs)
        if n == 0:
            continue

        required_deps = adapted.required_dependency_list()
        case_pures: list[dict[str, Any]] = []

        for frm, to in required_deps:
            gold_hits = sum(1 for cmp in cmps if _edge_matched(frm, to, cmp))
            if gold_hits > 0:
                continue
            total_stable_missing += 1

            # Endpoint frequency under curated_alias (matched topics)
            src_freq = sum(1 for cmp in cmps if _topic_present(frm, cmp)) / n
            tgt_freq = sum(1 for cmp in cmps if _topic_present(to, cmp)) / n

            # Strict acceptable-rep gate
            ok_gens = 0
            exclude_reasons: Counter[str] = Counter()
            for g, cmp in zip(graphs, cmps):
                gate = endpoints_acceptably_present(
                    frm, to, adapted, g, rep_map=rep_map, cmp=cmp
                )
                if gate["ok"]:
                    ok_gens += 1
                else:
                    for r in gate["exclude_reasons"]:
                        exclude_reasons[r] += 1
            ep = ok_gens / n

            if src_freq < FREQ_CONSISTENT or tgt_freq < FREQ_CONSISTENT or ep < FREQ_CONSISTENT:
                reason = "endpoint_or_representation"
                if src_freq < FREQ_CONSISTENT and tgt_freq < FREQ_CONSISTENT:
                    reason = "both_endpoints_inconsistent"
                elif src_freq < FREQ_CONSISTENT:
                    reason = "source_endpoint_inconsistent"
                elif tgt_freq < FREQ_CONSISTENT:
                    reason = "target_endpoint_inconsistent"
                elif ep < FREQ_CONSISTENT:
                    reason = "representation_not_exact_or_alias"
                excluded.append(
                    {
                        "case_id": eid,
                        "source_topic": frm,
                        "target_topic": to,
                        "exclude_reason": reason,
                        "source_frequency": src_freq,
                        "target_frequency": tgt_freq,
                        "acceptable_endpoint_presence": ep,
                        "exclude_detail": dict(exclude_reasons),
                    }
                )
                continue

            rec = analyze_pure_edge(
                case_id=eid,
                example=adapted,
                source=frm,
                target=to,
                graphs=graphs,
                cmps=cmps,
                remapped_by_gen=remapped_by_gen,
                rep_map=rep_map,
            )
            if rec is None:
                excluded.append(
                    {
                        "case_id": eid,
                        "source_topic": frm,
                        "target_topic": to,
                        "exclude_reason": "failed_pure_gate",
                        "source_frequency": src_freq,
                        "target_frequency": tgt_freq,
                        "acceptable_endpoint_presence": ep,
                    }
                )
                continue
            pure_failures.append(rec)
            case_pures.append(rec)

        if case_pures:
            cats = Counter(r["failure_category"] for r in case_pures)
            case_signatures.append(
                {
                    "case_id": eid,
                    "category": adapted.category,
                    "difficulty": adapted.difficulty,
                    "goal": adapted.goal,
                    "relationship_failure_count": len(case_pures),
                    "stable_relationship_failure_count": sum(
                        1 for r in case_pures if r["FAILURE_PERSISTENCE"] >= 1.0
                    ),
                    "direction_failure_count": cats.get("REVERSED_DIRECTION", 0),
                    "alternative_path_count": cats.get("ALTERNATIVE_PATH", 0),
                    "direct_omission_count": cats.get("MISSING_DIRECT_PREREQUISITE", 0),
                    "source_target_confusion_count": cats.get(
                        "ALTERNATE_DIRECT_RELATIONSHIP", 0
                    ),
                    "scope_error_count": cats.get("SCOPE_OR_DIRECTNESS_ERROR", 0),
                }
            )

    n_pure = len(pure_failures)
    cat_counts = Counter(r["failure_category"] for r in pure_failures)
    breakdown = [
        {
            "failure_type": c,
            "count": cat_counts.get(c, 0),
            "rate": _safe_rate(cat_counts.get(c, 0), n_pure),
        }
        for c in FAILURE_CATEGORIES
    ]

    # Aggregate frequencies
    freq_agg = {
        "gold_edge_frequency_mean": _safe_rate(
            sum(r["edge_generation_frequency"] for r in pure_failures), n_pure
        ),
        "reverse_frequency_mean": _safe_rate(
            sum(r["reverse_edge_frequency"] for r in pure_failures), n_pure
        ),
        "alternative_direct_frequency_mean": _safe_rate(
            sum(r["alternative_direct_frequency"] for r in pure_failures), n_pure
        ),
        "alternative_path_frequency_mean": _safe_rate(
            sum(r["alternative_path_frequency"] for r in pure_failures), n_pure
        ),
        "no_relationship_frequency_mean": _safe_rate(
            sum(r["no_relationship_frequency"] for r in pure_failures), n_pure
        ),
    }

    # Opportunity / conditional recall on the pure set:
    # by construction endpoints are present and gold edge freq=0 → opportunity=1, cond_recall=0
    edge_opportunity = {
        "PURE_SET_EDGE_OPPORTUNITY_RATE": 1.0 if n_pure else 0.0,
        "PURE_SET_CONDITIONAL_EDGE_RECALL": 0.0,
        "note": (
            "Pure set requires both endpoints present and gold edge never generated; "
            "conditional recall on this set is 0 by construction."
        ),
    }

    metrics = {
        "PURE_RELATIONSHIP_FAILURE_COUNT": n_pure,
        "PURE_RELATIONSHIP_FAILURE_RATE": _safe_rate(n_pure, total_stable_missing),
        "TOTAL_STABLE_MISSING_EDGES": total_stable_missing,
        "REVERSED_DIRECTION_RATE": _safe_rate(cat_counts.get("REVERSED_DIRECTION", 0), n_pure),
        "ALTERNATE_DIRECT_RELATIONSHIP_RATE": _safe_rate(
            cat_counts.get("ALTERNATE_DIRECT_RELATIONSHIP", 0), n_pure
        ),
        "ALTERNATIVE_PATH_RATE": _safe_rate(cat_counts.get("ALTERNATIVE_PATH", 0), n_pure),
        "MISSING_DIRECT_PREREQUISITE_RATE": _safe_rate(
            cat_counts.get("MISSING_DIRECT_PREREQUISITE", 0), n_pure
        ),
        "SCOPE_OR_DIRECTNESS_ERROR_RATE": _safe_rate(
            cat_counts.get("SCOPE_OR_DIRECTNESS_ERROR", 0), n_pure
        ),
        "SOURCE_TARGET_CONFUSION_RATE": _safe_rate(
            cat_counts.get("ALTERNATE_DIRECT_RELATIONSHIP", 0), n_pure
        ),
        "STABLE_RELATIONSHIP_FAILURE_RATE": _safe_rate(
            sum(1 for r in pure_failures if r["FAILURE_PERSISTENCE"] >= 1.0), n_pure
        ),
        **freq_agg,
        **edge_opportunity,
    }

    # Domain / difficulty aggregation
    by_domain: dict[str, int] = Counter(r["category"] for r in pure_failures)
    by_difficulty: dict[str, int] = Counter(r["difficulty"] for r in pure_failures)

    # Source / target centered leaders
    source_misses: Counter[str] = Counter()
    target_misses: Counter[str] = Counter()
    for r in pure_failures:
        source_misses[f"{r['case_id']}::{r['source_topic']}"] += 1
        target_misses[f"{r['case_id']}::{r['target_topic']}"] += 1
        # also aggregate by topic name across cases
        source_misses[f"topic::{r['source_topic']}"] += 1
        target_misses[f"topic::{r['target_topic']}"] += 1

    top_sources = [
        {"key": k, "count": c}
        for k, c in source_misses.most_common()
        if k.startswith("topic::")
    ][:15]
    top_targets = [
        {"key": k, "count": c}
        for k, c in target_misses.most_common()
        if k.startswith("topic::")
    ][:15]

    # Pareto
    pareto_sorted = sorted(
        pure_failures,
        key=lambda r: (-r["IMPACT"], -r["FAILURE_PERSISTENCE"], r["case_id"], r["source_topic"]),
    )
    pareto_rows: list[dict[str, Any]] = []
    cum = 0.0
    for i, r in enumerate(pareto_sorted, start=1):
        cum += r["IMPACT"]
        pareto_rows.append(
            {
                "rank": i,
                "case_id": r["case_id"],
                "source": r["source_topic"],
                "target": r["target_topic"],
                "frequency": r["edge_generation_frequency"],
                "failure_type": r["failure_category"],
                "alternative_path": r["alternative_paths"]["present_majority"],
                "reverse_frequency": r["reverse_edge_frequency"],
                "impact": r["IMPACT"],
                "cumulative_impact": cum,
                "cumulative_pct": _safe_rate(cum, n_pure),
            }
        )

    diagnosis, diagnosis_rationale = _diagnose(metrics, cat_counts, n_pure, freq_agg)

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    failures_path = out_dir / f"{ts}_pure_relationship_failures.json"
    analysis_md = out_dir / f"{ts}_pure_relationship_failure_analysis.md"
    pareto_md = out_dir / f"{ts}_pure_relationship_failure_pareto.md"

    payload_out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "no_new_llm_calls": True,
        "source_artifact": str(target),
        "aligned_artifact_note": aligned_note,
        "dataset": str(ds_path),
        "system": system,
        "matching_mode": "curated_alias",
        "edge_mode": "edge_calibrated",
        "purity_gate": {
            "endpoint_frequency_min": FREQ_CONSISTENT,
            "acceptable_representation": sorted(ACCEPTABLE_REP),
            "gold_edge_frequency_max": 0.0,
            "excluded_from_pure_set": sorted(UNACCEPTABLE_REP),
        },
        "metrics": metrics,
        "failure_breakdown": breakdown,
        "by_domain": dict(by_domain),
        "by_difficulty": dict(by_difficulty),
        "top_source_selection_failures": top_sources,
        "top_target_prerequisite_misses": top_targets,
        "case_signatures": case_signatures,
        "excluded_candidates": excluded,
        "diagnosis": {"code": diagnosis, "rationale": diagnosis_rationale},
        "failures": pure_failures,
        "pareto": pareto_rows,
    }
    failures_path.write_text(
        json.dumps(payload_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    analysis_md.write_text(_render_analysis_md(payload_out), encoding="utf-8")
    pareto_md.write_text(_render_pareto_md(payload_out), encoding="utf-8")
    return analysis_md, failures_path, pareto_md


def _diagnose(
    metrics: dict[str, Any],
    cat_counts: Counter[str],
    n_pure: int,
    freq_agg: dict[str, float],
) -> tuple[str, str]:
    if n_pure == 0:
        return (
            "INSUFFICIENT_EVIDENCE",
            "No edges passed the strict pure-relationship gate.",
        )
    rates = {c: _safe_rate(cat_counts.get(c, 0), n_pure) for c in FAILURE_CATEGORIES}
    top = max(FAILURE_CATEGORIES, key=lambda c: rates[c])
    top_rate = rates[top]
    rev = freq_agg["reverse_frequency_mean"]
    path = freq_agg["alternative_path_frequency_mean"]
    none = freq_agg["no_relationship_frequency_mean"]
    alt = freq_agg["alternative_direct_frequency_mean"]

    rationale = (
        f"n_pure={n_pure}, top={top} ({top_rate:.2f}), "
        f"mean_reverse={rev:.2f}, mean_path={path:.2f}, "
        f"mean_alt_direct={alt:.2f}, mean_none={none:.2f}."
    )

    # Dominant single cause if >= 50%
    if top_rate >= 0.5:
        mapping = {
            "REVERSED_DIRECTION": "RELATIONSHIP_DIRECTION_PROBLEM",
            "ALTERNATE_DIRECT_RELATIONSHIP": "RELATIONSHIP_SCOPE_PROBLEM",
            "ALTERNATIVE_PATH": "GRAPH_REASONING_PROBLEM",
            "MISSING_DIRECT_PREREQUISITE": "RELATIONSHIP_THRESHOLD_PROBLEM",
            "SCOPE_OR_DIRECTNESS_ERROR": "RELATIONSHIP_SCOPE_PROBLEM",
            "UNKNOWN": "INSUFFICIENT_EVIDENCE",
        }
        return mapping.get(top, "MIXED"), rationale

    # Mixed: several material contributors (>= 20% each, at least two)
    material = [c for c in FAILURE_CATEGORIES if rates[c] >= 0.2]
    if len(material) >= 2:
        return "MIXED", rationale + f" Material categories: {material}."

    # Soft mapping when no majority
    if rev >= 0.3:
        return "RELATIONSHIP_DIRECTION_PROBLEM", rationale
    if none >= 0.4 and path < 0.2:
        return "RELATIONSHIP_THRESHOLD_PROBLEM", rationale
    if path >= 0.3:
        return "GRAPH_REASONING_PROBLEM", rationale
    if alt >= 0.3:
        return "RELATIONSHIP_SCOPE_PROBLEM", rationale
    return "MIXED", rationale


def _pick_examples(failures: list[dict[str, Any]], category: str, n: int) -> list[dict[str, Any]]:
    return [f for f in failures if f["failure_category"] == category][:n]


def _render_analysis_md(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# Pure Relationship Failure Analysis",
        "",
        f"- Source: `{payload['source_artifact']}`",
        f"- NO_NEW_LLM_CALLS: `{payload['no_new_llm_calls']}`",
        f"- Matching: `{payload['matching_mode']}` / `{payload['edge_mode']}`",
        f"- Diagnosis: **{payload['diagnosis']['code']}**",
        f"- Rationale: {payload['diagnosis']['rationale']}",
        "",
        "## Pure relationship set",
        "",
        f"- PURE_RELATIONSHIP_FAILURE_COUNT: **{m['PURE_RELATIONSHIP_FAILURE_COUNT']}**",
        f"- TOTAL_STABLE_MISSING_EDGES: **{m['TOTAL_STABLE_MISSING_EDGES']}**",
        f"- PURE_RELATIONSHIP_FAILURE_RATE: **{m['PURE_RELATIONSHIP_FAILURE_RATE']:.3f}**",
        f"- STABLE_RELATIONSHIP_FAILURE_RATE: **{m['STABLE_RELATIONSHIP_FAILURE_RATE']:.3f}**",
        f"- Excluded candidates (endpoint/rep): **{len(payload.get('excluded_candidates') or [])}**",
        "",
        "## Failure breakdown",
        "",
        "| Failure Type | Count | Rate |",
        "| --- | ---: | ---: |",
    ]
    for row in payload["failure_breakdown"]:
        lines.append(
            f"| {row['failure_type']} | {row['count']} | {row['rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Frequency (means over pure set)",
            "",
            f"- gold edge: {m['gold_edge_frequency_mean']:.3f}",
            f"- reverse: {m['reverse_frequency_mean']:.3f}",
            f"- alternative direct: {m['alternative_direct_frequency_mean']:.3f}",
            f"- alternative path: {m['alternative_path_frequency_mean']:.3f}",
            f"- no relationship: {m['no_relationship_frequency_mean']:.3f}",
            "",
            "## Opportunity (pure set)",
            "",
            f"- EDGE_OPPORTUNITY_RATE: {m['PURE_SET_EDGE_OPPORTUNITY_RATE']:.3f}",
            f"- CONDITIONAL_EDGE_RECALL: {m['PURE_SET_CONDITIONAL_EDGE_RECALL']:.3f}",
            f"- Note: {m.get('note', '')}",
            "",
            "## Domain / difficulty",
            "",
            f"- By domain: `{payload.get('by_domain')}`",
            f"- By difficulty: `{payload.get('by_difficulty')}`",
            "",
            "## Source-centered (top topic keys)",
            "",
        ]
    )
    for row in (payload.get("top_source_selection_failures") or [])[:8]:
        lines.append(f"- {row['key']}: {row['count']}")
    lines.extend(["", "## Target-centered (top topic keys)", ""])
    for row in (payload.get("top_target_prerequisite_misses") or [])[:8]:
        lines.append(f"- {row['key']}: {row['count']}")

    failures = payload.get("failures") or []
    sections = [
        ("MISSING_DIRECT_PREREQUISITE", 3, "Clean missing-direct-prerequisite"),
        ("REVERSED_DIRECTION", 2, "Direction"),
        ("ALTERNATE_DIRECT_RELATIONSHIP", 2, "Alternate-direct-relationship"),
        ("ALTERNATIVE_PATH", 2, "Alternative-path"),
        ("SCOPE_OR_DIRECTNESS_ERROR", 2, "Scope/directness"),
        ("UNKNOWN", 2, "Unknown"),
    ]
    lines.extend(["", "## Representative cases", ""])
    for cat, n, title in sections:
        picks = _pick_examples(failures, cat, n)
        lines.append(f"### {title} (`{cat}`)")
        lines.append("")
        if not picks:
            lines.append(f"_No cases in this category (n=0)._")
            lines.append("")
            continue
        for rec in picks:
            lines.extend(_format_case(rec))
    # Mixed cases: cases with >1 failure category in signature
    mixed_cases = [
        s
        for s in payload.get("case_signatures") or []
        if sum(
            1
            for k in (
                "direction_failure_count",
                "alternative_path_count",
                "direct_omission_count",
                "source_target_confusion_count",
                "scope_error_count",
            )
            if s.get(k, 0) > 0
        )
        >= 2
    ][:2]
    lines.extend(["### Mixed cases (multiple failure modes in one learning goal)", ""])
    if not mixed_cases:
        lines.append("_Fewer than 2 mixed case signatures._")
        lines.append("")
    else:
        for s in mixed_cases:
            lines.append(f"#### {s['case_id']}")
            lines.append("")
            lines.append(f"**Objective:** {s['goal']}")
            lines.append(
                f"- relationship_failure_count={s['relationship_failure_count']}, "
                f"direction={s['direction_failure_count']}, "
                f"alt_path={s['alternative_path_count']}, "
                f"omission={s['direct_omission_count']}, "
                f"confusion={s['source_target_confusion_count']}"
            )
            lines.append("")
    return "\n".join(lines) + "\n"


def _format_case(rec: dict[str, Any]) -> list[str]:
    return [
        f"#### {rec['case_id']}: {rec['source_topic']} → {rec['target_topic']}",
        "",
        f"**Objective:** {rec['goal']}",
        f"- Gold dependency: `{rec['gold_direction']}`",
        f"- Endpoint presence: {rec['endpoint_presence']:.2f}",
        f"- Gold / reverse / alt-path / none freqs: "
        f"{rec['edge_generation_frequency']:.2f} / "
        f"{rec['reverse_edge_frequency']:.2f} / "
        f"{rec['alternative_path_frequency']:.2f} / "
        f"{rec['no_relationship_frequency']:.2f}",
        f"- Failure category: **{rec['failure_category']}**",
        f"- Why pure: {rec['why_pure_relationship']}",
        f"- Alt from source: `{rec['alternative_edges_from_source'][:5]}`",
        f"- Source gold-target recall: {rec['source_centered']['SOURCE_GOLD_TARGET_RECALL']:.2f}",
        f"- Target selection precision: {rec['source_centered']['TARGET_SELECTION_PRECISION']:.2f}",
        "",
    ]


def _render_pareto_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Pure Relationship Failure Pareto",
        "",
        f"- Source: `{payload['source_artifact']}`",
        f"- Pure failures: {payload['metrics']['PURE_RELATIONSHIP_FAILURE_COUNT']}",
        f"- Diagnosis: **{payload['diagnosis']['code']}**",
        "",
        "| Rank | Case | Source | Target | Freq | Type | Alt path | Reverse | Impact | Cum % |",
        "| ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload.get("pareto") or []:
        lines.append(
            f"| {row['rank']} | {row['case_id']} | {row['source']} | {row['target']} | "
            f"{row['frequency']:.2f} | {row['failure_type']} | {row['alternative_path']} | "
            f"{row['reverse_frequency']:.2f} | {row['impact']:.2f} | {row['cumulative_pct']:.2f} |"
        )
    return "\n".join(lines) + "\n"
