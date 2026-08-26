"""Persistent failure cluster attribution (offline, evaluation only).

Attributes STABLE_MISSING / NEVER_GENERATED required edges to endpoint vs
relationship root causes. Makes no LLM calls and does not change generation.
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
from app.evaluation.schemas import EvalExample, GeneratedGraph

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "failure_analysis"
DEFAULT_BENCH = _REPO_ROOT / "results" / "benchmarks"

# Scoring correctness fix (not matching calibration): compare_graphs now enforces
# one-to-one unique required-edge matches so missing_required_edge_rate ∈ [0, 1].
SCORING_FIX = {
    "id": "required_edge_one_to_one_v1",
    "kind": "metric_bug_fix",
    "not": "evaluation_calibration_change",
    "summary": (
        "Duplicate generated edges mapping to the same required gold edge no longer "
        "inflate matched counts; extras after the first match count as invalid extras. "
        "Ensures 0 <= missing_required_edge_rate <= 1 and unique matches ≤ unique required."
    ),
}

FREQ_CONSISTENT = 0.8
REP_MISMATCH_STATUSES = frozenset(
    {
        "GRANULARITY_VARIANT",
        "DECOMPOSED",
        "ABSTRACTED",
        "RELATED_BUT_DISTINCT",
        "UNKNOWN",
    }
)
REP_SUBTYPE_MAP = {
    "GRANULARITY_VARIANT": "GRANULARITY_MISMATCH",
    "DECOMPOSED": "DECOMPOSITION_MISMATCH",
    "ABSTRACTED": "ABSTRACTION_MISMATCH",
    "RELATED_BUT_DISTINCT": "RELATED_BUT_DISTINCT",
    "UNKNOWN": "UNMATCHED_REPLACEMENT",
    "MISSING": "TRUE_ABSENCE",
}

PRIMARY_ATTRIBUTIONS = (
    "SOURCE_NEVER_PRESENT",
    "TARGET_NEVER_PRESENT",
    "BOTH_ENDPOINTS_NEVER_PRESENT",
    "ENDPOINT_REPRESENTATION_MISMATCH",
    "BOTH_ENDPOINTS_PRESENT_EDGE_OMISSION",
    "MIXED_ENDPOINT_AVAILABILITY",
    "UNRESOLVED",
)

NODE_ENDPOINT_ATTRIBUTIONS = frozenset(
    {
        "SOURCE_NEVER_PRESENT",
        "TARGET_NEVER_PRESENT",
        "BOTH_ENDPOINTS_NEVER_PRESENT",
        "ENDPOINT_REPRESENTATION_MISMATCH",
    }
)


def find_latest_stability_artifact(bench_dir: Path | None = None) -> Path:
    root = Path(bench_dir) if bench_dir else DEFAULT_BENCH
    candidates = sorted(root.glob("*_quality_stability_*.json"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No *_quality_stability_*.json under {root}")
    return candidates[0]


def _norm_edge(frm: str, to: str) -> tuple[str, str]:
    return (normalize_topic(frm), normalize_topic(to))


def _edge_key(frm: str, to: str) -> str:
    return f"{frm}→{to}"


def has_prerequisite_path(
    dependencies: list[tuple[str, str]] | list[list[str]],
    source: str,
    target: str,
    *,
    topic_match: dict[str, str] | None = None,
) -> bool:
    """True if a directed path source → … → target exists.

    Synapse edge semantics: ``[from, to]`` means *from requires to*
    (``to`` is the prerequisite). Walking ``from → to`` follows the
    prerequisite chain toward foundations. A path from the dependent
    endpoint to the prerequisite endpoint means alternative reachability
    exists even if the direct required edge is missing.
    """
    def canon(t: str) -> str:
        if topic_match and t in topic_match:
            return normalize_topic(topic_match[t])
        return normalize_topic(t)

    src_n, tgt_n = canon(source), canon(target)
    if src_n == tgt_n:
        return True
    adj: dict[str, set[str]] = defaultdict(set)
    for dep in dependencies:
        if len(dep) != 2:
            continue
        adj[canon(str(dep[0]))].add(canon(str(dep[1])))
    seen = {src_n}
    stack = [src_n]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, ()):
            if nxt == tgt_n:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _group_rows(system_block: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in system_block.get("example_results") or []:
        eid = str(row.get("example_id") or "")
        if eid:
            by_case[eid].append(row)
    for eid in by_case:
        by_case[eid].sort(key=lambda r: int(r.get("repetition") or r.get("generation_index") or 0))
    return dict(by_case)


def _topic_present(gold: str, cmp: dict[str, Any]) -> bool:
    matched = {normalize_topic(t) for t in cmp.get("matched_required_topics") or []}
    return normalize_topic(gold) in matched


def _edge_matched(frm: str, to: str, cmp: dict[str, Any]) -> bool:
    missing = {_norm_edge(a, b) for a, b in (tuple(e) for e in cmp.get("missing_dependencies") or [])}
    reversed_gold = set()
    for item in cmp.get("reversed_dependencies") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            gold = item[1]
            if isinstance(gold, (list, tuple)) and len(gold) == 2:
                reversed_gold.add(_norm_edge(str(gold[0]), str(gold[1])))
    en = _norm_edge(frm, to)
    return en not in missing and en not in reversed_gold


def _rep_subtype(status: str) -> str:
    return REP_SUBTYPE_MAP.get(status, "UNKNOWN")


def _consistent_representation_mismatch(
    gold: str,
    example: EvalExample,
    graphs: list[GeneratedGraph],
    *,
    rep_map: dict[str, Any],
) -> dict[str, Any] | None:
    """Return mismatch summary if gold is never alias-matched but consistently replaced."""
    labels: list[dict[str, Any]] = []
    for g in graphs:
        # Only inspect representation when curated_alias match fails for this graph.
        cmp = compare_graphs(example, g)
        if _topic_present(gold, cmp):
            return None
        lab = classify_gold_topic_representation(gold, example, g, rep_map=rep_map)
        labels.append(lab)
    mismatch_labs = [x for x in labels if x.get("status") in REP_MISMATCH_STATUSES]
    if len(mismatch_labs) < max(1, (len(graphs) + 1) // 2):
        # Need mismatch evidence in a majority of generations.
        return None
    # Prefer a concrete subtype over UNKNOWN when any generation has one.
    statuses = [x["status"] for x in mismatch_labs]
    preferred = next(
        (s for s in statuses if s in {"GRANULARITY_VARIANT", "DECOMPOSED", "ABSTRACTED", "RELATED_BUT_DISTINCT"}),
        statuses[0],
    )
    if preferred == "UNKNOWN":
        # Require at least one candidate replacement title
        cands = [c for x in mismatch_labs for c in (x.get("candidates") or [])]
        if not cands:
            return None
    candidates: list[str] = []
    for x in mismatch_labs:
        if x.get("status") == preferred:
            candidates.extend(x.get("candidates") or [])
    return {
        "status": preferred,
        "subtype": _rep_subtype(preferred),
        "candidates": sorted({normalize_topic(c) for c in candidates}),
        "n_gens_with_mismatch": len(mismatch_labs),
    }


def attribute_stable_missing_edge(
    *,
    source: str,
    target: str,
    source_freq: float,
    target_freq: float,
    source_rep: dict[str, Any] | None,
    target_rep: dict[str, Any] | None,
) -> str:
    """Deterministic primary attribution for one NEVER_GENERATED required edge."""
    src_suf = source_freq >= FREQ_CONSISTENT
    tgt_suf = target_freq >= FREQ_CONSISTENT
    src_never = source_freq <= 0.0
    tgt_never = target_freq <= 0.0

    if src_suf and tgt_suf:
        return "BOTH_ENDPOINTS_PRESENT_EDGE_OMISSION"

    # Representation mismatch takes precedence over pure absence when evidence exists.
    if src_never and source_rep:
        return "ENDPOINT_REPRESENTATION_MISMATCH"
    if tgt_never and target_rep:
        return "ENDPOINT_REPRESENTATION_MISMATCH"
    if src_never and tgt_never and (source_rep or target_rep):
        return "ENDPOINT_REPRESENTATION_MISMATCH"

    if src_never and tgt_suf:
        return "SOURCE_NEVER_PRESENT"
    if tgt_never and src_suf:
        return "TARGET_NEVER_PRESENT"
    if src_never and tgt_never:
        return "BOTH_ENDPOINTS_NEVER_PRESENT"

    if (0.0 < source_freq < FREQ_CONSISTENT) or (0.0 < target_freq < FREQ_CONSISTENT):
        return "MIXED_ENDPOINT_AVAILABILITY"

    return "UNRESOLVED"


def run_persistent_failure_attribution(
    artifact_path: str | Path | None = None,
    *,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
    output_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Analyze stable missing edges; write cluster JSON, MD, and Pareto MD."""
    target = Path(artifact_path) if artifact_path else find_latest_stability_artifact()
    if not target.is_file():
        raise FileNotFoundError(f"Artifact not found: {target}")

    payload = json.loads(target.read_text(encoding="utf-8"))
    systems = payload.get("systems") or {}
    if system not in systems:
        raise ValueError(f"System {system!r} not in artifact; found {list(systems)}")

    rows_by_case = _group_rows(systems[system])
    n_gens = max((len(v) for v in rows_by_case.values()), default=0)
    if n_gens < 2:
        raise ValueError(
            f"Artifact needs ≥2 generations per case for persistent attribution; observed max={n_gens}"
        )

    ds_stem = payload.get("dataset") or payload.get("dataset_version") or "learning_graph_quality_v1"
    ds_path = Path(dataset_path) if dataset_path else _REPO_ROOT / "data" / "eval" / f"{ds_stem}.jsonl"
    if not ds_path.is_file():
        ds_path = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    examples = {ex.id: ex for ex in load_dataset(ds_path)}
    rep_map = load_node_representation_map()

    stable_missing_edges: list[dict[str, Any]] = []
    never_present_topics: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    alt_path_hits = 0
    pure_relationship: list[dict[str, Any]] = []

    for eid, rows in rows_by_case.items():
        ex = examples.get(eid)
        if not ex:
            continue
        adapted = adapt_example_for_edge_mode(
            ex, "edge_calibrated", topic_matching_mode="curated_alias"
        )
        graphs = [_graph_from_row(r) for r in rows]
        cmps = [compare_graphs(adapted, g) for g in graphs]
        n = len(graphs)
        if n == 0:
            continue

        # Topic frequencies (curated_alias)
        topic_freq: dict[str, float] = {}
        topic_rep: dict[str, Any | None] = {}
        for gold in adapted.required_topic_list():
            hits = sum(1 for cmp in cmps if _topic_present(gold, cmp))
            freq = hits / n
            topic_freq[gold] = freq
            if freq <= 0.0:
                topic_rep[gold] = _consistent_representation_mismatch(
                    gold, adapted, graphs, rep_map=rep_map
                )
            else:
                topic_rep[gold] = None

        # Edge frequencies
        edge_freq: dict[tuple[str, str], float] = {}
        required_deps = adapted.required_dependency_list()
        for frm, to in required_deps:
            hits = sum(1 for cmp in cmps if _edge_matched(frm, to, cmp))
            edge_freq[(frm, to)] = hits / n

        case_stable: list[dict[str, Any]] = []
        for frm, to in required_deps:
            if edge_freq[(frm, to)] > 0.0:
                continue
            src_f = topic_freq.get(frm, 0.0)
            tgt_f = topic_freq.get(to, 0.0)
            src_r = topic_rep.get(frm)
            tgt_r = topic_rep.get(to)
            primary = attribute_stable_missing_edge(
                source=frm,
                target=to,
                source_freq=src_f,
                target_freq=tgt_f,
                source_rep=src_r,
                target_rep=tgt_r,
            )

            # Alternative path: majority of generations have directed prerequisite reachability.
            path_hits = 0
            for g in graphs:
                gen_to_canon: dict[str, str] = {}
                for t in g.topics:
                    m = match_topic(t, adapted)
                    if m is not None:
                        gen_to_canon[t] = m
                remapped = [
                    (gen_to_canon.get(a, a), gen_to_canon.get(b, b))
                    for a, b in g.dependencies
                ]
                if has_prerequisite_path(remapped, frm, to):
                    path_hits += 1
            alt_path = path_hits >= max(1, (n + 1) // 2)
            if alt_path:
                alt_path_hits += 1

            rec = {
                "case_id": eid,
                "category": adapted.category,
                "goal": adapted.goal,
                "from": frm,
                "to": to,
                "edge_key": _edge_key(frm, to),
                "primary_attribution": primary,
                "source_frequency": src_f,
                "target_frequency": tgt_f,
                "source_representation": src_r,
                "target_representation": tgt_r,
                "alternative_path_present": alt_path,
                "alternative_path_gens": path_hits,
                "n_generations": n,
            }
            case_stable.append(rec)
            stable_missing_edges.append(rec)
            if primary == "BOTH_ENDPOINTS_PRESENT_EDGE_OMISSION":
                pure_relationship.append(rec)

        # Never-present topics + gold degree
        gold_deps = required_deps
        for gold, freq in topic_freq.items():
            if freq > 0.0:
                continue
            src_deg = sum(1 for a, _b in gold_deps if normalize_topic(a) == normalize_topic(gold))
            tgt_deg = sum(1 for _a, b in gold_deps if normalize_topic(b) == normalize_topic(gold))
            impact_edges = [
                e
                for e in case_stable
                if normalize_topic(e["from"]) == normalize_topic(gold)
                or normalize_topic(e["to"]) == normalize_topic(gold)
            ]
            source_impact = sum(
                1 for e in impact_edges if normalize_topic(e["from"]) == normalize_topic(gold)
            )
            target_impact = sum(
                1 for e in impact_edges if normalize_topic(e["to"]) == normalize_topic(gold)
            )
            never_present_topics.append(
                {
                    "case_id": eid,
                    "category": adapted.category,
                    "topic": gold,
                    "gold_source_degree": src_deg,
                    "gold_target_degree": tgt_deg,
                    "gold_degree": src_deg + tgt_deg,
                    "source_impact": source_impact,
                    "target_impact": target_impact,
                    "total_impact": source_impact + target_impact,
                    "representation": topic_rep.get(gold),
                }
            )

        case_summaries.append(
            {
                "case_id": eid,
                "goal": adapted.goal,
                "category": adapted.category,
                "gold_topics": list(adapted.required_topic_list()),
                "gold_dependencies": [list(d) for d in required_deps],
                "n_generations": n,
                "generations": [
                    {
                        "generation_index": int(rows[i].get("generation_index", rows[i].get("repetition") or i)),
                        "seed": (rows[i].get("generation_meta") or {}).get("seed", rows[i].get("seed")),
                        "topics": list(graphs[i].topics),
                        "dependencies": [list(d) for d in graphs[i].dependencies],
                    }
                    for i in range(n)
                ],
                "topic_frequencies": topic_freq,
                "stable_missing_edges": case_stable,
                "never_present_topics": [t for t in never_present_topics if t["case_id"] == eid],
                "attribution_counts": dict(Counter(e["primary_attribution"] for e in case_stable)),
            }
        )

    total_stable = len(stable_missing_edges)
    attr_counts = Counter(e["primary_attribution"] for e in stable_missing_edges)
    attribution_summary = [
        {
            "primary_attribution": a,
            "count": attr_counts.get(a, 0),
            "rate": (attr_counts.get(a, 0) / total_stable) if total_stable else 0.0,
        }
        for a in PRIMARY_ATTRIBUTIONS
    ]
    node_endpoint_n = sum(attr_counts[a] for a in NODE_ENDPOINT_ATTRIBUTIONS)
    pure_n = attr_counts.get("BOTH_ENDPOINTS_PRESENT_EDGE_OMISSION", 0)
    node_endpoint_rate = (node_endpoint_n / total_stable) if total_stable else 0.0
    pure_relationship_rate = (pure_n / total_stable) if total_stable else 0.0

    # Pareto by never-present topic impact
    never_sorted = sorted(
        never_present_topics,
        key=lambda t: (-t["total_impact"], t["case_id"], t["topic"]),
    )
    pareto_rows: list[dict[str, Any]] = []
    cum = 0
    for i, t in enumerate(never_sorted, start=1):
        cum += t["total_impact"]
        pareto_rows.append(
            {
                "rank": i,
                "case_id": t["case_id"],
                "topic": t["topic"],
                "attribution": "NEVER_PRESENT_TOPIC",
                "source_impact": t["source_impact"],
                "target_impact": t["target_impact"],
                "stable_missing_edges_explained": t["total_impact"],
                "pct_of_stable_missing": (t["total_impact"] / total_stable) if total_stable else 0.0,
                "cumulative_pct": (cum / total_stable) if total_stable else 0.0,
                "representation_subtype": (t.get("representation") or {}).get("subtype"),
            }
        )

    def _pct_at(threshold: float) -> int | None:
        for row in pareto_rows:
            if row["cumulative_pct"] >= threshold:
                return int(row["rank"])
        return None

    pareto_thresholds = {
        "concepts_for_25pct": _pct_at(0.25),
        "concepts_for_50pct": _pct_at(0.50),
        "concepts_for_75pct": _pct_at(0.75),
        "total_never_present_topics": len(never_sorted),
        "total_impact_sum": sum(t["total_impact"] for t in never_sorted),
        "note": (
            "Impact sums endpoint incidences; an edge with two never-present endpoints "
            "contributes to both topics (double-count in impact totals)."
        ),
    }

    # Combined root-cause ranking (topics + pure edges + rep clusters)
    root_causes: list[dict[str, Any]] = []
    for t in never_sorted:
        if t["total_impact"] <= 0:
            continue
        root_causes.append(
            {
                "kind": "MISSING_CONCEPT",
                "case_id": t["case_id"],
                "label": t["topic"],
                "attribution": (
                    "ENDPOINT_REPRESENTATION_MISMATCH"
                    if t.get("representation")
                    else (
                        "TARGET_NEVER_PRESENT"
                        if t["target_impact"] >= t["source_impact"]
                        else "SOURCE_NEVER_PRESENT"
                    )
                ),
                "edges_explained": t["total_impact"],
            }
        )
    for e in pure_relationship:
        root_causes.append(
            {
                "kind": "PURE_RELATIONSHIP_OMISSION",
                "case_id": e["case_id"],
                "label": e["edge_key"],
                "attribution": "BOTH_ENDPOINTS_PRESENT_EDGE_OMISSION",
                "edges_explained": 1,
            }
        )
    # Representation clusters by (case, gold, subtype)
    rep_cluster_edges: dict[tuple[str, str, str], int] = Counter()
    for e in stable_missing_edges:
        if e["primary_attribution"] != "ENDPOINT_REPRESENTATION_MISMATCH":
            continue
        for side, gold in (("source", e["from"]), ("target", e["to"])):
            rep = e.get(f"{side}_representation")
            if not rep:
                continue
            key = (e["case_id"], gold, str(rep.get("subtype") or "UNKNOWN"))
            rep_cluster_edges[key] += 1
    for (cid, gold, subtype), cnt in rep_cluster_edges.items():
        root_causes.append(
            {
                "kind": "REPRESENTATION_MISMATCH",
                "case_id": cid,
                "label": f"{gold} [{subtype}]",
                "attribution": "ENDPOINT_REPRESENTATION_MISMATCH",
                "edges_explained": cnt,
            }
        )

    root_causes.sort(key=lambda r: (-r["edges_explained"], r["case_id"], r["label"]))
    # Deduplicate overlapping topic/rep entries preferring higher edge counts already sorted
    seen_keys: set[tuple[str, str]] = set()
    deduped_roots: list[dict[str, Any]] = []
    for r in root_causes:
        key = (r["case_id"], r["label"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_roots.append(r)
    root_causes = deduped_roots

    cum = 0
    top_roots = []
    for i, r in enumerate(root_causes[:20], start=1):
        cum += r["edges_explained"]
        top_roots.append(
            {
                "rank": i,
                **r,
                "pct": (r["edges_explained"] / total_stable) if total_stable else 0.0,
                "cumulative_pct": (cum / total_stable) if total_stable else 0.0,
            }
        )

    clusters = {
        "HIGH_IMPACT_MISSING_CONCEPT": [
            t for t in never_sorted if t["total_impact"] >= 2
        ],
        "REPRESENTATION_FAILURE": [
            {
                "case_id": cid,
                "topic": gold,
                "subtype": subtype,
                "edges": cnt,
            }
            for (cid, gold, subtype), cnt in sorted(
                rep_cluster_edges.items(), key=lambda x: -x[1]
            )
        ],
        "TRUE_RELATIONSHIP_FAILURE": pure_relationship,
        "CURRICULUM_SCOPE_FAILURE": _curriculum_scope_clusters(case_summaries, never_present_topics),
    }

    diagnosis, rationale = _choose_diagnosis(
        attr_counts=attr_counts,
        total_stable=total_stable,
        node_endpoint_rate=node_endpoint_rate,
        pure_relationship_rate=pure_relationship_rate,
        pareto_rows=pareto_rows,
        n_gens=n_gens,
        n_cases=len(case_summaries),
    )

    representatives = _pick_representatives(case_summaries, stable_missing_edges)

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_persistent_failure_cluster_analysis.json"
    md_path = out_dir / f"{ts}_persistent_failure_cluster_analysis.md"
    pareto_path = out_dir / f"{ts}_persistent_failure_pareto.md"

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(target),
        "dataset": str(ds_path),
        "system": system,
        "matching_mode": "curated_alias",
        "edge_mode": "edge_calibrated",
        "scoring_fix": SCORING_FIX,
        "configuration": {
            "model": payload.get("model"),
            "generations_per_case": n_gens,
            "n_cases": len(case_summaries),
            "seed_supported": payload.get("seed_supported"),
            "FREQ_CONSISTENT": FREQ_CONSISTENT,
            "edge_semantics": (
                "[from, to] means from requires to (to is prerequisite). "
                "Alternative path = directed walk from → … → to in the generated dependency graph."
            ),
        },
        "diagnosis": {"code": diagnosis, "rationale": rationale},
        "stable_missing_edge_count": total_stable,
        "attribution_summary": attribution_summary,
        "node_or_endpoint_failure_rate": node_endpoint_rate,
        "pure_relationship_failure_rate": pure_relationship_rate,
        "alternative_path": {
            "stable_missing_with_alternative_path": alt_path_hits,
            "rate": (alt_path_hits / total_stable) if total_stable else 0.0,
            "interpretation": (
                "ALTERNATIVE_PATH_PRESENT when a majority of generations contain a directed "
                "prerequisite path from the dependent endpoint to the gold prerequisite."
            ),
        },
        "never_present_topic_count": len(never_present_topics),
        "never_present_topics": never_sorted,
        "pareto_by_never_present_topic": pareto_rows,
        "pareto_thresholds": pareto_thresholds,
        "top_root_causes": top_roots,
        "clusters": {
            "HIGH_IMPACT_MISSING_CONCEPT": clusters["HIGH_IMPACT_MISSING_CONCEPT"],
            "REPRESENTATION_FAILURE": clusters["REPRESENTATION_FAILURE"],
            "TRUE_RELATIONSHIP_FAILURE_COUNT": len(clusters["TRUE_RELATIONSHIP_FAILURE"]),
            "TRUE_RELATIONSHIP_FAILURE": clusters["TRUE_RELATIONSHIP_FAILURE"][:50],
            "CURRICULUM_SCOPE_FAILURE": clusters["CURRICULUM_SCOPE_FAILURE"],
        },
        "pure_relationship_failures": pure_relationship,
        "stable_missing_edges": stable_missing_edges,
        "representative_case_ids": {k: v["case_id"] for k, v in representatives.items() if v},
        "cases": case_summaries,
        "notes": [
            "Evaluation-only; gold used solely for attribution.",
            "No aliases/matching/gold/prompt/generation changes.",
            SCORING_FIX["summary"],
        ],
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_md(result, representatives), encoding="utf-8")
    pareto_path.write_text(_render_pareto(result), encoding="utf-8")
    return md_path, json_path, pareto_path


def _curriculum_scope_clusters(
    case_summaries: list[dict[str, Any]],
    never_topics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group cases with many never-present topics by dataset category (no LLM)."""
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in never_topics:
        by_case[t["case_id"]].append(t)
    out = []
    for c in case_summaries:
        topics = by_case.get(c["case_id"]) or []
        if len(topics) < 2:
            continue
        out.append(
            {
                "case_id": c["case_id"],
                "category": c.get("category"),
                "never_present_count": len(topics),
                "topics": [t["topic"] for t in topics],
                "total_impact": sum(t["total_impact"] for t in topics),
            }
        )
    out.sort(key=lambda x: -x["total_impact"])
    return out


def _choose_diagnosis(
    *,
    attr_counts: Counter[str],
    total_stable: int,
    node_endpoint_rate: float,
    pure_relationship_rate: float,
    pareto_rows: list[dict[str, Any]],
    n_gens: int,
    n_cases: int,
) -> tuple[str, str]:
    if n_gens < 2 or total_stable == 0 or n_cases < 3:
        return (
            "INSUFFICIENT_EVIDENCE",
            f"n_gens={n_gens}, n_cases={n_cases}, stable_missing={total_stable}",
        )

    absence_n = (
        attr_counts.get("SOURCE_NEVER_PRESENT", 0)
        + attr_counts.get("TARGET_NEVER_PRESENT", 0)
        + attr_counts.get("BOTH_ENDPOINTS_NEVER_PRESENT", 0)
    )
    absence_rate = absence_n / total_stable
    rep_rate = attr_counts.get("ENDPOINT_REPRESENTATION_MISMATCH", 0) / total_stable

    # Concentration: share of impact from top never-present concepts
    top5_share = 0.0
    if pareto_rows and total_stable:
        top5_share = min(1.0, sum(r["stable_missing_edges_explained"] for r in pareto_rows[:5]) / total_stable)

    rationale = (
        f"node_endpoint_rate={node_endpoint_rate:.3f} (absence={absence_rate:.3f}, "
        f"rep={rep_rate:.3f}), pure_rel={pure_relationship_rate:.3f}, "
        f"top5_never_present_impact_share≈{top5_share:.3f}, stable_missing={total_stable}."
    )

    if node_endpoint_rate >= 0.60 and absence_rate >= rep_rate and absence_rate >= pure_relationship_rate:
        return ("ENDPOINT_COVERAGE_DOMINANT", rationale)
    if rep_rate >= 0.40 and rep_rate >= absence_rate and rep_rate >= pure_relationship_rate:
        return ("REPRESENTATION_DOMINANT", rationale)
    if pure_relationship_rate >= 0.50:
        return ("RELATIONSHIP_REASONING_DOMINANT", rationale)
    if node_endpoint_rate >= 0.45 or pure_relationship_rate >= 0.25 or rep_rate >= 0.25:
        return ("MIXED_ROOT_CAUSES", rationale)
    return ("MIXED_ROOT_CAUSES", rationale)


def _pick_representatives(
    cases: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    def _best(pred) -> dict[str, Any] | None:
        scored = []
        for c in cases:
            edges_c = [e for e in edges if e["case_id"] == c["case_id"]]
            if not edges_c:
                continue
            if pred(c, edges_c):
                scored.append((len(edges_c), c))
        if not scored:
            return None
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    missing_endpoint = _best(
        lambda c, es: sum(
            1
            for e in es
            if e["primary_attribution"]
            in {"SOURCE_NEVER_PRESENT", "TARGET_NEVER_PRESENT", "BOTH_ENDPOINTS_NEVER_PRESENT"}
        )
        >= max(1, len(es) // 2)
    )
    rep_mismatch = _best(
        lambda c, es: sum(1 for e in es if e["primary_attribution"] == "ENDPOINT_REPRESENTATION_MISMATCH")
        >= 1
    )
    pure_rel = _best(
        lambda c, es: any(e["primary_attribution"] == "BOTH_ENDPOINTS_PRESENT_EDGE_OMISSION" for e in es)
    )
    alt_path = _best(lambda c, es: any(e.get("alternative_path_present") for e in es))
    mixed = _best(lambda c, es: len({e["primary_attribution"] for e in es}) >= 2)

    return {
        "missing_endpoint_cluster": missing_endpoint,
        "representation_mismatch": rep_mismatch,
        "pure_relationship_omission": pure_rel,
        "alternative_path": alt_path,
        "mixed_causes": mixed,
    }


def _render_pareto(payload: dict[str, Any]) -> str:
    lines = [
        "# Persistent Failure Pareto",
        "",
        f"- Source: `{payload['source_artifact']}`",
        f"- Stable missing edges: **{payload['stable_missing_edge_count']}**",
        f"- Diagnosis: **{payload['diagnosis']['code']}**",
        "",
        "## Thresholds (never-present concept impact)",
        "",
        f"- Concepts for ≥25%: {payload['pareto_thresholds'].get('concepts_for_25pct')}",
        f"- Concepts for ≥50%: {payload['pareto_thresholds'].get('concepts_for_50pct')}",
        f"- Concepts for ≥75%: {payload['pareto_thresholds'].get('concepts_for_75pct')}",
        "",
        "## Top 10 persistent root causes",
        "",
        "| Rank | Case | Root cause | Attribution | Edges | % | Cumulative % |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in payload["top_root_causes"][:10]:
        lines.append(
            f"| {r['rank']} | {r['case_id']} | {r['label']} | {r['attribution']} | "
            f"{r['edges_explained']} | {100 * r['pct']:.1f}% | {100 * r['cumulative_pct']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Never-present topic impact (top 15)",
            "",
            "| Rank | Topic | Case | Edges explained | % | Cumulative % |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for r in payload["pareto_by_never_present_topic"][:15]:
        lines.append(
            f"| {r['rank']} | {r['topic']} | {r['case_id']} | "
            f"{r['stable_missing_edges_explained']} | {100 * r['pct_of_stable_missing']:.1f}% | "
            f"{100 * r['cumulative_pct']:.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_md(
    payload: dict[str, Any],
    representatives: dict[str, dict[str, Any] | None],
) -> str:
    lines = [
        "# Persistent Failure Cluster Analysis",
        "",
        f"- Source: `{payload['source_artifact']}`",
        f"- Matching: curated_alias + edge_calibrated",
        f"- Scoring fix: `{payload['scoring_fix']['id']}` ({payload['scoring_fix']['kind']})",
        f"- Stable missing edges: **{payload['stable_missing_edge_count']}**",
        f"- Node/endpoint failure rate: **{payload['node_or_endpoint_failure_rate']:.3f}**",
        f"- Pure relationship failure rate: **{payload['pure_relationship_failure_rate']:.3f}**",
        f"- Alternative path rate: **{payload['alternative_path']['rate']:.3f}**",
        f"- Diagnosis: **{payload['diagnosis']['code']}**",
        f"- Rationale: {payload['diagnosis']['rationale']}",
        "",
        "## Attribution summary",
        "",
        "| Primary Attribution | Count | Rate |",
        "| --- | ---: | ---: |",
    ]
    for row in payload["attribution_summary"]:
        lines.append(
            f"| {row['primary_attribution']} | {row['count']} | {row['rate']:.3f} |"
        )

    lines.extend(["", "## Representative cases", ""])
    labels = {
        "missing_endpoint_cluster": "Missing endpoint cluster",
        "representation_mismatch": "Representation mismatch",
        "pure_relationship_omission": "Pure relationship omission",
        "alternative_path": "Alternative generated path",
        "mixed_causes": "Mixed causes",
    }
    for key, title in labels.items():
        c = representatives.get(key)
        lines.append(f"### {title}")
        if not c:
            lines.extend(["", "_No case found for this pattern._", ""])
            continue
        lines.extend(
            [
                "",
                f"**Case:** `{c['case_id']}`",
                "",
                f"**Objective:** {c['goal']}",
                "",
                f"- Gold topics: {c['gold_topics']}",
                f"- Gold deps: {c['gold_dependencies']}",
                f"- Attribution counts: {c['attribution_counts']}",
                f"- Never-present: {[t['topic'] for t in c.get('never_present_topics') or []]}",
                "",
            ]
        )
        for g in c.get("generations") or []:
            lines.append(
                f"  - gen {g['generation_index']} seed={g.get('seed')}: "
                f"topics={g['topics']} deps={g['dependencies']}"
            )
        lines.append("")
        for e in (c.get("stable_missing_edges") or [])[:12]:
            lines.append(
                f"  - missing `{e['edge_key']}` → {e['primary_attribution']} "
                f"(src_f={e['source_frequency']:.2f}, tgt_f={e['target_frequency']:.2f}, "
                f"alt_path={e['alternative_path_present']})"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
