"""Offline representation-alignment replay on stored baseline generations (no LLM)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.inspect import _graph_from_row
from app.evaluation.metrics import aggregate_scores, score_graph
from app.evaluation.inventory_attribution import edge_opportunity_and_conditional_recall
from app.evaluation.node_edge_attribution import (
    attribute_invalid_extra_edge,
    attribute_missing_required_edge,
    classify_gold_topic_representation,
    load_node_representation_map,
)
from app.evaluation.schemas import GeneratedGraph
from app.services.representation_alignment import align_graph

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "failure_analysis"
DEFAULT_BENCH = _REPO_ROOT / "results" / "benchmarks"


def find_baseline_artifact(bench_dir: Path | None = None) -> Path:
    root = Path(bench_dir) if bench_dir else DEFAULT_BENCH
    # Prefer multi-gen stability, else latest quality with synapse
    for pattern in ("*_quality_stability_*.json", "*_quality_*_baseline.json", "*_quality_*.json"):
        cands = sorted(root.glob(pattern), reverse=True)
        for p in cands:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "synapse" in (data.get("systems") or {}):
                return p
    raise FileNotFoundError(f"No synapse quality artifact under {root}")


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _graph_attr_counts(
    example,
    graph: GeneratedGraph,
    *,
    rep_map: dict[str, Any],
) -> Counter[str]:
    from app.evaluation.metrics import compare_graphs

    cmp = compare_graphs(example, graph)
    counts: Counter[str] = Counter()
    for frm, to in (tuple(e) for e in cmp["missing_dependencies"]):
        rec = attribute_missing_required_edge(frm, to, example, graph, rep_map=rep_map)
        counts[rec["primary_attribution"]] += 1
    for frm, to in (tuple(e) for e in cmp["extra_dependencies"]):
        rec = attribute_invalid_extra_edge(frm, to, example, graph, rep_map=rep_map)
        counts[rec["primary_attribution"]] += 1
    # Gold topic representation statuses for mismatch rates
    for g in example.required_topic_list():
        st = classify_gold_topic_representation(g, example, graph, rep_map=rep_map)["status"]
        counts[f"GOLD_REP::{st}"] += 1
    return counts


def run_representation_alignment_replay(
    artifact_path: str | Path | None = None,
    *,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
    output_dir: str | Path | None = None,
    enable_framing: bool = True,
    enable_context: bool = True,
    enable_merge: bool = True,
    first_repetition_only: bool = True,
) -> tuple[Path, Path, Path]:
    """Replay alignment on stored generations; write analysis JSON/MD + rescored artifact."""
    target = Path(artifact_path) if artifact_path else find_baseline_artifact()
    payload = json.loads(target.read_text(encoding="utf-8"))
    systems = payload.get("systems") or {}
    if system not in systems:
        raise ValueError(f"System {system!r} not in artifact; found {list(systems)}")

    ds_stem = payload.get("dataset") or "learning_graph_quality_v1"
    ds_path = Path(dataset_path) if dataset_path else _REPO_ROOT / "data" / "eval" / f"{ds_stem}.jsonl"
    if not ds_path.is_file():
        ds_path = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    examples = {ex.id: ex for ex in load_dataset(ds_path)}
    rep_map = load_node_representation_map()

    rows = list((systems[system] or {}).get("example_results") or [])
    if first_repetition_only:
        seen: set[str] = set()
        filtered = []
        for r in rows:
            eid = str(r.get("example_id") or "")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            filtered.append(r)
        rows = filtered

    baseline_scores = []
    aligned_scores = []
    case_details: list[dict[str, Any]] = []
    global_counts: Counter[str] = Counter()
    base_attr: Counter[str] = Counter()
    aligned_attr: Counter[str] = Counter()
    opp_base_totals = {"opportunity_edge_count": 0, "required_edge_count": 0, "opportunity_correct": 0}
    opp_align_totals = {"opportunity_edge_count": 0, "required_edge_count": 0, "opportunity_correct": 0}
    rescored_rows: list[dict[str, Any]] = []
    safety = {
        "new_topics_created": 0,
        "topics_deleted_without_merge": 0,
        "dag_violations": 0,
        "unsafe_merges_detected": 0,
        "incorrect_alignments_flagged": 0,
    }

    for row in rows:
        eid = str(row.get("example_id") or "")
        ex = examples.get(eid)
        if not ex:
            continue
        adapted = adapt_example_for_edge_mode(
            ex, "edge_calibrated", topic_matching_mode="curated_alias"
        )
        g0 = _graph_from_row(row)
        if not g0.parse_ok:
            continue
        s0 = score_graph(adapted, g0)
        baseline_scores.append(s0)
        base_attr.update(_graph_attr_counts(adapted, g0, rep_map=rep_map))
        opp0 = edge_opportunity_and_conditional_recall(adapted, g0, list(g0.dependencies))
        for k in opp_base_totals:
            opp_base_totals[k] += int(opp0.get(k, 0))

        result = align_graph(
            list(g0.topics),
            list(g0.dependencies),
            request_text=ex.goal,
            enable_framing=enable_framing,
            enable_context=enable_context,
            enable_merge=enable_merge,
        )
        safety["new_topics_created"] += result.new_topics_created
        safety["topics_deleted_without_merge"] += result.topics_deleted_without_merge
        if not result.dag_valid:
            safety["dag_violations"] += 1

        for k, v in result.counts.items():
            if isinstance(v, int):
                global_counts[k] += v

        g1 = GeneratedGraph(
            topics=list(result.topics_after),
            dependencies=list(result.dependencies_after),
            parse_ok=True,
            generation_meta=result.to_meta(),
        )
        s1 = score_graph(adapted, g1)
        aligned_scores.append(s1)
        aligned_attr.update(_graph_attr_counts(adapted, g1, rep_map=rep_map))
        opp1 = edge_opportunity_and_conditional_recall(adapted, g1, list(g1.dependencies))
        for k in opp_align_totals:
            opp_align_totals[k] += int(opp1.get(k, 0))

        case_details.append(
            {
                "case_id": eid,
                "goal": ex.goal,
                "topics_before": result.topics_before,
                "topics_after": result.topics_after,
                "dependencies_before": [list(e) for e in result.dependencies_before],
                "dependencies_after": [list(e) for e in result.dependencies_after],
                "records": [r.to_dict() for r in result.records],
                "counts": result.counts,
                "edge_opportunity": {"baseline": opp0, "aligned": opp1},
                "baseline_scores": {
                    "topic_f1": s0.topic_f1,
                    "required_edge_f1": s0.required_edge_f1,
                    "required_edge_recall": s0.required_edge_recall,
                    "missing_required_edge_rate": s0.missing_required_edge_rate,
                    "invalid_extra_edge_rate": s0.invalid_extra_edge_rate,
                    "hallucinated_topic_rate": s0.hallucinated_topic_rate,
                    "dependency_direction_error_rate": s0.dependency_direction_error_rate,
                },
                "aligned_scores": {
                    "topic_f1": s1.topic_f1,
                    "required_edge_f1": s1.required_edge_f1,
                    "required_edge_recall": s1.required_edge_recall,
                    "missing_required_edge_rate": s1.missing_required_edge_rate,
                    "invalid_extra_edge_rate": s1.invalid_extra_edge_rate,
                    "hallucinated_topic_rate": s1.hallucinated_topic_rate,
                    "dependency_direction_error_rate": s1.dependency_direction_error_rate,
                },
                "deltas": {
                    "topic_f1": s1.topic_f1 - s0.topic_f1,
                    "required_edge_f1": s1.required_edge_f1 - s0.required_edge_f1,
                    "required_edge_recall": s1.required_edge_recall - s0.required_edge_recall,
                },
            }
        )

        new_row = dict(row)
        new_row["generated_topics"] = list(result.topics_after)
        new_row["generated_dependencies"] = [list(e) for e in result.dependencies_after]
        new_row["generation_meta"] = {
            **(row.get("generation_meta") or {}),
            **result.to_meta(),
        }
        new_row["scores"] = {
            "topic_precision": s1.topic_precision,
            "topic_recall": s1.topic_recall,
            "topic_f1": s1.topic_f1,
            "dependency_precision": s1.dependency_precision,
            "dependency_recall": s1.dependency_recall,
            "dependency_f1": s1.dependency_f1,
            "required_edge_precision": s1.required_edge_precision,
            "required_edge_recall": s1.required_edge_recall,
            "required_edge_f1": s1.required_edge_f1,
            "missing_required_edge_rate": s1.missing_required_edge_rate,
            "invalid_extra_edge_rate": s1.invalid_extra_edge_rate,
            "hallucinated_topic_rate": s1.hallucinated_topic_rate,
            "dependency_direction_error_rate": s1.dependency_direction_error_rate,
            "graph_valid": s1.graph_valid,
        }
        rescored_rows.append(new_row)

    base_agg = aggregate_scores(baseline_scores)
    align_agg = aggregate_scores(aligned_scores)

    def _safe_div(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    def _gold_rate(attr: Counter[str], status: str) -> float:
        gold_n = sum(v for k, v in attr.items() if k.startswith("GOLD_REP::"))
        return _safe_div(attr.get(f"GOLD_REP::{status}", 0), gold_n)

    changed = global_counts.get("normalized", 0) + global_counts.get("merged", 0)
    # Deterministic offline pass: no LLM merges; treat automated safety failures as unsafe.
    unsafe = (
        safety["new_topics_created"]
        + safety["topics_deleted_without_merge"]
        + safety["dag_violations"]
        + safety["unsafe_merges_detected"]
    )
    safety["SAFE_ALIGNMENT_RATE"] = _safe_div(max(changed - unsafe, 0), changed) if changed else 1.0
    safety["UNSAFE_ALIGNMENT_RATE"] = _safe_div(unsafe, changed) if changed else 0.0

    edge_opportunity = {
        "baseline": {
            **opp_base_totals,
            "EDGE_OPPORTUNITY_RATE": _safe_div(
                opp_base_totals["opportunity_edge_count"], opp_base_totals["required_edge_count"]
            ),
            "CONDITIONAL_EDGE_RECALL": _safe_div(
                opp_base_totals["opportunity_correct"], opp_base_totals["opportunity_edge_count"]
            ),
        },
        "representation_alignment": {
            **opp_align_totals,
            "EDGE_OPPORTUNITY_RATE": _safe_div(
                opp_align_totals["opportunity_edge_count"], opp_align_totals["required_edge_count"]
            ),
            "CONDITIONAL_EDGE_RECALL": _safe_div(
                opp_align_totals["opportunity_correct"], opp_align_totals["opportunity_edge_count"]
            ),
        },
    }

    gran_rates = {
        "baseline": {
            "granularity_mismatch_rate": _gold_rate(base_attr, "GRANULARITY_VARIANT"),
            "granularity_drift_rate": _safe_div(
                base_attr.get("ENDPOINT_GRANULARITY_DRIFT", 0),
                sum(v for k, v in base_attr.items() if not k.startswith("GOLD_REP::")),
            ),
        },
        "representation_alignment": {
            "granularity_mismatch_rate": _gold_rate(aligned_attr, "GRANULARITY_VARIANT"),
            "granularity_drift_rate": _safe_div(
                aligned_attr.get("ENDPOINT_GRANULARITY_DRIFT", 0),
                sum(v for k, v in aligned_attr.items() if not k.startswith("GOLD_REP::")),
            ),
        },
    }

    n_cases = len(case_details)
    diagnosis, rationale = _decide(base_agg, align_agg, global_counts, base_attr, aligned_attr, safety)

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_representation_alignment_analysis.json"
    md_path = out_dir / f"{ts}_representation_alignment_comparison.md"
    bench_path = DEFAULT_BENCH / f"{ts}_quality_representation_alignment_replay.json"
    DEFAULT_BENCH.mkdir(parents=True, exist_ok=True)

    analysis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(target),
        "dataset": str(ds_path),
        "system": system,
        "no_new_llm_calls": True,
        "matching_mode": "curated_alias",
        "edge_mode": "edge_calibrated",
        "config": {
            "enable_framing": enable_framing,
            "enable_context": enable_context,
            "enable_merge": enable_merge,
            "first_repetition_only": first_repetition_only,
        },
        "n_cases": n_cases,
        "alignment_behavior": {
            "topics_inspected": global_counts.get("topics_inspected", 0),
            "normalized": global_counts.get("normalized", 0),
            "framing_normalized": global_counts.get("framing_normalized", 0),
            "context_aligned": global_counts.get("context_aligned", 0),
            "merged": global_counts.get("merged", 0),
            "kept": global_counts.get("kept", 0),
            "unresolved": global_counts.get("unresolved", 0),
            "NORMALIZED_TOPIC_COUNT": global_counts.get("normalized", 0),
            "MERGED_TOPIC_COUNT": global_counts.get("merged", 0),
            "FRAMING_NORMALIZED_COUNT": global_counts.get("framing_normalized", 0),
            "UNRESOLVED_COUNT": global_counts.get("unresolved", 0),
            "ALIGNMENT_RATE": (
                changed / global_counts["topics_inspected"]
                if global_counts.get("topics_inspected")
                else 0.0
            ),
            "MERGE_RATE": (
                global_counts.get("merged", 0) / global_counts["topics_inspected"]
                if global_counts.get("topics_inspected")
                else 0.0
            ),
            "SAFE_ALIGNMENT_RATE": safety["SAFE_ALIGNMENT_RATE"],
            "UNSAFE_ALIGNMENT_RATE": safety["UNSAFE_ALIGNMENT_RATE"],
            "alignment_rate": (
                changed / global_counts["topics_inspected"]
                if global_counts.get("topics_inspected")
                else 0.0
            ),
            "merge_rate": (
                global_counts.get("merged", 0) / global_counts["topics_inspected"]
                if global_counts.get("topics_inspected")
                else 0.0
            ),
        },
        "aggregate": {
            "baseline": {
                **base_agg,
                **gran_rates["baseline"],
                "EDGE_OPPORTUNITY_RATE": edge_opportunity["baseline"]["EDGE_OPPORTUNITY_RATE"],
                "CONDITIONAL_EDGE_RECALL": edge_opportunity["baseline"]["CONDITIONAL_EDGE_RECALL"],
            },
            "representation_alignment": {
                **align_agg,
                **gran_rates["representation_alignment"],
                "EDGE_OPPORTUNITY_RATE": edge_opportunity["representation_alignment"][
                    "EDGE_OPPORTUNITY_RATE"
                ],
                "CONDITIONAL_EDGE_RECALL": edge_opportunity["representation_alignment"][
                    "CONDITIONAL_EDGE_RECALL"
                ],
            },
            "delta": {
                k: (
                    (
                        {
                            **align_agg,
                            **gran_rates["representation_alignment"],
                            "EDGE_OPPORTUNITY_RATE": edge_opportunity["representation_alignment"][
                                "EDGE_OPPORTUNITY_RATE"
                            ],
                            "CONDITIONAL_EDGE_RECALL": edge_opportunity["representation_alignment"][
                                "CONDITIONAL_EDGE_RECALL"
                            ],
                        }.get(k, 0)
                    )
                    - (
                        {
                            **base_agg,
                            **gran_rates["baseline"],
                            "EDGE_OPPORTUNITY_RATE": edge_opportunity["baseline"][
                                "EDGE_OPPORTUNITY_RATE"
                            ],
                            "CONDITIONAL_EDGE_RECALL": edge_opportunity["baseline"][
                                "CONDITIONAL_EDGE_RECALL"
                            ],
                        }.get(k, 0)
                    )
                )
                for k in (
                    "topic_f1",
                    "required_edge_f1",
                    "required_edge_recall",
                    "missing_required_edge_rate",
                    "invalid_extra_edge_rate",
                    "hallucinated_topic_rate",
                    "dependency_direction_error_rate",
                    "granularity_mismatch_rate",
                    "granularity_drift_rate",
                    "EDGE_OPPORTUNITY_RATE",
                    "CONDITIONAL_EDGE_RECALL",
                )
            },
        },
        "edge_opportunity": edge_opportunity,
        "root_cause_impact": {
            "baseline_attr": dict(base_attr),
            "aligned_attr": dict(aligned_attr),
            "endpoint_representation_mismatch_baseline": base_attr.get(
                "ENDPOINT_REPRESENTATION_MISMATCH", 0
            ),
            "endpoint_representation_mismatch_aligned": aligned_attr.get(
                "ENDPOINT_REPRESENTATION_MISMATCH", 0
            ),
            "endpoint_granularity_mismatch_baseline": base_attr.get(
                "ENDPOINT_GRANULARITY_MISMATCH", 0
            ),
            "endpoint_granularity_mismatch_aligned": aligned_attr.get(
                "ENDPOINT_GRANULARITY_MISMATCH", 0
            ),
            "endpoint_granularity_drift_baseline": base_attr.get(
                "ENDPOINT_GRANULARITY_DRIFT", 0
            ),
            "endpoint_granularity_drift_aligned": aligned_attr.get(
                "ENDPOINT_GRANULARITY_DRIFT", 0
            ),
            "EDGE_OPPORTUNITY_RATE": {
                "baseline": edge_opportunity["baseline"]["EDGE_OPPORTUNITY_RATE"],
                "aligned": edge_opportunity["representation_alignment"]["EDGE_OPPORTUNITY_RATE"],
            },
            "CONDITIONAL_EDGE_RECALL": {
                "baseline": edge_opportunity["baseline"]["CONDITIONAL_EDGE_RECALL"],
                "aligned": edge_opportunity["representation_alignment"]["CONDITIONAL_EDGE_RECALL"],
            },
            "note": (
                "Attribution counts are absolute event counts over missing/extra edges; "
                "alignment cannot recover never-present concepts. "
                "Node-edge attribution uses ENDPOINT_GRANULARITY_*; "
                "persistent-failure cluster ENDPOINT_REPRESENTATION_MISMATCH is a coarser offline label."
            ),
        },
        "safety": safety,
        "diagnosis": {"code": diagnosis, "rationale": rationale},
        "cases": case_details,
        "rescored_artifact": str(bench_path),
    }
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_md(analysis), encoding="utf-8")

    rescored_payload = {
        **{k: v for k, v in payload.items() if k != "systems"},
        "benchmark_type": "quality_representation_alignment_replay",
        "source_artifact": str(target),
        "no_new_llm_calls": True,
        "systems": {
            "synapse_baseline_replay": systems[system],
            "representation_alignment": {
                "metrics": align_agg,
                "example_results": rescored_rows,
                "note": "Offline title/merge alignment only; no new concepts.",
            },
        },
        "metrics": {
            "synapse_baseline_replay": base_agg,
            "representation_alignment": align_agg,
        },
    }
    bench_path.write_text(json.dumps(rescored_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return md_path, json_path, bench_path


def _decide(
    base_agg: dict[str, float],
    align_agg: dict[str, float],
    counts: Counter[str],
    base_attr: Counter[str],
    aligned_attr: Counter[str],
    safety: dict[str, int],
) -> tuple[str, str]:
    if safety.get("new_topics_created", 0) > 0:
        return (
            "NOT_SUPPORTED",
            f"Safety violation: new_topics_created={safety['new_topics_created']}",
        )
    if counts.get("topics_inspected", 0) == 0:
        return ("INSUFFICIENT_EVIDENCE", "No topics inspected.")

    dt = align_agg.get("topic_f1", 0) - base_agg.get("topic_f1", 0)
    de = align_agg.get("required_edge_f1", 0) - base_agg.get("required_edge_f1", 0)
    dr = align_agg.get("required_edge_recall", 0) - base_agg.get("required_edge_recall", 0)
    changed = counts.get("normalized", 0) + counts.get("merged", 0)

    rationale = (
        f"Δtopic_f1={dt:.4f}, Δreq_edge_f1={de:.4f}, Δreq_edge_recall={dr:.4f}, "
        f"changed_topics={changed}/{counts.get('topics_inspected', 0)}, "
        f"dag_violations={safety.get('dag_violations', 0)}."
    )

    if changed == 0:
        return ("NOT_SUPPORTED", rationale + " No alignments applied.")
    if de >= 0.02 and dt >= 0.01 and safety.get("dag_violations", 0) == 0:
        return ("SUPPORTED", rationale)
    if (dt >= 0.005 or de > 0 or dr > 0) and safety.get("dag_violations", 0) == 0:
        return ("PARTIALLY_SUPPORTED", rationale)
    if abs(dt) < 0.005 and abs(de) < 0.005:
        return ("NOT_SUPPORTED", rationale + " Negligible metric movement.")
    return ("PARTIALLY_SUPPORTED", rationale)


def _render_md(payload: dict[str, Any]) -> str:
    a = payload["aggregate"]
    b, c, d = a["baseline"], a["representation_alignment"], a["delta"]
    beh = payload["alignment_behavior"]
    lines = [
        "# Constrained Representation Alignment Comparison",
        "",
        f"- Source: `{payload['source_artifact']}`",
        f"- NO_NEW_LLM_CALLS: `{payload['no_new_llm_calls']}`",
        f"- Cases: {payload['n_cases']}",
        f"- Diagnosis: **{payload['diagnosis']['code']}**",
        f"- Rationale: {payload['diagnosis']['rationale']}",
        "",
        "## Alignment behavior",
        "",
        f"- Inspected: {beh['topics_inspected']}",
        f"- Normalized: {beh['normalized']} (framing={beh['framing_normalized']}, context={beh['context_aligned']})",
        f"- Merged: {beh['merged']}",
        f"- Kept: {beh['kept']}",
        f"- Unresolved: {beh['unresolved']}",
        f"- Alignment rate: {beh['alignment_rate']:.3f}",
        f"- Merge rate: {beh['merge_rate']:.3f}",
        f"- SAFE_ALIGNMENT_RATE: {beh.get('SAFE_ALIGNMENT_RATE', 1.0):.3f}",
        f"- UNSAFE_ALIGNMENT_RATE: {beh.get('UNSAFE_ALIGNMENT_RATE', 0.0):.3f}",
        "",
        "## Benchmark comparison",
        "",
        "| Metric | Baseline | Representation Alignment | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("topic_f1", "Topic F1"),
        ("required_edge_f1", "Required Edge F1"),
        ("required_edge_recall", "Required Edge Recall"),
        ("missing_required_edge_rate", "Missing Required Edge Rate"),
        ("invalid_extra_edge_rate", "Invalid Extra Edge Rate"),
        ("hallucinated_topic_rate", "Hallucinated Topic Rate"),
        ("dependency_direction_error_rate", "Direction Error Rate"),
        ("granularity_mismatch_rate", "Granularity Mismatch Rate"),
        ("granularity_drift_rate", "Granularity Drift Rate"),
        ("EDGE_OPPORTUNITY_RATE", "EDGE_OPPORTUNITY_RATE"),
        ("CONDITIONAL_EDGE_RECALL", "CONDITIONAL_EDGE_RECALL"),
    ]:
        lines.append(
            f"| {label} | {b.get(key, 0):.3f} | {c.get(key, 0):.3f} | {d.get(key, 0):+.3f} |"
        )
    rc = payload["root_cause_impact"]
    lines.extend(
        [
            "",
            "## Root-cause impact (attribution event counts)",
            "",
            f"- ENDPOINT_REPRESENTATION_MISMATCH: "
            f"{rc['endpoint_representation_mismatch_baseline']} → {rc['endpoint_representation_mismatch_aligned']}",
            f"- ENDPOINT_GRANULARITY_MISMATCH: "
            f"{rc['endpoint_granularity_mismatch_baseline']} → {rc['endpoint_granularity_mismatch_aligned']}",
            f"- ENDPOINT_GRANULARITY_DRIFT: "
            f"{rc.get('endpoint_granularity_drift_baseline', 0)} → {rc.get('endpoint_granularity_drift_aligned', 0)}",
            f"- EDGE_OPPORTUNITY_RATE: {rc['EDGE_OPPORTUNITY_RATE']['baseline']:.3f} → {rc['EDGE_OPPORTUNITY_RATE']['aligned']:.3f}",
            f"- CONDITIONAL_EDGE_RECALL: {rc['CONDITIONAL_EDGE_RECALL']['baseline']:.3f} → {rc['CONDITIONAL_EDGE_RECALL']['aligned']:.3f}",
            "",
            "## Safety",
            "",
            f"`{payload['safety']}`",
            "",
            "## Representative cases (changed alignments)",
            "",
        ]
    )
    shown = 0
    for case in payload["cases"]:
        changed = [
            r
            for r in case["records"]
            if r["decision"] in {"NORMALIZE_TITLE", "MERGE_WITH_EXISTING_GENERATED_TOPIC"}
        ]
        if not changed:
            continue
        shown += 1
        if shown > 12:
            break
        lines.extend(
            [
                f"### {case['case_id']}",
                "",
                f"**Objective:** {case['goal']}",
                f"- Δ Topic F1: {case['deltas']['topic_f1']:+.3f}",
                f"- Δ Required Edge F1: {case['deltas']['required_edge_f1']:+.3f}",
                "",
            ]
        )
        for r in changed[:8]:
            lines.append(
                f"  - `{r['original_title']}` → `{r['aligned_title']}` "
                f"({r['decision']} / {r['method']}) — {r['reason']}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
