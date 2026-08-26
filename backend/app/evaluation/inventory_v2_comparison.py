"""Targeted v1 vs v2 inventory comparison for databases + data_engineering."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.curriculum.inventory import load_case_domain_map
from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.metrics import compare_graphs, score_graph
from app.evaluation.schemas import GeneratedGraph

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "curriculum"
DEFAULT_DS = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"

TARGET_DOMAINS = ("databases", "data_engineering")


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _graph(row: dict[str, Any]) -> GeneratedGraph:
    return GeneratedGraph(
        topics=list(row.get("generated_topics") or row.get("topics") or []),
        dependencies=[tuple(e) for e in (row.get("generated_dependencies") or row.get("dependencies") or [])],
        parse_ok=bool(row.get("parse_ok", True)),
        error=row.get("error"),
        generation_meta=dict(row.get("generation_meta") or {}),
    )


def _agg_scores(rows: list[dict[str, Any]], examples: dict, case_map: dict[str, str]) -> dict[str, float]:
    scores = []
    lats = []
    costs = []
    for row in rows:
        eid = str(row.get("example_id"))
        domain = case_map.get(eid)
        if domain not in TARGET_DOMAINS:
            continue
        ex = examples[eid]
        adapted = adapt_example_for_edge_mode(ex, "edge_calibrated", topic_matching_mode="curated_alias")
        sc = score_graph(adapted, _graph(row))
        scores.append(sc)
        lat = row.get("llm_latency_ms") or row.get("total_latency_ms")
        if lat is not None:
            lats.append(float(lat))
        cost = row.get("cost_usd")
        if cost is None:
            cost = (row.get("generation_meta") or {}).get("estimated_cost_usd")
        if cost is not None:
            costs.append(float(cost))
    if not scores:
        return {"n": 0}
    return {
        "n": len(scores),
        "topic_precision": _mean([s.topic_precision for s in scores]),
        "topic_recall": _mean([s.topic_recall for s in scores]),
        "topic_f1": _mean([s.topic_f1 for s in scores]),
        "required_edge_precision": _mean([s.required_edge_precision for s in scores]),
        "required_edge_recall": _mean([s.required_edge_recall for s in scores]),
        "required_edge_f1": _mean([s.required_edge_f1 for s in scores]),
        "missing_required_edge_rate": _mean([s.missing_required_edge_rate for s in scores]),
        "invalid_extra_edge_rate": _mean([s.invalid_extra_edge_rate for s in scores]),
        "dependency_direction_error_rate": _mean([s.dependency_direction_error_rate for s in scores]),
        "redundant_transitive_edge_rate": _mean([s.redundant_transitive_edge_rate for s in scores]),
        "hallucinated_topic_rate": _mean([s.hallucinated_topic_rate for s in scores]),
        "latency_ms": _mean(lats),
        "estimated_cost_usd": _mean(costs),
    }


def _edge_impact(v1_rows: list[dict], v2_rows: list[dict], examples: dict, case_map: dict) -> dict[str, Any]:
    """Compare required/extra edges between aligned prior v1 and v2 generations."""
    def key(row):
        return (row["example_id"], int(row.get("generation_index", row.get("repetition", 0))))

    v1_map = {key(r): r for r in v1_rows if case_map.get(r["example_id"]) in TARGET_DOMAINS}
    v2_map = {key(r): r for r in v2_rows if case_map.get(r["example_id"]) in TARGET_DOMAINS}
    common = sorted(set(v1_map) & set(v2_map))

    recovered = lost = unchanged_miss = 0
    inv_removed = inv_added = 0
    red_delta = 0.0
    per_domain = defaultdict(lambda: defaultdict(int))

    for k in common:
        eid = k[0]
        domain = case_map[eid]
        ex = examples[eid]
        adapted = adapt_example_for_edge_mode(ex, "edge_calibrated", topic_matching_mode="curated_alias")
        c1 = compare_graphs(adapted, _graph(v1_map[k]))
        c2 = compare_graphs(adapted, _graph(v2_map[k]))
        miss1 = {tuple(e) for e in c1["missing_dependencies"]}
        miss2 = {tuple(e) for e in c2["missing_dependencies"]}
        extra1 = {tuple(e) for e in c1["extra_dependencies"]}
        extra2 = {tuple(e) for e in c2["extra_dependencies"]}
        recovered += len(miss1 - miss2)
        lost += len(miss2 - miss1)
        unchanged_miss += len(miss1 & miss2)
        inv_removed += len(extra1 - extra2)
        inv_added += len(extra2 - extra1)
        s1 = score_graph(adapted, _graph(v1_map[k]))
        s2 = score_graph(adapted, _graph(v2_map[k]))
        red_delta += s2.redundant_transitive_edge_rate - s1.redundant_transitive_edge_rate
        per_domain[domain]["recovered"] += len(miss1 - miss2)
        per_domain[domain]["lost"] += len(miss2 - miss1)
        per_domain[domain]["invalid_removed"] += len(extra1 - extra2)
        per_domain[domain]["invalid_added"] += len(extra2 - extra1)

    n = max(len(common), 1)
    return {
        "aligned_generations": len(common),
        "required_edges_recovered": recovered,
        "required_edges_lost": lost,
        "required_edges_still_missing": unchanged_miss,
        "invalid_extras_removed": inv_removed,
        "invalid_extras_added": inv_added,
        "transitive_redundancy_rate_delta_mean": red_delta / n,
        "per_domain": {d: dict(v) for d, v in per_domain.items()},
    }


def decide(v1: dict, v2: dict, baseline: dict) -> str:
    if v2.get("n", 0) < 3 or v1.get("n", 0) < 3:
        return "NEEDS_FURTHER_REVIEW"
    edge_gain = float(v2.get("required_edge_f1") or 0) - float(v1.get("required_edge_f1") or 0)
    recall_gain = float(v2.get("required_edge_recall") or 0) - float(v1.get("required_edge_recall") or 0)
    topic_drop = float(v1.get("topic_f1") or 0) - float(v2.get("topic_f1") or 0)
    inv_delta = float(v2.get("invalid_extra_edge_rate") or 0) - float(v1.get("invalid_extra_edge_rate") or 0)
    # Primary success: required-edge quality. Mild topic drop is acceptable when edges improve.
    topic_ok = topic_drop <= 0.10 or (edge_gain >= 0.10 and topic_drop <= 0.15)
    inv_ok = inv_delta <= 0.08
    if edge_gain >= 0.03 and recall_gain >= 0.0 and topic_ok and inv_ok:
        return "KEEP_V2"
    if edge_gain > 0.05 and recall_gain > 0.05 and inv_delta <= 0.0 and topic_drop <= 0.15:
        return "KEEP_V2"
    if edge_gain <= 0.0 and recall_gain <= 0.0:
        return "KEEP_V1"
    if abs(edge_gain) < 0.03:
        return "NEEDS_FURTHER_REVIEW"
    if topic_drop > 0.15 or inv_delta > 0.12:
        return "KEEP_V1"
    return "NEEDS_FURTHER_REVIEW"


def run_inventory_v2_comparison(
    *,
    v1_artifact: str | Path,
    v2_artifact: str | Path,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    examples = {ex.id: ex for ex in load_dataset(Path(dataset_path) if dataset_path else DEFAULT_DS)}
    case_map = load_case_domain_map()
    v1_payload = json.loads(Path(v1_artifact).read_text(encoding="utf-8"))
    v2_payload = json.loads(Path(v2_artifact).read_text(encoding="utf-8"))

    baseline_rows = (v1_payload.get("systems") or {}).get("synapse", {}).get("example_results") or []
    prior_v1_rows = (v1_payload.get("systems") or {}).get("domain_curriculum_prior", {}).get("example_results") or []
    # v2 artifact may only contain domain_curriculum_prior (or both)
    systems_v2 = v2_payload.get("systems") or {}
    prior_v2_rows = (systems_v2.get("domain_curriculum_prior") or systems_v2.get("synapse") or {}).get(
        "example_results"
    ) or []
    if "domain_curriculum_prior" in systems_v2:
        prior_v2_rows = systems_v2["domain_curriculum_prior"].get("example_results") or []

    baseline = _agg_scores(baseline_rows, examples, case_map)
    prior_v1 = _agg_scores(prior_v1_rows, examples, case_map)
    prior_v2 = _agg_scores(prior_v2_rows, examples, case_map)
    impact = _edge_impact(prior_v1_rows, prior_v2_rows, examples, case_map)

    per_domain = {}
    for domain in TARGET_DOMAINS:
        def filt(rows):
            return [r for r in rows if case_map.get(r.get("example_id")) == domain]

        b = _agg_scores(filt(baseline_rows), examples, case_map)
        p1 = _agg_scores(filt(prior_v1_rows), examples, case_map)
        p2 = _agg_scores(filt(prior_v2_rows), examples, case_map)
        per_domain[domain] = {
            "baseline": b,
            "prior_v1": p1,
            "prior_v2": p2,
            "decision": decide(p1, p2, b),
        }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "matching": "curated_alias + edge_calibrated",
        "v1_artifact": str(v1_artifact),
        "v2_artifact": str(v2_artifact),
        "target_domains": list(TARGET_DOMAINS),
        "aggregate": {
            "baseline": baseline,
            "prior_v1": prior_v1,
            "prior_v2": prior_v2,
        },
        "edge_level_impact": impact,
        "per_domain": per_domain,
        "production_status": {
            "baseline": "PRODUCTION_DEFAULT",
            "domain_curriculum_prior": "OPT_IN_EXPERIMENTAL",
            "edge_classifier": "EXPERIMENTAL_ONLY",
        },
    }

    out = Path(output_dir) if output_dir else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    jp = out / f"{ts}_inventory_v2_targeted_comparison.json"
    mp = out / f"{ts}_inventory_v2_targeted_comparison.md"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def row(label, m):
        return (
            f"| {label} | {m.get('topic_f1', 0):.3f} | {m.get('required_edge_precision', 0):.3f} | "
            f"{m.get('required_edge_recall', 0):.3f} | {m.get('required_edge_f1', 0):.3f} | "
            f"{m.get('missing_required_edge_rate', 0):.3f} | {m.get('invalid_extra_edge_rate', 0):.3f} | "
            f"{m.get('dependency_direction_error_rate', 0):.3f} | {m.get('redundant_transitive_edge_rate', 0):.3f} | "
            f"{m.get('latency_ms', 0):.0f} | {m.get('estimated_cost_usd', 0):.6f} |"
        )

    lines = [
        "# Inventory v2 Targeted Comparison",
        "",
        f"- Domains: `{', '.join(TARGET_DOMAINS)}`",
        f"- Matching: curated_alias + edge_calibrated",
        "",
        "## Aggregate (both domains)",
        "",
        "| System | Topic F1 | Edge P | Edge R | Edge F1 | Miss Edge | Invalid Extra | Dir Err | Trans Redund | Latency ms | Cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        row("Baseline", baseline),
        row("Prior v1", prior_v1),
        row("Prior v2", prior_v2),
        "",
        "## Edge-level impact (v1 → v2)",
        "",
        f"- Required recovered: **{impact['required_edges_recovered']}**",
        f"- Required lost: **{impact['required_edges_lost']}**",
        f"- Invalid extras removed: **{impact['invalid_extras_removed']}**",
        f"- Invalid extras added: **{impact['invalid_extras_added']}**",
        f"- Transitive redundancy Δ: **{impact['transitive_redundancy_rate_delta_mean']:.3f}**",
        "",
        "## Per-domain decisions",
        "",
    ]
    for domain, block in per_domain.items():
        lines.append(f"### {domain}: **{block['decision']}**")
        lines.append("")
        lines.append("| System | Topic F1 | Edge R | Edge F1 | Invalid Extra | Halluc |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for label, key in [("Baseline", "baseline"), ("Prior v1", "prior_v1"), ("Prior v2", "prior_v2")]:
            m = block[key]
            lines.append(
                f"| {label} | {m.get('topic_f1', 0):.3f} | {m.get('required_edge_recall', 0):.3f} | "
                f"{m.get('required_edge_f1', 0):.3f} | {m.get('invalid_extra_edge_rate', 0):.3f} | "
                f"{m.get('hallucinated_topic_rate', 0):.3f} |"
            )
        lines.append("")
    mp.write_text("\n".join(lines), encoding="utf-8")
    return mp, jp
