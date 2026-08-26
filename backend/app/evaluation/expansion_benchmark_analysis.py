"""Offline expansion benchmark analysis: availability vs supported-domain vs product behavior.

Rescores with curated_alias + edge_calibrated. Does not mutate inventories.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.curriculum.inventory import load_case_domain_map
from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.metrics import score_graph
from app.evaluation.schemas import GeneratedGraph

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "curriculum"
DEFAULT_DS = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"

NEW_EXPANSION_DOMAINS = (
    "cloud_computing",
    "frontend_engineering",
    "backend_engineering",
    "databases",
    "data_engineering",
    "security",
    "machine_learning",
)


def _safe(n: float, d: float) -> float:
    return (n / d) if d else 0.0


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _graph_from_row(row: dict[str, Any]) -> GeneratedGraph:
    topics = row.get("generated_topics")
    if topics is None:
        topics = row.get("topics") or []
    deps = row.get("generated_dependencies")
    if deps is None:
        deps = row.get("dependencies") or []
    return GeneratedGraph(
        topics=list(topics),
        dependencies=[tuple(e) for e in deps],
        parse_ok=bool(row.get("parse_ok", True)),
        error=row.get("error"),
        generation_meta=dict(row.get("generation_meta") or {}),
    )


def _aggregate(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {}
    keys = scores[0].keys()
    return {k: _mean([float(s[k]) for s in scores]) for k in keys}


def run_expansion_benchmark_analysis(
    artifact_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    examples = {ex.id: ex for ex in load_dataset(Path(dataset_path) if dataset_path else DEFAULT_DS)}
    case_map = load_case_domain_map()
    systems = payload.get("systems") or {}

    # Per system / domain / scope metrics
    by_sys_domain: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    by_sys_scope: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    fallback_counts: dict[str, int] = defaultdict(int)
    resolved_counts: dict[str, int] = defaultdict(int)
    latencies: dict[str, list[float]] = defaultdict(list)
    costs: dict[str, list[float]] = defaultdict(list)

    for sys, block in systems.items():
        for row in block.get("example_results") or []:
            eid = str(row.get("example_id"))
            ex = examples.get(eid)
            if not ex:
                continue
            adapted = adapt_example_for_edge_mode(
                ex, "edge_calibrated", topic_matching_mode="curated_alias"
            )
            graph = _graph_from_row(row)
            scores_obj = score_graph(adapted, graph)
            scores = {
                "topic_precision": scores_obj.topic_precision,
                "topic_recall": scores_obj.topic_recall,
                "topic_f1": scores_obj.topic_f1,
                "required_edge_precision": scores_obj.required_edge_precision,
                "required_edge_recall": scores_obj.required_edge_recall,
                "required_edge_f1": scores_obj.required_edge_f1,
                "missing_required_edge_rate": scores_obj.missing_required_edge_rate,
                "invalid_extra_edge_rate": scores_obj.invalid_extra_edge_rate,
                "hallucinated_topic_rate": scores_obj.hallucinated_topic_rate,
                "dependency_direction_error_rate": scores_obj.dependency_direction_error_rate,
            }
            domain = case_map.get(eid) or f"unmapped:{ex.category}"
            by_sys_domain[sys][domain].append(scores)
            by_sys_scope[sys]["all_rows_in_artifact"].append(scores)
            if eid in case_map:
                by_sys_scope[sys]["supported_mapped"].append(scores)
                if case_map[eid] in NEW_EXPANSION_DOMAINS:
                    by_sys_scope[sys]["new_expansion_domains"].append(scores)
                else:
                    by_sys_scope[sys]["legacy_domains"].append(scores)

            meta = row.get("generation_meta") or {}
            if meta.get("fallback_reason"):
                fallback_counts[sys] += 1
            if meta.get("domain") or case_map.get(eid):
                resolved_counts[sys] += 1
            lat = row.get("llm_latency_ms")
            if lat is None:
                lat = row.get("total_latency_ms")
            if lat is None:
                lat = (row.get("latency") or {}).get("llm_ms")
            if lat is not None:
                latencies[sys].append(float(lat))
            cost = row.get("cost_usd")
            if cost is None:
                cost = (row.get("generation_meta") or {}).get("estimated_cost_usd")
            if cost is None:
                cost = (row.get("generation_meta") or {}).get("estimated_cost")
            if cost is not None:
                costs[sys].append(float(cost))

    domain_table = []
    all_domains = sorted({d for sys in by_sys_domain for d in by_sys_domain[sys]})
    for domain in all_domains:
        row: dict[str, Any] = {"domain": domain, "systems": {}}
        for sys in systems:
            agg = _aggregate(by_sys_domain[sys].get(domain, []))
            row["systems"][sys] = {
                **agg,
                "n": len(by_sys_domain[sys].get(domain, [])),
            }
        # Decision heuristic (experimental keep — no auto promote)
        prior = row["systems"].get("domain_curriculum_prior") or {}
        base = row["systems"].get("synapse") or {}
        decision = "INSUFFICIENT_EVIDENCE"
        if prior.get("n", 0) >= 3 and base.get("n", 0) >= 3:
            topic_gain = float(prior.get("topic_f1") or 0) - float(base.get("topic_f1") or 0)
            edge_gain = float(prior.get("required_edge_f1") or 0) - float(base.get("required_edge_f1") or 0)
            if topic_gain >= 0.05 or edge_gain >= 0.05:
                if float(prior.get("required_edge_recall") or 0) + 0.02 < float(
                    base.get("required_edge_recall") or 0
                ) and edge_gain < 0:
                    decision = "NEEDS_REVIEW"
                else:
                    decision = "KEEP_EXPERIMENTAL"
            elif topic_gain < -0.05 or edge_gain < -0.05:
                decision = "NEEDS_REVIEW"
            else:
                decision = "INSUFFICIENT_EVIDENCE"
        row["decision"] = decision
        domain_table.append(row)

    scopes = {}
    for sys in systems:
        scopes[sys] = {
            scope: {**_aggregate(vals), "n": len(vals)}
            for scope, vals in by_sys_scope[sys].items()
        }

    n_mapped = len(case_map)
    n_total = len(examples)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(artifact_path),
        "matching": "curated_alias + edge_calibrated",
        "generations": payload.get("generations") or payload.get("repetitions"),
        "inventory_availability": {
            "mapped_cases": n_mapped,
            "total_cases": n_total,
            "availability_rate": _safe(n_mapped, n_total),
            "unmapped_cases": n_total - n_mapped,
        },
        "A_inventory_availability": {
            "mapped_domains": sorted(set(case_map.values())),
            "mapped_case_count": n_mapped,
            "coverage_rate": _safe(n_mapped, n_total),
        },
        "B_supported_domain_performance": scopes,
        "C_full_dataset_product_behavior_note": (
            "This artifact is mapped-cases-only unless --full-dataset was used. "
            "Unmapped cases are not algorithm failures when product falls back to baseline."
        ),
        "fallback_rate_by_system": {
            sys: _safe(fallback_counts[sys], len((systems[sys].get("example_results") or [])))
            for sys in systems
        },
        "domain_resolution_rate_by_system": {
            sys: _safe(resolved_counts[sys], len((systems[sys].get("example_results") or [])))
            for sys in systems
        },
        "latency_mean_ms": {sys: _mean(latencies[sys]) for sys in systems},
        "estimated_cost_mean_usd": {sys: _mean(costs[sys]) for sys in systems},
        "per_domain": domain_table,
        "production_status": {
            "baseline": "PRODUCTION_DEFAULT",
            "domain_curriculum_prior": "OPT_IN_EXPERIMENTAL",
            "domain_prior_edge_classifier": "EXPERIMENTAL_ONLY",
        },
        "new_expansion_domains": list(NEW_EXPANSION_DOMAINS),
    }

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_expansion_benchmark_analysis.json"
    md_path = out_dir / f"{ts}_expansion_benchmark_analysis.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Domain Prior Expansion Benchmark Analysis",
        "",
        f"- Source: `{artifact_path}`",
        f"- Matching: curated_alias + edge_calibrated",
        f"- Inventory availability: {n_mapped}/{n_total} ({_safe(n_mapped, n_total):.1%})",
        "",
        "## Supported-domain aggregates",
        "",
        "| System | Scope | Topic F1 | Req Edge F1 | Req Edge Recall | Invalid Extra | Halluc Topic | n |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sys, scope_map in scopes.items():
        for scope, m in scope_map.items():
            lines.append(
                f"| {sys} | {scope} | {m.get('topic_f1', 0):.3f} | {m.get('required_edge_f1', 0):.3f} | "
                f"{m.get('required_edge_recall', 0):.3f} | {m.get('invalid_extra_edge_rate', 0):.3f} | "
                f"{m.get('hallucinated_topic_rate', 0):.3f} | {m.get('n', 0)} |"
            )
    lines.extend(
        [
            "",
            "## Per-domain decisions",
            "",
            "| Domain | Baseline Topic F1 | Prior Topic F1 | Baseline Edge F1 | Prior Edge F1 | Decision |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in domain_table:
        b = row["systems"].get("synapse") or {}
        p = row["systems"].get("domain_curriculum_prior") or {}
        lines.append(
            f"| {row['domain']} | {b.get('topic_f1', 0):.3f} | {p.get('topic_f1', 0):.3f} | "
            f"{b.get('required_edge_f1', 0):.3f} | {p.get('required_edge_f1', 0):.3f} | {row['decision']} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path
