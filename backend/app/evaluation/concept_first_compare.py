"""Controlled BASELINE vs CONCEPT_FIRST comparison artifacts (no gold/alias changes)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.inspect import _graph_from_row
from app.evaluation.metrics import score_graph
from app.evaluation.node_edge_attribution import (
    INVALID_EDGE_ATTRS,
    MISSING_EDGE_ATTRS,
    attribute_invalid_extra_edge,
    attribute_missing_required_edge,
)
from app.evaluation.schemas import EvalExample, GeneratedGraph

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "failure_analysis"

_METRIC_KEYS = (
    "topic_precision",
    "topic_recall",
    "topic_f1",
    "required_edge_precision",
    "required_edge_recall",
    "required_edge_f1",
    "missing_required_edge_rate",
    "invalid_extra_edge_rate",
    "dependency_direction_error_rate",
    "redundant_transitive_edge_rate",
)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _p50(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return s[len(s) // 2]


def _rows_by_id(system_block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in system_block.get("example_results") or []:
        eid = str(row.get("example_id") or "")
        if eid and int(row.get("repetition") or 0) == 0:
            out[eid] = row
    return out


def _attribution_rates(example: EvalExample, graph: GeneratedGraph) -> dict[str, Any]:
    from app.evaluation.metrics import compare_graphs

    missing_c: Counter[str] = Counter()
    invalid_c: Counter[str] = Counter()
    cmp = compare_graphs(example, graph)
    for edge in cmp["missing_dependencies"]:
        attr = attribute_missing_required_edge(str(edge[0]), str(edge[1]), example, graph)
        missing_c[attr["primary_attribution"]] += 1
    for edge in cmp["extra_dependencies"]:
        attr = attribute_invalid_extra_edge(str(edge[0]), str(edge[1]), example, graph)
        invalid_c[attr["primary_attribution"]] += 1

    m_total = sum(missing_c.values())
    i_total = sum(invalid_c.values())
    return {
        "missing_total": m_total,
        "invalid_total": i_total,
        "missing_rates": {k: (missing_c[k] / m_total if m_total else 0.0) for k in MISSING_EDGE_ATTRS},
        "invalid_rates": {k: (invalid_c[k] / i_total if i_total else 0.0) for k in INVALID_EDGE_ATTRS},
        "missing_counts": dict(missing_c),
        "invalid_counts": dict(invalid_c),
        "missing_edges": [list(e) for e in cmp["missing_dependencies"]],
        "extra_edges": [list(e) for e in cmp["extra_dependencies"]],
    }


def _score_system(
    examples: dict[str, EvalExample],
    rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scores = []
    latencies = []
    costs = []
    attr_missing: Counter[str] = Counter()
    attr_invalid: Counter[str] = Counter()
    per_case: dict[str, Any] = {}
    for eid, ex in examples.items():
        row = rows.get(eid)
        if not row or not row.get("parse_ok", True):
            continue
        graph = _graph_from_row(row)
        # Attach generation_meta if present
        if row.get("generation_meta"):
            graph.generation_meta = dict(row["generation_meta"])
        adapted = adapt_example_for_edge_mode(
            ex,
            "edge_calibrated",
            topic_matching_mode="curated_alias",
        )
        sc = score_graph(adapted, graph)
        scores.append(sc)
        latencies.append(float(row.get("total_latency_ms") or 0.0))
        if row.get("cost_usd") is not None:
            costs.append(float(row["cost_usd"]))
        atr = _attribution_rates(adapted, graph)
        attr_missing.update(atr["missing_counts"])
        attr_invalid.update(atr["invalid_counts"])
        per_case[eid] = {
            "scores": {
                k: getattr(sc, k)
                for k in _METRIC_KEYS
                if hasattr(sc, k)
            },
            "topics": list(graph.topics),
            "dependencies": [list(d) for d in graph.dependencies],
            "latency_ms": float(row.get("total_latency_ms") or 0.0),
            "cost_usd": row.get("cost_usd"),
            "generation_meta": row.get("generation_meta"),
            "attribution": atr,
        }

    m_total = sum(attr_missing.values())
    i_total = sum(attr_invalid.values())
    metrics = {k: _mean([float(getattr(s, k)) for s in scores]) for k in _METRIC_KEYS}
    metrics["ENDPOINT_GRANULARITY_MISMATCH_RATE"] = (
        attr_missing.get("ENDPOINT_GRANULARITY_MISMATCH", 0) / m_total if m_total else 0.0
    )
    metrics["ENDPOINT_GRANULARITY_DRIFT_RATE"] = (
        attr_invalid.get("ENDPOINT_GRANULARITY_DRIFT", 0) / i_total if i_total else 0.0
    )
    metrics["ENDPOINT_ABSTRACTION_MISMATCH_RATE"] = (
        attr_missing.get("ENDPOINT_ABSTRACTION_MISMATCH", 0) / m_total if m_total else 0.0
    )
    metrics["CURRICULUM_SCOPE_DRIFT_RATE"] = (
        attr_invalid.get("CURRICULUM_SCOPE_DRIFT", 0) / i_total if i_total else 0.0
    )
    for cat in MISSING_EDGE_ATTRS:
        metrics[f"{cat}_RATE"] = attr_missing.get(cat, 0) / m_total if m_total else 0.0
    for cat in INVALID_EDGE_ATTRS:
        metrics[f"{cat}_RATE"] = attr_invalid.get(cat, 0) / i_total if i_total else 0.0

    return {
        "n": len(scores),
        "metrics": metrics,
        "latency": {"p50_ms": _p50(latencies), "mean_ms": _mean(latencies) if latencies else None},
        "cost": {"average_usd": _mean(costs) if costs else None, "n_with_cost": len(costs)},
        "attribution_missing_counts": dict(attr_missing),
        "attribution_invalid_counts": dict(attr_invalid),
        "per_case": per_case,
    }


def _classify_case(base: dict[str, Any], cf: dict[str, Any]) -> str:
    bt = float(base["scores"].get("topic_f1") or 0.0)
    ct = float(cf["scores"].get("topic_f1") or 0.0)
    be = float(base["scores"].get("required_edge_f1") or 0.0)
    ce = float(cf["scores"].get("required_edge_f1") or 0.0)
    bg = float(base["attribution"].get("missing_rates", {}).get("ENDPOINT_GRANULARITY_MISMATCH") or 0.0)
    cg = float(cf["attribution"].get("missing_rates", {}).get("ENDPOINT_GRANULARITY_MISMATCH") or 0.0)
    # Prefer structural quality: topic+edge F1, then granularity
    delta = (ct - bt) + (ce - be) + 0.25 * (bg - cg)
    if delta > 0.03:
        return "IMPROVED"
    if delta < -0.03:
        return "REGRESSED"
    return "UNCHANGED"


def _pick_representative_ids(
    examples: dict[str, EvalExample],
    base_cases: dict[str, Any],
    cf_cases: dict[str, Any],
    *,
    n: int = 10,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for eid in examples:
        b = base_cases.get(eid)
        c = cf_cases.get(eid)
        if not b or not c:
            continue
        bm = b["attribution"].get("missing_counts", {})
        bi = b["attribution"].get("invalid_counts", {})
        interest = (
            bm.get("ENDPOINT_GRANULARITY_MISMATCH", 0)
            + bi.get("ENDPOINT_GRANULARITY_DRIFT", 0)
            + bm.get("ENDPOINT_ABSTRACTION_MISMATCH", 0)
            + bi.get("CURRICULUM_SCOPE_DRIFT", 0)
            + bm.get("BOTH_ENDPOINTS_MISSING", 0)
        )
        scored.append((float(interest), eid))
    scored.sort(reverse=True)
    picked = [eid for _, eid in scored[:n]]
    # Fill if needed
    for eid in examples:
        if len(picked) >= n:
            break
        if eid not in picked and eid in base_cases and eid in cf_cases:
            picked.append(eid)
    return picked


def write_normalization_analysis(
    quality_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    system: str = "concept_first",
) -> Path:
    target = Path(quality_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    block = (payload.get("systems") or {}).get(system) or {}
    rows = _rows_by_id(block)
    dataset = load_dataset(
        Path(payload.get("dataset_path") or _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl")
        if payload.get("dataset_path")
        else _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    )
    # Prefer stem match
    ds_stem = payload.get("dataset") or payload.get("dataset_version") or "learning_graph_quality_v1"
    ds_path = _REPO_ROOT / "data" / "eval" / f"{ds_stem}.jsonl"
    if ds_path.is_file():
        dataset = load_dataset(ds_path)

    cases = []
    totals = Counter()
    for ex in dataset:
        row = rows.get(ex.id)
        if not row:
            continue
        meta = row.get("generation_meta") or {}
        norm = meta.get("normalization") or {}
        atr = None
        if row.get("parse_ok", True):
            adapted = adapt_example_for_edge_mode(
                ex, "edge_calibrated", topic_matching_mode="curated_alias"
            )
            atr = _attribution_rates(adapted, _graph_from_row(row))
        case = {
            "case_id": ex.id,
            "candidate_concepts": meta.get("candidate_concepts") or [],
            "normalized_inventory": meta.get("normalized_inventory") or row.get("generated_topics") or [],
            "accepted_count": norm.get("accepted_count", 0),
            "merged_count": norm.get("merged_count", 0),
            "duplicate_rejection_count": norm.get("duplicate_rejection_count", 0),
            "out_of_scope_rejection_count": norm.get("out_of_scope_rejection_count", 0),
            "granularity_conflict_count": norm.get("granularity_conflict_count", 0),
            "abstraction_conflict_count": norm.get("abstraction_conflict_count", 0),
            "decomposition_conflict_count": norm.get("decomposition_conflict_count", 0),
            "unresolved_count": norm.get("unresolved_count", 0),
            "normalization_decisions": norm.get("decisions") or [],
            "downstream_dependency_errors": {
                "skipped_dependencies": row.get("skipped_dependencies") or [],
                "errors": meta.get("errors") or [],
                "missing_required": atr["missing_edges"] if atr else [],
                "invalid_extra": atr["extra_edges"] if atr else [],
                "attribution_missing": atr["missing_counts"] if atr else {},
                "attribution_invalid": atr["invalid_counts"] if atr else {},
            },
            "stage_latency_ms": meta.get("stage_latency_ms"),
            "status": meta.get("status"),
        }
        cases.append(case)
        for k in (
            "accepted_count",
            "merged_count",
            "duplicate_rejection_count",
            "out_of_scope_rejection_count",
            "granularity_conflict_count",
            "abstraction_conflict_count",
            "decomposition_conflict_count",
            "unresolved_count",
        ):
            totals[k] += int(case.get(k) or 0)
        totals["candidate_concepts"] += len(case["candidate_concepts"])
        totals["normalized_inventory"] += len(case["normalized_inventory"])

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    path = out_dir / f"{ts}_concept_normalization_analysis.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_artifact": str(target),
                "system": system,
                "n_cases": len(cases),
                "totals": dict(totals),
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def compare_concept_first_runs(
    quality_path: str | Path,
    *,
    baseline_system: str = "synapse",
    concept_system: str = "concept_first",
    output_dir: str | Path | None = None,
    max_cases: int = 10,
) -> tuple[Path, Path]:
    """Rescore with curated_alias + edge_calibrated; write analysis JSON + comparison MD."""
    target = Path(quality_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    systems = payload.get("systems") or {}
    ds_stem = payload.get("dataset") or payload.get("dataset_version") or "learning_graph_quality_v1"
    ds_path = _REPO_ROOT / "data" / "eval" / f"{ds_stem}.jsonl"
    if not ds_path.is_file():
        ds_path = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    dataset = {ex.id: ex for ex in load_dataset(ds_path)}

    base_rows = _rows_by_id(systems.get(baseline_system) or {})
    cf_rows = _rows_by_id(systems.get(concept_system) or {})
    base = _score_system(dataset, base_rows)
    cf = _score_system(dataset, cf_rows)

    norm_path = write_normalization_analysis(target, output_dir=output_dir, system=concept_system)

    picked = _pick_representative_ids(dataset, base["per_case"], cf["per_case"], n=max_cases)
    case_rows = []
    for eid in picked:
        ex = dataset[eid]
        b = base["per_case"][eid]
        c = cf["per_case"][eid]
        label = _classify_case(b, c)
        meta = c.get("generation_meta") or {}
        case_rows.append(
            {
                "case_id": eid,
                "classification": label,
                "learning_objective": ex.goal,
                "gold_topics": list(ex.gold_topics),
                "baseline_topics": b["topics"],
                "concept_first_topics": c["topics"],
                "normalization_decisions": (meta.get("normalization") or {}).get("decisions") or [],
                "gold_dependencies": [list(d) for d in ex.gold_dependencies],
                "baseline_dependencies": b["dependencies"],
                "concept_first_dependencies": c["dependencies"],
                "missing_edge_changes": {
                    "baseline_missing": b["attribution"]["missing_edges"],
                    "concept_first_missing": c["attribution"]["missing_edges"],
                    "baseline_missing_attr": b["attribution"]["missing_counts"],
                    "concept_first_missing_attr": c["attribution"]["missing_counts"],
                },
                "invalid_extra_changes": {
                    "baseline_extra": b["attribution"]["extra_edges"],
                    "concept_first_extra": c["attribution"]["extra_edges"],
                    "baseline_invalid_attr": b["attribution"]["invalid_counts"],
                    "concept_first_invalid_attr": c["attribution"]["invalid_counts"],
                },
                "latency_ms": {"baseline": b["latency_ms"], "concept_first": c["latency_ms"]},
                "cost_usd": {"baseline": b["cost_usd"], "concept_first": c["cost_usd"]},
                "scores": {"baseline": b["scores"], "concept_first": c["scores"]},
            }
        )

    delta = {
        k: float(cf["metrics"].get(k, 0.0)) - float(base["metrics"].get(k, 0.0))
        for k in sorted(set(base["metrics"]) | set(cf["metrics"]))
    }

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    md_path = out_dir / f"{ts}_concept_first_comparison.md"
    json_side = out_dir / f"{ts}_concept_first_comparison.json"

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(target),
        "exploratory": True,
        "note": "n=1 generation per case unless repetitions>1; do not claim statistical significance.",
        "benchmark_config": {
            "model": payload.get("model"),
            "dataset": ds_stem,
            "dataset_size": len(dataset),
            "n_scored_baseline": base["n"],
            "n_scored_concept_first": cf["n"],
            "generations_per_case": payload.get("repetitions", 1),
            "seed": payload.get("seed"),
            "matching_mode": "curated_alias",
            "edge_evaluation_mode": "edge_calibrated",
            "baseline_system": baseline_system,
            "concept_system": concept_system,
        },
        "baseline": {
            "metrics": base["metrics"],
            "latency": base["latency"],
            "cost": base["cost"],
            "attribution_missing_counts": base["attribution_missing_counts"],
            "attribution_invalid_counts": base["attribution_invalid_counts"],
        },
        "concept_first": {
            "metrics": cf["metrics"],
            "latency": cf["latency"],
            "cost": cf["cost"],
            "attribution_missing_counts": cf["attribution_missing_counts"],
            "attribution_invalid_counts": cf["attribution_invalid_counts"],
        },
        "delta": delta,
        "normalization_analysis": str(norm_path),
        "representative_cases": case_rows,
    }
    json_side.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Concept-First vs Baseline Comparison",
        "",
        f"- Source: `{target}`",
        f"- Matching: `curated_alias` + `edge_calibrated`",
        f"- Exploratory: **yes** (do not claim statistical significance)",
        f"- Normalization analysis: `{norm_path}`",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Baseline | Concept-First | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k in (
        "topic_precision",
        "topic_recall",
        "topic_f1",
        "required_edge_precision",
        "required_edge_recall",
        "required_edge_f1",
        "missing_required_edge_rate",
        "invalid_extra_edge_rate",
        "ENDPOINT_GRANULARITY_MISMATCH_RATE",
        "ENDPOINT_GRANULARITY_DRIFT_RATE",
        "ENDPOINT_ABSTRACTION_MISMATCH_RATE",
        "CURRICULUM_SCOPE_DRIFT_RATE",
        "dependency_direction_error_rate",
    ):
        lines.append(
            f"| {k} | {base['metrics'].get(k, 0):.3f} | {cf['metrics'].get(k, 0):.3f} | {delta.get(k, 0):+.3f} |"
        )
    lines.extend(
        [
            "",
            f"| p50 latency (ms) | {base['latency'].get('p50_ms')} | {cf['latency'].get('p50_ms')} | |",
            f"| average cost (USD) | {base['cost'].get('average_usd')} | {cf['cost'].get('average_usd')} | |",
            "",
            "## Representative cases",
            "",
        ]
    )
    for row in case_rows:
        lines.extend(
            [
                f"### {row['case_id']} — {row['classification']}",
                "",
                f"**Learning objective:** {row['learning_objective']}",
                "",
                f"- Gold topics: {row['gold_topics']}",
                f"- Baseline topics: {row['baseline_topics']}",
                f"- Concept-First topics: {row['concept_first_topics']}",
                f"- Normalization decisions: `{json.dumps(row['normalization_decisions'][:8], ensure_ascii=False)}`",
                f"- Gold deps: {row['gold_dependencies']}",
                f"- Baseline deps: {row['baseline_dependencies']}",
                f"- Concept-First deps: {row['concept_first_dependencies']}",
                f"- Missing-edge attr (baseline → CF): "
                f"{row['missing_edge_changes']['baseline_missing_attr']} → "
                f"{row['missing_edge_changes']['concept_first_missing_attr']}",
                f"- Invalid-extra attr (baseline → CF): "
                f"{row['invalid_extra_changes']['baseline_invalid_attr']} → "
                f"{row['invalid_extra_changes']['concept_first_invalid_attr']}",
                f"- Latency ms: {row['latency_ms']}",
                f"- Cost USD: {row['cost_usd']}",
                "",
            ]
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_side
