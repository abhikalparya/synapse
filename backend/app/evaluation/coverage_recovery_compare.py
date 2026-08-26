"""Offline comparison: baseline synapse vs baseline_coverage_recovery (eval-only)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.inspect import _graph_from_row
from app.evaluation.metrics import compare_graphs, match_topic, normalize_topic, score_graph

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "failure_analysis"


def _row_by_id(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in block.get("example_results") or []:
        eid = str(row.get("example_id") or "")
        if eid and eid not in out:
            # first repetition if multi-gen
            out[eid] = row
    return out


def classify_recovery_candidate(
    *,
    title: str,
    accepted: bool,
    rejection_reason: str | None,
    example,
    baseline_topics: list[str],
) -> str:
    if not accepted:
        if rejection_reason in {"optional_nice_to_have", "related_but_not_required", "out_of_scope"}:
            return "REJECTED_SAFELY"
        if rejection_reason in {
            "duplicate_edges",
            "truncated_by_max_candidates",
            "empty_title",
            "unknown_category",
        }:
            return "REJECTED_SAFELY"
        if rejection_reason in {"would create a cycle", "self_loop"} or (
            rejection_reason and "cycle" in rejection_reason
        ):
            return "REJECTED_SAFELY"
        return "REJECTED_SAFELY"

    # Accepted — compare to gold
    hit = match_topic(title, example)
    if hit is not None:
        # Required or optional gold
        req = {normalize_topic(t) for t in example.required_topic_list()}
        if normalize_topic(hit) in req:
            return "CORRECT_RECOVERY"
        return "PARTIAL_RECOVERY"
    # Already in baseline?
    if any(normalize_topic(title) == normalize_topic(t) for t in baseline_topics):
        return "DUPLICATE_RECOVERY"
    return "INCORRECT_RECOVERY"


def compare_coverage_recovery_runs(
    artifact_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    target = Path(artifact_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    systems = payload.get("systems") or {}
    if "synapse" not in systems or "baseline_coverage_recovery" not in systems:
        raise ValueError("Artifact must contain systems 'synapse' and 'baseline_coverage_recovery'")

    ds_stem = payload.get("dataset") or "learning_graph_quality_v1"
    ds_path = Path(dataset_path) if dataset_path else _REPO_ROOT / "data" / "eval" / f"{ds_stem}.jsonl"
    if not ds_path.is_file():
        ds_path = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    examples = {ex.id: ex for ex in load_dataset(ds_path)}

    base_rows = _row_by_id(systems["synapse"])
    cov_rows = _row_by_id(systems["baseline_coverage_recovery"])

    case_rows: list[dict[str, Any]] = []
    deltas = {
        "topic_f1": [],
        "required_edge_f1": [],
        "required_edge_recall": [],
        "missing_required_edge_rate": [],
        "invalid_extra_edge_rate": [],
        "hallucinated_topic_rate": [],
    }
    recovery_class_counts: Counter[str] = Counter()
    total_applied = 0
    total_candidates = 0
    recovered_required_topics = 0
    recoverable_missing_topics = 0
    recovered_required_edges = 0
    recoverable_missing_edges = 0
    improved = 0
    regressed = 0

    for eid, brow in base_rows.items():
        crow = cov_rows.get(eid)
        ex = examples.get(eid)
        if not crow or not ex:
            continue
        adapted = adapt_example_for_edge_mode(
            ex, "edge_calibrated", topic_matching_mode="curated_alias"
        )
        bg = _graph_from_row(brow)
        cg = _graph_from_row(crow)
        bs = score_graph(adapted, bg) if bg.parse_ok else None
        cs = score_graph(adapted, cg) if cg.parse_ok else None
        bcmp = compare_graphs(adapted, bg) if bg.parse_ok else None
        ccmp = compare_graphs(adapted, cg) if cg.parse_ok else None

        meta = crow.get("generation_meta") or {}
        accepted = meta.get("recovery_accepted") or []
        all_cands = meta.get("recovery_all_candidates") or []
        total_candidates += int(meta.get("recovery_candidate_count") or len(all_cands) or 0)
        total_applied += int(meta.get("recovery_applied_count") or len(accepted) or 0)

        baseline_missing_topics = set(bcmp["missing_topics"]) if bcmp else set()
        cov_missing_topics = set(ccmp["missing_topics"]) if ccmp else set()
        baseline_missing_edges = {tuple(e) for e in (bcmp["missing_dependencies"] if bcmp else [])}
        cov_missing_edges = {tuple(e) for e in (ccmp["missing_dependencies"] if ccmp else [])}

        recoverable_missing_topics += len(baseline_missing_topics)
        recovered_required_topics += len(baseline_missing_topics - cov_missing_topics)
        recoverable_missing_edges += len(baseline_missing_edges)
        recovered_required_edges += len(baseline_missing_edges - cov_missing_edges)

        for c in accepted:
            title = str(c.get("title") or "")
            label = classify_recovery_candidate(
                title=title,
                accepted=True,
                rejection_reason=c.get("rejection_reason"),
                example=adapted,
                baseline_topics=list(bg.topics),
            )
            recovery_class_counts[label] += 1
        for c in meta.get("recovery_rejected") or []:
            label = classify_recovery_candidate(
                title=str(c.get("title") or ""),
                accepted=False,
                rejection_reason=c.get("rejection_reason"),
                example=adapted,
                baseline_topics=list(bg.topics),
            )
            recovery_class_counts[label] += 1

        row = {
            "case_id": eid,
            "goal": ex.goal,
            "baseline": None
            if bs is None
            else {
                "topic_f1": bs.topic_f1,
                "required_edge_f1": bs.required_edge_f1,
                "required_edge_recall": bs.required_edge_recall,
                "missing_required_edge_rate": bs.missing_required_edge_rate,
                "invalid_extra_edge_rate": bs.invalid_extra_edge_rate,
                "hallucinated_topic_rate": bs.hallucinated_topic_rate,
            },
            "coverage_recovery": None
            if cs is None
            else {
                "topic_f1": cs.topic_f1,
                "required_edge_f1": cs.required_edge_f1,
                "required_edge_recall": cs.required_edge_recall,
                "missing_required_edge_rate": cs.missing_required_edge_rate,
                "invalid_extra_edge_rate": cs.invalid_extra_edge_rate,
                "hallucinated_topic_rate": cs.hallucinated_topic_rate,
            },
            "recovery_meta": {
                "candidate_count": meta.get("recovery_candidate_count"),
                "applied_count": meta.get("recovery_applied_count"),
                "rejected_count": meta.get("recovery_rejected_count"),
                "new_topics": meta.get("recovery_new_topics"),
                "new_edges": meta.get("recovery_new_edges"),
                "llm_latency_ms": meta.get("recovery_llm_latency_ms"),
                "cost_usd": meta.get("recovery_cost_usd"),
            },
            "recovered_topics": sorted(baseline_missing_topics - cov_missing_topics),
            "recovered_edges": [list(e) for e in sorted(baseline_missing_edges - cov_missing_edges)],
        }
        if bs and cs:
            for k in deltas:
                bv = getattr(bs, k if k != "required_edge_recall" else "required_edge_recall", None)
                if k == "required_edge_recall":
                    bv = bs.required_edge_recall
                    cv = cs.required_edge_recall
                else:
                    bv = getattr(bs, k)
                    cv = getattr(cs, k)
                deltas[k].append(cv - bv)
            if cs.required_edge_f1 > bs.required_edge_f1 + 1e-9 or cs.topic_f1 > bs.topic_f1 + 1e-9:
                if not (
                    cs.required_edge_f1 + 1e-9 < bs.required_edge_f1
                    and cs.topic_f1 + 1e-9 < bs.topic_f1
                ):
                    improved += 1
            if cs.required_edge_f1 + 1e-9 < bs.required_edge_f1 or cs.topic_f1 + 1e-9 < bs.topic_f1:
                if cs.required_edge_f1 < bs.required_edge_f1 - 1e-9 or cs.topic_f1 < bs.topic_f1 - 1e-9:
                    regressed += 1
        case_rows.append(row)

    def _mean(xs: list[float]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    base_metrics = (systems["synapse"].get("metrics") or {})
    cov_metrics = (systems["baseline_coverage_recovery"].get("metrics") or {})
    base_lat = (systems["synapse"].get("latency") or {})
    cov_lat = (systems["baseline_coverage_recovery"].get("latency") or {})
    base_cost = (systems["synapse"].get("cost") or {})
    cov_cost = (systems["baseline_coverage_recovery"].get("cost") or {})

    accepted_correct = recovery_class_counts.get("CORRECT_RECOVERY", 0)
    accepted_total = sum(
        recovery_class_counts[k]
        for k in ("CORRECT_RECOVERY", "PARTIAL_RECOVERY", "INCORRECT_RECOVERY", "DUPLICATE_RECOVERY", "UNNECESSARY_RECOVERY", "OUT_OF_SCOPE_RECOVERY")
        if k in recovery_class_counts
    )
    # accepted_total from applied classifications
    applied_labeled = (
        recovery_class_counts.get("CORRECT_RECOVERY", 0)
        + recovery_class_counts.get("PARTIAL_RECOVERY", 0)
        + recovery_class_counts.get("INCORRECT_RECOVERY", 0)
        + recovery_class_counts.get("DUPLICATE_RECOVERY", 0)
    )

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(target),
        "dataset": str(ds_path),
        "n_cases": len(case_rows),
        "aggregate": {
            "baseline": base_metrics,
            "coverage_recovery": cov_metrics,
            "delta_means": {k: _mean(v) for k, v in deltas.items()},
        },
        "latency": {
            "baseline_p50_ms": base_lat.get("p50_ms"),
            "coverage_p50_ms": cov_lat.get("p50_ms"),
            "additional_p50_ms": (
                (cov_lat.get("p50_ms") - base_lat.get("p50_ms"))
                if base_lat.get("p50_ms") is not None and cov_lat.get("p50_ms") is not None
                else None
            ),
        },
        "cost": {
            "baseline_avg_usd": base_cost.get("average_cost_usd"),
            "coverage_avg_usd": cov_cost.get("average_cost_usd"),
            "additional_avg_usd": (
                (cov_cost.get("average_cost_usd") - base_cost.get("average_cost_usd"))
                if base_cost.get("average_cost_usd") is not None
                and cov_cost.get("average_cost_usd") is not None
                else None
            ),
        },
        "recovery_metrics": {
            "candidate_count": total_candidates,
            "applied_count": total_applied,
            "recovery_candidate_precision": (
                accepted_correct / applied_labeled if applied_labeled else None
            ),
            "recovery_endpoint_recall": (
                recovered_required_topics / recoverable_missing_topics
                if recoverable_missing_topics
                else None
            ),
            "recovery_edge_recall": (
                recovered_required_edges / recoverable_missing_edges
                if recoverable_missing_edges
                else None
            ),
            "false_positive_rate": (
                recovery_class_counts.get("INCORRECT_RECOVERY", 0) / applied_labeled
                if applied_labeled
                else None
            ),
            "classification_counts": dict(recovery_class_counts),
            "recovered_required_topics": recovered_required_topics,
            "recoverable_missing_topics": recoverable_missing_topics,
            "recovered_required_edges": recovered_required_edges,
            "recoverable_missing_edges": recoverable_missing_edges,
            "cases_improved": improved,
            "cases_regressed": regressed,
        },
        "cases": case_rows,
    }

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_coverage_recovery_compare.json"
    md_path = out_dir / f"{ts}_coverage_recovery_compare.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_md(result), encoding="utf-8")
    return md_path, json_path


def _render_md(payload: dict[str, Any]) -> str:
    a = payload["aggregate"]
    b, c = a.get("baseline") or {}, a.get("coverage_recovery") or {}
    d = a.get("delta_means") or {}
    lines = [
        "# Coverage Recovery Comparison",
        "",
        f"- Source: `{payload['source_artifact']}`",
        f"- Cases: {payload['n_cases']}",
        "",
        "| Metric | Baseline | Coverage Recovery | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("topic_f1", "Topic F1"),
        ("required_edge_f1", "Required Edge F1"),
        ("required_edge_recall", "Required Edge Recall"),
        ("missing_required_edge_rate", "Missing Required Edge Rate"),
        ("invalid_extra_edge_rate", "Invalid Extra Edge Rate"),
        ("hallucinated_topic_rate", "Hallucinated Topic Rate"),
    ]:
        bv, cv = b.get(key), c.get(key)
        dv = d.get(key)
        def fmt(x):
            return "—" if x is None else f"{float(x):.3f}"
        lines.append(f"| {label} | {fmt(bv)} | {fmt(cv)} | {fmt(dv)} |")
    rm = payload["recovery_metrics"]
    lines.extend(
        [
            "",
            "## Recovery metrics",
            "",
            f"- Candidates: {rm['candidate_count']}",
            f"- Applied: {rm['applied_count']}",
            f"- Candidate precision: {rm['recovery_candidate_precision']}",
            f"- Endpoint recall: {rm['recovery_endpoint_recall']}",
            f"- Edge recall: {rm['recovery_edge_recall']}",
            f"- False positive rate: {rm['false_positive_rate']}",
            f"- Recovered topics: {rm['recovered_required_topics']} / {rm['recoverable_missing_topics']}",
            f"- Recovered edges: {rm['recovered_required_edges']} / {rm['recoverable_missing_edges']}",
            f"- Cases improved / regressed: {rm['cases_improved']} / {rm['cases_regressed']}",
            f"- Classifications: `{rm['classification_counts']}`",
            "",
            "## Latency / cost",
            "",
            f"- Baseline p50: {payload['latency'].get('baseline_p50_ms')}",
            f"- Coverage p50: {payload['latency'].get('coverage_p50_ms')}",
            f"- Additional p50: {payload['latency'].get('additional_p50_ms')}",
            f"- Baseline avg cost: {payload['cost'].get('baseline_avg_usd')}",
            f"- Coverage avg cost: {payload['cost'].get('coverage_avg_usd')}",
            f"- Additional avg cost: {payload['cost'].get('additional_avg_usd')}",
            "",
        ]
    )
    return "\n".join(lines)
