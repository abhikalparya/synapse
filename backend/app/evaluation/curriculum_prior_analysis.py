"""Offline analysis comparing baseline vs domain_curriculum_prior on shared cases."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.curriculum.inventory import (
    inventory_matches_title,
    load_case_domain_map,
    load_domain_inventory,
)
from app.evaluation.curriculum_inventory_check import gold_topic_coverage
from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.inspect import _graph_from_row
from app.evaluation.metrics import aggregate_scores, compare_graphs, normalize_topic, score_graph

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "failure_analysis"


def _safe_rate(n: float, d: float) -> float:
    return (n / d) if d else 0.0


def _row_by_id(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in block.get("example_results") or []:
        eid = str(row.get("example_id") or "")
        if eid and eid not in out:
            out[eid] = row
    return out


def classify_selection(
    *,
    gold_topics: list[str],
    selected_titles: list[str],
    inventory,
) -> list[dict[str, Any]]:
    gold_n = {normalize_topic(g): g for g in gold_topics}
    sel_n = {normalize_topic(t): t for t in selected_titles}
    rows = []
    for gn, g in gold_n.items():
        in_inv = inventory_matches_title(inventory, g) is not None
        selected = gn in sel_n
        if not in_inv:
            label = "INVENTORY_MISSING"
        elif selected:
            label = "SELECTED_CORRECT"
        else:
            label = "SHOULD_HAVE_SELECTED"
        rows.append({"gold_topic": g, "label": label, "selected": selected, "in_inventory": in_inv})
    for sn, title in sel_n.items():
        if sn not in gold_n:
            rows.append(
                {
                    "gold_topic": None,
                    "selected_title": title,
                    "label": "SELECTED_INCORRECT",
                    "selected": True,
                    "in_inventory": inventory_matches_title(inventory, title) is not None,
                }
            )
    return rows


def run_curriculum_prior_analysis(
    artifact_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    baseline_system: str = "synapse",
    prior_system: str = "domain_curriculum_prior",
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    target = Path(artifact_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    systems = payload.get("systems") or {}
    if baseline_system not in systems or prior_system not in systems:
        raise ValueError(
            f"Artifact must contain {baseline_system!r} and {prior_system!r}; found {list(systems)}"
        )

    ds_path = (
        Path(dataset_path)
        if dataset_path
        else _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    )
    examples = {ex.id: ex for ex in load_dataset(ds_path)}
    case_map = load_case_domain_map()
    base_rows = _row_by_id(systems[baseline_system])
    prior_rows = _row_by_id(systems[prior_system])
    shared = sorted(set(base_rows) & set(prior_rows) & set(case_map))

    base_scores = []
    prior_scores = []
    cases = []
    sel_labels: Counter[str] = Counter()
    edge_labels: Counter[str] = Counter()

    for eid in shared:
        ex = examples.get(eid)
        if not ex:
            continue
        domain = case_map[eid]
        inventory = load_domain_inventory(domain)
        adapted = adapt_example_for_edge_mode(
            ex, "edge_calibrated", topic_matching_mode="curated_alias"
        )
        gb = _graph_from_row(base_rows[eid])
        gp = _graph_from_row(prior_rows[eid])
        if gb.parse_ok:
            base_scores.append(score_graph(adapted, gb))
        if gp.parse_ok:
            prior_scores.append(score_graph(adapted, gp))

        meta = prior_rows[eid].get("generation_meta") or {}
        selected = list(meta.get("selected_titles") or gp.topics)
        sel_rows = classify_selection(
            gold_topics=adapted.required_topic_list(),
            selected_titles=selected,
            inventory=inventory,
        )
        for r in sel_rows:
            sel_labels[r["label"]] += 1

        cmp = compare_graphs(adapted, gp) if gp.parse_ok else {"missing_dependencies": []}
        for frm, to in (tuple(e) for e in cmp.get("missing_dependencies") or []):
            sf = inventory_matches_title(inventory, frm) is not None
            st = inventory_matches_title(inventory, to) is not None
            sn = {normalize_topic(t) for t in selected}
            if not sf or not st:
                lab = "INVENTORY_MISSING_ENDPOINT"
            elif normalize_topic(frm) not in sn or normalize_topic(to) not in sn:
                lab = "INVENTORY_ENDPOINT_NOT_SELECTED"
            else:
                lab = "INVENTORY_SELECTED_BUT_EDGE_MISSING"
            edge_labels[lab] += 1

        cov = gold_topic_coverage(adapted, inventory)
        cases.append(
            {
                "case_id": eid,
                "domain": domain,
                "goal": adapted.goal,
                "baseline_scores": None
                if not gb.parse_ok
                else {
                    "topic_f1": score_graph(adapted, gb).topic_f1,
                    "required_edge_f1": score_graph(adapted, gb).required_edge_f1,
                    "required_edge_recall": score_graph(adapted, gb).required_edge_recall,
                },
                "prior_scores": None
                if not gp.parse_ok
                else {
                    "topic_f1": score_graph(adapted, gp).topic_f1,
                    "required_edge_f1": score_graph(adapted, gp).required_edge_f1,
                    "required_edge_recall": score_graph(adapted, gp).required_edge_recall,
                    "hallucinated_topic_rate": score_graph(adapted, gp).hallucinated_topic_rate,
                },
                "selected_titles": selected,
                "selection_labels": sel_rows,
                "inventory_coverage": cov,
                "meta": {
                    k: meta.get(k)
                    for k in (
                        "selected_concept_count",
                        "unknown_selection_count",
                        "new_concept_count",
                        "inventory_size",
                        "cost_usd",
                        "llm_latency_ms",
                    )
                },
            }
        )

    base_agg = aggregate_scores(base_scores)
    prior_agg = aggregate_scores(prior_scores)
    delta = {
        k: prior_agg.get(k, 0) - base_agg.get(k, 0)
        for k in (
            "topic_precision",
            "topic_recall",
            "topic_f1",
            "required_edge_precision",
            "required_edge_recall",
            "required_edge_f1",
            "missing_required_edge_rate",
            "invalid_extra_edge_rate",
            "hallucinated_topic_rate",
            "dependency_direction_error_rate",
        )
    }

    # Decision heuristic
    de = delta.get("required_edge_f1", 0)
    dr = delta.get("required_edge_recall", 0)
    dh = delta.get("hallucinated_topic_rate", 0)
    if de >= 0.05 and dr >= 0.05 and dh <= 0.05:
        decision = "SUPPORTED"
    elif de > 0 or dr > 0:
        decision = "PARTIALLY_SUPPORTED"
    else:
        decision = "NOT_SUPPORTED"

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_domain_curriculum_prior_analysis.json"
    md_path = out_dir / f"{ts}_domain_curriculum_prior_analysis.md"
    payload_out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(target),
        "n_cases": len(cases),
        "baseline_system": baseline_system,
        "prior_system": prior_system,
        "aggregate": {"baseline": base_agg, "domain_curriculum_prior": prior_agg, "delta": delta},
        "selection_label_counts": dict(sel_labels),
        "missing_edge_label_counts": dict(edge_labels),
        "decision": {"code": decision, "rationale": f"Δreq_edge_f1={de:+.3f}, Δreq_edge_recall={dr:+.3f}, Δhalluc={dh:+.3f}"},
        "cases": cases,
    }
    json_path.write_text(json.dumps(payload_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Domain Curriculum Prior Analysis",
        "",
        f"- Artifact: `{target}`",
        f"- Cases: {len(cases)}",
        f"- Decision: **{decision}**",
        f"- Rationale: {payload_out['decision']['rationale']}",
        "",
        "| Metric | Baseline | Domain Prior | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, label in [
        ("topic_precision", "Topic Precision"),
        ("topic_recall", "Topic Recall"),
        ("topic_f1", "Topic F1"),
        ("required_edge_precision", "Required Edge Precision"),
        ("required_edge_recall", "Required Edge Recall"),
        ("required_edge_f1", "Required Edge F1"),
        ("missing_required_edge_rate", "Missing Required Edge Rate"),
        ("invalid_extra_edge_rate", "Invalid Extra Edge Rate"),
        ("hallucinated_topic_rate", "Hallucinated Topic Rate"),
        ("dependency_direction_error_rate", "Direction Error Rate"),
    ]:
        lines.append(
            f"| {label} | {base_agg.get(k, 0):.3f} | {prior_agg.get(k, 0):.3f} | {delta.get(k, 0):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Selection labels",
            "",
            f"`{dict(sel_labels)}`",
            "",
            "## Missing-edge labels",
            "",
            f"`{dict(edge_labels)}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path
