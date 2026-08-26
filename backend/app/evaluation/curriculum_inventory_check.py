"""Offline curriculum inventory coverage gate (no LLM)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.curriculum.inventory import (
    DEFAULT_CURRICULUM_DIR,
    DomainInventory,
    inventory_matches_title,
    load_case_domain_map,
    load_domain_inventory,
    load_experiment_config,
    load_inventory,
    validate_inventory_dict,
)
from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.metrics import normalize_topic
from app.evaluation.schemas import EvalExample

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "curriculum"


def _safe_rate(n: float, d: float) -> float:
    return (n / d) if d else 0.0


def gold_topic_coverage(example: EvalExample, inventory: DomainInventory) -> dict[str, Any]:
    gold = example.required_topic_list()
    present = []
    missing = []
    for g in gold:
        hit = inventory_matches_title(inventory, g)
        if hit is not None:
            present.append({"gold": g, "inventory_id": hit.id, "inventory_title": hit.title})
        else:
            missing.append(g)
    return {
        "gold_topic_count": len(gold),
        "covered_count": len(present),
        "coverage": _safe_rate(len(present), len(gold)),
        "present": present,
        "missing": missing,
    }


def gold_endpoint_coverage(example: EvalExample, inventory: DomainInventory) -> dict[str, Any]:
    deps = example.required_dependency_list()
    covered_edges = 0
    missing_endpoints: list[dict[str, Any]] = []
    for frm, to in deps:
        sf = inventory_matches_title(inventory, frm) is not None
        st = inventory_matches_title(inventory, to) is not None
        if sf and st:
            covered_edges += 1
        else:
            missing_endpoints.append(
                {
                    "edge": [frm, to],
                    "source_in_inventory": sf,
                    "target_in_inventory": st,
                }
            )
    return {
        "required_edge_count": len(deps),
        "covered_edge_count": covered_edges,
        "coverage": _safe_rate(covered_edges, len(deps)),
        "missing_endpoints": missing_endpoints,
    }


def run_curriculum_inventory_check(
    *,
    dataset_path: str | Path | None = None,
    curriculum_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    domains: list[str] | None = None,
    include_gold_gate: bool = True,
) -> tuple[Path, Path]:
    """Validate inventories; optionally compute gold coverage gates (offline).

    Product-neutral health fields never require gold. Gold coverage is evaluation-only.
    """
    from app.curriculum.inventory import inventory_health_report

    root = Path(curriculum_dir) if curriculum_dir else DEFAULT_CURRICULUM_DIR
    cfg = load_experiment_config(root)
    gate = cfg.get("inventory_coverage_gate") or {}
    min_topic = float(gate.get("min_gold_topic_coverage") or 0.75)
    min_edge = float(gate.get("min_gold_endpoint_coverage") or 0.75)
    case_map = load_case_domain_map(root)

    ds_path = (
        Path(dataset_path)
        if dataset_path
        else _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    )
    examples = {ex.id: ex for ex in load_dataset(ds_path)} if include_gold_gate else {}

    selected_domains = domains or sorted((cfg.get("inventory_files") or {}).keys())
    domain_reports: list[dict[str, Any]] = []
    all_pass = True

    for domain in selected_domains:
        inv_file = (cfg.get("inventory_files") or {}).get(domain)
        validation_errors: list[str] = []
        inventory: DomainInventory | None = None
        health: dict[str, Any] | None = None
        if not inv_file:
            validation_errors.append(f"No inventory file configured for {domain}")
        else:
            path = root / inv_file
            if not path.is_file():
                validation_errors.append(f"Missing inventory file {path}")
            else:
                raw = json.loads(path.read_text(encoding="utf-8"))
                validation_errors = validate_inventory_dict(raw, path=path)
                if not validation_errors:
                    inventory = load_inventory(path)
                    health = inventory_health_report(inventory)

        cases = [cid for cid, d in case_map.items() if d == domain]
        case_rows = []
        topic_num = topic_den = 0
        edge_num = edge_den = 0
        if include_gold_gate:
            for cid in cases:
                ex = examples.get(cid)
                if not ex or inventory is None:
                    continue
                adapted = adapt_example_for_edge_mode(
                    ex, "edge_calibrated", topic_matching_mode="curated_alias"
                )
                tc = gold_topic_coverage(adapted, inventory)
                ec = gold_endpoint_coverage(adapted, inventory)
                topic_num += tc["covered_count"]
                topic_den += tc["gold_topic_count"]
                edge_num += ec["covered_edge_count"]
                edge_den += ec["required_edge_count"]
                case_rows.append(
                    {
                        "case_id": cid,
                        "goal": adapted.goal,
                        "topic_coverage": tc,
                        "endpoint_coverage": ec,
                    }
                )

        topic_cov = _safe_rate(topic_num, topic_den) if include_gold_gate else None
        edge_cov = _safe_rate(edge_num, edge_den) if include_gold_gate else None
        validation_ok = inventory is not None and not validation_errors
        if include_gold_gate:
            gate_ok = (
                validation_ok
                and topic_cov is not None
                and edge_cov is not None
                and topic_cov >= min_topic
                and edge_cov >= min_edge
            )
        else:
            gate_ok = validation_ok
        if not gate_ok:
            all_pass = False
        domain_reports.append(
            {
                "domain": domain,
                "inventory_file": inv_file,
                "inventory_size": inventory.size() if inventory else 0,
                "alias_count": inventory.alias_count() if inventory else 0,
                "prerequisite_edge_count": inventory.prerequisite_edge_count() if inventory else 0,
                "version": inventory.version if inventory else None,
                "inventory_hash": inventory.content_hash if inventory else None,
                "review_status": inventory.review_status if inventory else None,
                "provenance": inventory.provenance if inventory else None,
                "health": health,
                "cases": cases,
                "n_cases": len(cases),
                "validation_errors": validation_errors,
                "gold_topic_coverage": topic_cov,
                "gold_endpoint_coverage": edge_cov,
                "gate": {
                    "min_gold_topic_coverage": min_topic if include_gold_gate else None,
                    "min_gold_endpoint_coverage": min_edge if include_gold_gate else None,
                    "include_gold_gate": include_gold_gate,
                    "pass": gate_ok,
                    "result": (
                        "PASS"
                        if gate_ok
                        else (
                            "DOMAIN_INVENTORY_INSUFFICIENT"
                            if include_gold_gate
                            else "INVALID"
                        )
                    ),
                },
                "case_details": case_rows,
                "frozen": True,
            }
        )

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_curriculum_inventory_check.json"
    md_path = out_dir / f"{ts}_curriculum_inventory_check.md"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "no_new_llm_calls": True,
        "dataset": str(ds_path),
        "curriculum_dir": str(root),
        "all_gates_pass": all_pass,
        "domains": domain_reports,
        "case_domain_map": case_map,
        "note": "Inventories are frozen for subsequent live runs; do not expand after observing failures.",
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Curriculum Inventory Check",
        "",
        f"- NO_NEW_LLM_CALLS: `True`",
        f"- All gates pass: **{all_pass}**",
        "",
        "| Domain | Inventory Size | Gold Topic Coverage | Gold Endpoint Coverage | Gate Result |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for d in domain_reports:
        tc = d["gold_topic_coverage"]
        ec = d["gold_endpoint_coverage"]
        tc_s = f"{tc:.3f}" if isinstance(tc, (int, float)) else "n/a"
        ec_s = f"{ec:.3f}" if isinstance(ec, (int, float)) else "n/a"
        lines.append(
            f"| {d['domain']} | {d['inventory_size']} | {tc_s} | {ec_s} | {d['gate']['result']} |"
        )
    lines.append("")
    for d in domain_reports:
        lines.append(f"## {d['domain']}")
        lines.append("")
        lines.append(f"- Cases: `{d['cases']}`")
        lines.append(f"- Aliases: {d['alias_count']}; inventory prereq edges: {d['prerequisite_edge_count']}")
        if d["validation_errors"]:
            lines.append(f"- Validation errors: `{d['validation_errors']}`")
        for c in d["case_details"]:
            miss = c["topic_coverage"]["missing"]
            lines.append(
                f"- `{c['case_id']}` topic_cov={c['topic_coverage']['coverage']:.2f} "
                f"endpoint_cov={c['endpoint_coverage']['coverage']:.2f} missing={miss}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path
