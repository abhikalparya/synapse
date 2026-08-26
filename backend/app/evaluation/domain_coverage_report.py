"""Offline domain coverage status report (no inventory mutation)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.curriculum.inventory import (
    DEFAULT_CURRICULUM_DIR,
    inventory_health_report,
    load_case_domain_map,
    load_domain_inventory,
    load_experiment_config,
    load_inventory,
    validate_inventory_dict,
)
from app.curriculum.resolution import inventory_configured

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "curriculum"
DEFAULT_QUALITY = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"


def _load_case_categories(dataset_path: Path) -> dict[str, dict[str, str]]:
    """Load case id → {category, goal} for coverage status only (not gold topics)."""
    out: dict[str, dict[str, str]] = {}
    if not dataset_path.is_file():
        return out
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = str(row.get("id") or "").strip()
        if not cid:
            continue
        out[cid] = {
            "category": str(row.get("category") or ""),
            "goal": str(row.get("goal") or ""),
        }
    return out


def run_domain_coverage_report(
    *,
    dataset_path: str | Path | None = None,
    curriculum_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Report mapped/unmapped domain coverage for the evaluation case set."""
    root = Path(curriculum_dir) if curriculum_dir else DEFAULT_CURRICULUM_DIR
    cfg = load_experiment_config(root)
    case_map = load_case_domain_map(root)
    ds_path = Path(dataset_path) if dataset_path else DEFAULT_QUALITY
    cases = _load_case_categories(ds_path)

    inventory_files = cfg.get("inventory_files") or {}
    inventory_rows: list[dict[str, Any]] = []
    for domain, rel in sorted(inventory_files.items()):
        path = root / rel
        validation_errors: list[str] = []
        health: dict[str, Any] | None = None
        if not path.is_file():
            validation_errors.append(f"missing file {path}")
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            validation_errors = validate_inventory_dict(raw, path=path)
            if not validation_errors:
                inv = load_inventory(path)
                health = inventory_health_report(inv)
        inventory_rows.append(
            {
                "domain": domain,
                "inventory_file": rel,
                "available": path.is_file() and not validation_errors,
                "validation_errors": validation_errors,
                "health": health,
                "version": (health or {}).get("version"),
                "inventory_version": (health or {}).get("inventory_version"),
            }
        )

    mapped_cases: list[dict[str, Any]] = []
    unmapped_cases: list[dict[str, Any]] = []
    for cid, meta in sorted(cases.items()):
        domain = case_map.get(cid)
        if domain:
            available = inventory_configured(domain, curriculum_dir=root)
            mapped_cases.append(
                {
                    "case_id": cid,
                    "category": meta.get("category"),
                    "domain": domain,
                    "inventory_available": available,
                    "inventory_version": next(
                        (r.get("version") for r in inventory_rows if r["domain"] == domain),
                        None,
                    ),
                    "coverage_status": "mapped_inventory_ok" if available else "mapped_inventory_missing",
                }
            )
        else:
            unmapped_cases.append(
                {
                    "case_id": cid,
                    "category": meta.get("category"),
                    "domain": None,
                    "inventory_available": False,
                    "coverage_status": "unmapped",
                }
            )

    domains_from_cases: dict[str, int] = {}
    for cid, meta in cases.items():
        cat = meta.get("category") or "unknown"
        domains_from_cases[cat] = domains_from_cases.get(cat, 0) + 1

    mapped_domain_set = sorted({r["domain"] for r in mapped_cases if r.get("domain")})
    unmapped_categories = sorted(
        {
            r["category"]
            for r in unmapped_cases
            if r.get("category") and r["category"] not in mapped_domain_set
        }
    )

    n_cases = len(cases)
    n_mapped = len(mapped_cases)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "no_new_llm_calls": True,
        "dataset": str(ds_path),
        "curriculum_dir": str(root),
        "total_cases": n_cases,
        "mapped_case_count": n_mapped,
        "unmapped_case_count": len(unmapped_cases),
        "inventory_availability_rate": (n_mapped / n_cases) if n_cases else 0.0,
        "mapped_domains": mapped_domain_set,
        "unmapped_categories": unmapped_categories,
        "cases_per_category": domains_from_cases,
        "inventories": inventory_rows,
        "mapped_cases": mapped_cases,
        "unmapped_cases": unmapped_cases,
        "note": "Status report only; does not mutate inventories or use gold topic lists.",
    }

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_domain_coverage_report.json"
    md_path = out_dir / f"{ts}_domain_coverage_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Domain Coverage Report",
        "",
        f"- Timestamp: `{payload['timestamp']}`",
        f"- Total cases: **{n_cases}**",
        f"- Mapped cases: **{n_mapped}** ({payload['inventory_availability_rate']:.1%})",
        f"- Unmapped cases: **{len(unmapped_cases)}**",
        f"- Mapped domains: `{', '.join(mapped_domain_set)}`",
        f"- Unmapped categories (no case mapping): `{', '.join(unmapped_categories) or '(none)'}`",
        "",
        "## Inventories",
        "",
        "| Domain | Version | Available | Concepts | Aliases | Edges | Review |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in inventory_rows:
        h = row.get("health") or {}
        lines.append(
            f"| {row['domain']} | {row.get('version')} | {row['available']} | "
            f"{h.get('concept_count', '')} | {h.get('alias_count', '')} | "
            f"{h.get('prerequisite_edge_count', '')} | {h.get('review_status', '')} |"
        )
    lines.extend(["", "## Mapped cases", ""])
    for row in mapped_cases:
        lines.append(
            f"- `{row['case_id']}` → `{row['domain']}` "
            f"(inventory_available={row['inventory_available']})"
        )
    lines.extend(["", "## Unmapped cases", ""])
    if not unmapped_cases:
        lines.append("- (none)")
    for row in unmapped_cases:
        lines.append(f"- `{row['case_id']}` category=`{row['category']}`")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path


def run_inventory_health_only(
    *,
    curriculum_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Validate all inventories and emit product-neutral health reports (no gold)."""
    root = Path(curriculum_dir) if curriculum_dir else DEFAULT_CURRICULUM_DIR
    cfg = load_experiment_config(root)
    rows: list[dict[str, Any]] = []
    all_ok = True
    for domain, rel in sorted((cfg.get("inventory_files") or {}).items()):
        path = root / rel
        if not path.is_file():
            rows.append(
                {
                    "domain": domain,
                    "version": None,
                    "concept_count": 0,
                    "alias_count": 0,
                    "edge_count": 0,
                    "validation_status": "invalid",
                    "diagnostics": [f"missing file {path}"],
                }
            )
            all_ok = False
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        errs = validate_inventory_dict(raw, path=path)
        if errs:
            rows.append(
                {
                    "domain": domain,
                    "version": raw.get("version"),
                    "concept_count": len(raw.get("concepts") or []),
                    "alias_count": 0,
                    "edge_count": 0,
                    "validation_status": "invalid",
                    "diagnostics": errs,
                }
            )
            all_ok = False
            continue
        inv = load_domain_inventory(domain, curriculum_dir=root)
        health = inventory_health_report(inv)
        rows.append(
            {
                "domain": domain,
                "version": health["version"],
                "inventory_version": health["inventory_version"],
                "concept_count": health["concept_count"],
                "alias_count": health["alias_count"],
                "edge_count": health["prerequisite_edge_count"],
                "cycle_count": health["cycle_count"],
                "duplicate_count": health["duplicate_count"],
                "validation_status": health["validation_status"],
                "review_status": health["review_status"],
                "inventory_hash": health["inventory_hash"],
                "diagnostics": [],
            }
        )

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_curriculum_inventory_health.json"
    md_path = out_dir / f"{ts}_curriculum_inventory_health.md"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "no_new_llm_calls": True,
        "no_api_key_required": True,
        "all_valid": all_ok,
        "inventories": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Curriculum Inventory Health",
        "",
        f"- All valid: **{all_ok}**",
        "",
        "| Domain | Version | Concepts | Aliases | Edges | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['domain']} | {r.get('version')} | {r['concept_count']} | "
            f"{r['alias_count']} | {r['edge_count']} | {r['validation_status']} |"
        )
        for d in r.get("diagnostics") or []:
            lines.append(f"  - {d}")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path
