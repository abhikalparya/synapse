"""Gold-edge ambiguity review and calibration (evaluation-only; no LLM).

Before-state (dataset / metrics)
--------------------------------
1. Required edges: ``required_dependencies`` if set, else all ``gold_dependencies``.
2. Extra generated edges (not required, not reverse, not in ``acceptable_dependencies``)
   are penalized via ``extra_dependency_rate`` / EXTRA_DEPENDENCY.
3. Existing ``acceptable_dependencies`` already count toward legacy dependency
   *precision* only — never toward recall.
4. No AMBIGUOUS class existed before this module; unresolved extras were all "invalid".

Edge modes
----------
fair / current_fair
    Quality schema as loaded (dataset acceptable_dependencies only).
edge_calibrated
    Fair + approved ACCEPTABLE_ALTERNATIVE / AMBIGUOUS entries from
    ``data/eval/acceptable_dependencies_v1.json``.
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.evaluation.dataset import load_dataset
from app.evaluation.inspect import _graph_from_row
from app.evaluation.matching_modes import adapt_example_for_mode
from app.evaluation.metrics import (
    aggregate_scores,
    compare_graphs,
    find_redundant_transitive_edges,
    match_topic,
    normalize_topic,
    score_graph,
)
from app.evaluation.schemas import EvalExample

EdgeMode = Literal["fair", "edge_calibrated"]
EDGE_MODES: tuple[EdgeMode, ...] = ("fair", "edge_calibrated")

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EDGE_POLICY_PATH = _REPO_ROOT / "data" / "eval" / "acceptable_dependencies_v1.json"

CANDIDATE_CATEGORIES = (
    "REQUIRED",
    "ACCEPTABLE_ALTERNATIVE",
    "INVALID",
    "AMBIGUOUS",
    "UNKNOWN",
)


def resolve_edge_mode(raw: str | None) -> EdgeMode:
    key = (raw or "fair").strip().casefold().replace("-", "_")
    aliases = {
        "fair": "fair",
        "current_fair": "fair",
        "edge_calibrated": "edge_calibrated",
        "edge_ambiguity_calibrated": "edge_calibrated",
        "calibrated": "edge_calibrated",
    }
    if key not in aliases:
        raise ValueError(f"Unknown edge matching mode {raw!r}; choose one of {list(EDGE_MODES)}")
    return aliases[key]  # type: ignore[return-value]


def load_edge_policy(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else DEFAULT_EDGE_POLICY_PATH
    if not target.is_file():
        return {"version": "acceptable_dependencies_v1", "entries": []}
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ValueError(f"Edge policy must be an object with entries[]: {target}")
    return data


def approved_policy_edges(
    registry: dict[str, Any] | None = None,
    *,
    classification: str,
) -> dict[str, list[tuple[str, str]]]:
    """case_id -> list of (from, to) for approved entries of a given classification."""
    reg = registry if registry is not None else load_edge_policy()
    out: dict[str, list[tuple[str, str]]] = {}
    for entry in reg.get("entries") or []:
        if not entry.get("approved"):
            continue
        if str(entry.get("classification") or "") != classification:
            continue
        case_id = str(entry.get("case_id") or "").strip()
        frm = str(entry.get("from") or "").strip()
        to = str(entry.get("to") or "").strip()
        if not case_id or not frm or not to:
            continue
        out.setdefault(case_id, []).append((frm, to))
    return out


def approved_acceptable_records(registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    reg = registry if registry is not None else load_edge_policy()
    return [
        e
        for e in (reg.get("entries") or [])
        if e.get("approved") and e.get("classification") == "ACCEPTABLE_ALTERNATIVE"
    ]


def adapt_example_for_edge_mode(
    example: EvalExample,
    mode: EdgeMode | str,
    *,
    topic_matching_mode: str = "fair",
    edge_policy: dict[str, Any] | None = None,
) -> EvalExample:
    """Apply topic matching mode, then optional calibrated edge policy."""
    resolved = resolve_edge_mode(mode if isinstance(mode, str) else mode)
    base = adapt_example_for_mode(example, topic_matching_mode)
    if resolved == "fair":
        return replace(
            base,
            acceptable_dependencies=list(base.acceptable_dependencies),
            ambiguous_dependencies=[],
            dataset_version=f"{base.dataset_version}+edge_fair",
        )

    policy = edge_policy if edge_policy is not None else load_edge_policy()
    acc_map = approved_policy_edges(policy, classification="ACCEPTABLE_ALTERNATIVE")
    amb_map = approved_policy_edges(policy, classification="AMBIGUOUS")
    acc = list(base.acceptable_dependencies)
    for frm, to in acc_map.get(example.id, []):
        if (frm, to) not in acc:
            acc.append((frm, to))
    amb = list(amb_map.get(example.id, []))
    return replace(
        base,
        acceptable_dependencies=acc,
        ambiguous_dependencies=amb,
        dataset_version=f"{base.dataset_version}+edge_calibrated_v1",
    )


def _propose_candidate_category(
    *,
    both_matched: bool,
    is_self_loop: bool,
    is_redundant: bool,
    is_reverse_of_required: bool,
) -> tuple[str, str]:
    """Heuristic proposal only — never auto-approved."""
    if is_self_loop:
        return "INVALID", "Self-loop dependency is structurally invalid."
    if is_reverse_of_required:
        return "INVALID", "Edge is the reverse of a required gold dependency (direction error)."
    if is_redundant:
        return "INVALID", "Edge is a redundant transitive shortcut in the generated graph."
    if not both_matched:
        return (
            "UNKNOWN",
            "One or both endpoints are unmatched under current topic matching; "
            "cannot review as a gold structural alternative.",
        )
    return (
        "AMBIGUOUS",
        "Both endpoints match gold/optional topics, but educational necessity is not "
        "automatically decidable; needs human review.",
    )


def build_gold_edge_ambiguity_review(
    result_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
    output_dir: str | Path | None = None,
    topic_matching_mode: str = "fair",
    edge_policy_path: str | Path | None = None,
) -> Path:
    """Extract EXTRA_DEPENDENCY rows with UNREVIEWED status (no auto-accept)."""
    target = Path(result_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    examples = {e.id: e for e in load_dataset(dataset_path)}
    policy = load_edge_policy(edge_policy_path)
    approved_acc = {
        (e["case_id"], normalize_topic(e["from"]), normalize_topic(e["to"])): e
        for e in approved_acceptable_records(policy)
    }
    approved_amb = {
        (str(e["case_id"]), normalize_topic(e["from"]), normalize_topic(e["to"])): e
        for e in (policy.get("entries") or [])
        if e.get("approved") and e.get("classification") == "AMBIGUOUS"
    }

    rows = ((payload.get("systems") or {}).get(system) or {}).get("example_results") or []
    records: list[dict[str, Any]] = []
    for row in rows:
        eid = str(row.get("example_id") or "")
        base = examples.get(eid)
        if base is None:
            continue
        # Fair edge schema (dataset acceptable only) so review sees true extras.
        ex = adapt_example_for_edge_mode(base, "fair", topic_matching_mode=topic_matching_mode)
        graph = _graph_from_row(row)
        if not graph.parse_ok:
            continue
        comparison = compare_graphs(ex, graph)
        required = ex.required_dependency_list()
        reverse_req = {(normalize_topic(b), normalize_topic(a)) for a, b in required}
        red = set(find_redundant_transitive_edges(list(graph.dependencies)))
        for frm, to in comparison.get("extra_dependencies") or []:
            cf = match_topic(frm, ex)
            ct = match_topic(to, ex)
            both = cf is not None and ct is not None
            is_self = (frm == to) or (cf is not None and cf == ct)
            is_red = (frm, to) in red
            is_rev = False
            if both:
                is_rev = (normalize_topic(cf), normalize_topic(ct)) in reverse_req
            cand, reason = _propose_candidate_category(
                both_matched=both,
                is_self_loop=bool(is_self),
                is_redundant=is_red,
                is_reverse_of_required=is_rev,
            )
            key = (eid, normalize_topic(cf or frm), normalize_topic(ct or to))
            review_status = "UNREVIEWED"
            if key in approved_acc:
                cand = "ACCEPTABLE_ALTERNATIVE"
                review_status = "APPROVED_IN_POLICY"
                reason = str(approved_acc[key].get("reason") or reason)
            elif key in approved_amb:
                cand = "AMBIGUOUS"
                review_status = "APPROVED_IN_POLICY"
                reason = str(approved_amb[key].get("reason") or reason)
            elif cand == "INVALID":
                review_status = "HEURISTIC_INVALID"

            records.append(
                {
                    "case_id": eid,
                    "generated_from": frm,
                    "generated_to": to,
                    "canonical_from": cf,
                    "canonical_to": ct,
                    "current_classification": "EXTRA_DEPENDENCY",
                    "candidate_category": cand,
                    "reason": reason,
                    "review_status": review_status,
                },
            )

    by_cat = dict(Counter(r["candidate_category"] for r in records))
    by_status = dict(Counter(r["review_status"] for r in records))
    artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_benchmark": str(target),
        "system": system,
        "topic_matching_mode": topic_matching_mode,
        "edge_baseline": "fair",
        "note": (
            "Deterministic EXTRA_DEPENDENCY inventory. candidate_category is a review aid; "
            "review_status defaults to UNREVIEWED unless an explicit approved policy entry exists. "
            "No automatic promotion from frequency, similarity, aliases, or connectivity."
        ),
        "summary": {
            "extra_dependencies_reviewed": len(records),
            "by_candidate_category": dict(sorted(by_cat.items(), key=lambda kv: (-kv[1], kv[0]))),
            "by_review_status": dict(sorted(by_status.items(), key=lambda kv: (-kv[1], kv[0]))),
            "approved_acceptable_in_policy": len(approved_acceptable_records(policy)),
        },
        "records": records,
    }

    out = Path(output_dir) if output_dir else _REPO_ROOT / "results" / "failure_analysis"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    path = out / f"{stamp}_gold_edge_ambiguity_review.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        f"# Gold edge ambiguity review — {artifact['timestamp']}",
        "",
        f"- Source: `{target}`",
        f"- Extra dependencies: {len(records)}",
        f"- Approved ACCEPTABLE_ALTERNATIVE in policy: {artifact['summary']['approved_acceptable_in_policy']}",
        "",
        "## Candidate category breakdown",
        "",
    ]
    for k, v in artifact["summary"]["by_candidate_category"].items():
        md.append(f"- {k}: {v}")
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


_COMPARE_METRICS = (
    "required_edge_precision",
    "required_edge_recall",
    "required_edge_f1",
    "missing_required_edge_rate",
    "invalid_extra_edge_rate",
    "acceptable_alternative_rate",
    "ambiguous_edge_rate",
    "dependency_direction_error_rate",
    "redundant_transitive_edge_rate",
    # legacy mirrors
    "dependency_precision",
    "dependency_recall",
    "dependency_f1",
    "extra_dependency_rate",
)


def rescore_edge_ambiguity_modes(
    result_path: str | Path,
    *,
    modes: list[str] | None = None,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
    output_dir: str | Path | None = None,
    topic_matching_mode: str = "curated_alias",
    edge_policy_path: str | Path | None = None,
) -> Path:
    """Compare fair vs edge_calibrated on identical stored generations.

    Topic matching defaults to ``curated_alias`` so both edge modes share the same
    topic identity layer; only the reviewed edge policy differs.
    """
    target = Path(result_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    examples = {e.id: e for e in load_dataset(dataset_path)}
    policy = load_edge_policy(edge_policy_path)
    resolved_modes = [resolve_edge_mode(m) for m in (modes or list(EDGE_MODES))]

    systems = payload.get("systems") or {}
    if system not in systems and systems:
        system = next(iter(systems))
    rows = (systems.get(system) or {}).get("example_results") or []

    metrics_by_mode: dict[str, Any] = {}
    disagreement_breakdown: dict[str, dict[str, int]] = {}

    for mode in resolved_modes:
        scores = []
        breakdown: Counter[str] = Counter()
        for row in rows:
            eid = str(row.get("example_id") or "")
            base = examples.get(eid)
            if base is None:
                continue
            graph = _graph_from_row(row)
            ex = adapt_example_for_edge_mode(
                base,
                mode,
                topic_matching_mode=topic_matching_mode,
                edge_policy=policy,
            )
            if not graph.parse_ok:
                continue
            sc = score_graph(ex, graph)
            scores.append(sc)
            comp = compare_graphs(ex, graph)
            breakdown["MATCHED_REQUIRED_EDGE"] += len(comp["matched_dependencies"])
            breakdown["MISSING_REQUIRED_EDGE"] += len(comp["missing_dependencies"])
            breakdown["MATCHED_ACCEPTABLE_ALTERNATIVE_EDGE"] += len(comp["acceptable_dependencies_used"])
            breakdown["INVALID_EXTRA_EDGE"] += len(comp["extra_dependencies"])
            breakdown["AMBIGUOUS_EDGE"] += len(comp.get("ambiguous_dependencies_used") or [])
            breakdown["WRONG_DIRECTION"] += len(comp["reversed_dependencies"])
            breakdown["TRANSITIVE_REDUNDANT_EDGE"] += len(comp["redundant_transitive_edges"])
            for s in graph.skipped_dependencies or []:
                reason = (s.get("reason") or "").casefold()
                if "unknown" in reason:
                    breakdown["UNKNOWN_REFERENCE"] += 1
        metrics_by_mode[mode] = aggregate_scores(scores)
        disagreement_breakdown[mode] = dict(breakdown)

    fair_m = metrics_by_mode.get("fair") or {}
    table = []
    for key in _COMPARE_METRICS:
        row = {"metric": key}
        for mode in resolved_modes:
            row[mode] = float((metrics_by_mode.get(mode) or {}).get(key) or 0.0)
        if "fair" in resolved_modes and "edge_calibrated" in resolved_modes:
            row["delta"] = row["edge_calibrated"] - row["fair"]
        table.append(row)

    # Breakdown of legacy extra-dependency mass under fair
    fair_break = disagreement_breakdown.get("fair") or {}
    cal_break = disagreement_breakdown.get("edge_calibrated") or {}
    fair_extra_total = fair_break.get("INVALID_EXTRA_EDGE", 0) + fair_break.get("AMBIGUOUS_EDGE", 0)
    newly_acceptable = max(
        0,
        cal_break.get("MATCHED_ACCEPTABLE_ALTERNATIVE_EDGE", 0)
        - fair_break.get("MATCHED_ACCEPTABLE_ALTERNATIVE_EDGE", 0),
    )

    stamp = datetime.now(timezone.utc)
    approved = approved_acceptable_records(policy)
    artifact = {
        "timestamp": stamp.isoformat(),
        "benchmark_type": "edge_ambiguity_calibration",
        "rescored_from": str(target),
        "system": system,
        "topic_matching_mode": topic_matching_mode,
        "edge_modes": resolved_modes,
        "approved_acceptable_alternative_count": len(approved),
        "metrics_by_mode": metrics_by_mode,
        "comparison_table": table,
        "disagreement_breakdown": disagreement_breakdown,
        "extra_dependency_decomposition": {
            "fair_invalid_extra_edges": fair_break.get("INVALID_EXTRA_EDGE", 0),
            "fair_ambiguous_edges": fair_break.get("AMBIGUOUS_EDGE", 0),
            "fair_extra_like_total": fair_extra_total,
            "calibrated_invalid_extra_edges": cal_break.get("INVALID_EXTRA_EDGE", 0),
            "calibrated_ambiguous_edges": cal_break.get("AMBIGUOUS_EDGE", 0),
            "calibrated_acceptable_used": cal_break.get("MATCHED_ACCEPTABLE_ALTERNATIVE_EDGE", 0),
            "newly_accepted_alternative_matches": newly_acceptable,
            "note": (
                "Counts are absolute edge occurrences across examples (not rates). "
                "Legacy extra_dependency_rate ≈ invalid(+ambiguous)/generated under each mode."
            ),
        },
        "notes": [
            "Identical stored generations; only reviewed gold-edge interpretation changes.",
            "No new LLM calls. Metric shifts are measurement calibration, not generation improvement.",
            "ACCEPTABLE_ALTERNATIVE never inflates required-edge recall.",
        ],
    }

    out = Path(output_dir) if output_dir else target.parent
    out.mkdir(parents=True, exist_ok=True)
    model = str(payload.get("model") or "unknown")
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in model)
    path = out / f"{stamp.strftime('%Y-%m-%d_%H%M%S')}_edge_ambiguity_calibration_{safe}.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        f"# Gold graph ambiguity calibration — {artifact['timestamp']}",
        "",
        f"- Source: `{target}`",
        f"- Topic matching: `{topic_matching_mode}`",
        f"- Approved ACCEPTABLE_ALTERNATIVE rules: {len(approved)}",
        "",
        "| Metric | Current Fair | Edge-Calibrated | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in table:
        def fmt(x):
            return f"{float(x):.3f}" if isinstance(x, (int, float)) else "n/a"

        md.append(
            f"| {row['metric']} | {fmt(row.get('fair'))} | {fmt(row.get('edge_calibrated'))} | {fmt(row.get('delta'))} |",
        )
    md.extend(["", "## Disagreement breakdown", ""])
    for mode, br in disagreement_breakdown.items():
        md.append(f"### {mode}")
        for k, v in sorted(br.items()):
            md.append(f"- {k}: {v}")
        md.append("")
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Audit every approved alternative that actually matched in calibrated mode
    fail_dir = _REPO_ROOT / "results" / "failure_analysis"
    fail_dir.mkdir(parents=True, exist_ok=True)
    audit_path = fail_dir / f"{stamp.strftime('%Y-%m-%d_%H%M%S')}_accepted_alternative_edges.md"
    used_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        eid = str(row.get("example_id") or "")
        base = examples.get(eid)
        if base is None:
            continue
        graph = _graph_from_row(row)
        ex = adapt_example_for_edge_mode(
            base, "edge_calibrated", topic_matching_mode=topic_matching_mode, edge_policy=policy,
        )
        if not graph.parse_ok:
            continue
        comp = compare_graphs(ex, graph)
        for frm, to in comp.get("acceptable_dependencies_used") or []:
            cf = match_topic(frm, ex) or frm
            ct = match_topic(to, ex) or to
            used_keys.add((eid, normalize_topic(cf), normalize_topic(ct)))

    lines = [
        f"# Accepted alternative edges — {artifact['timestamp']}",
        "",
        "Every approved ACCEPTABLE_ALTERNATIVE policy entry, with match status on this artifact.",
        "",
    ]
    for entry in approved:
        key = (entry["case_id"], normalize_topic(entry["from"]), normalize_topic(entry["to"]))
        matched = key in used_keys
        lines.append(f"## Case: `{entry['case_id']}`")
        lines.append("")
        lines.append(f"Generated / policy edge: **{entry['from']}** requires **{entry['to']}**")
        lines.append("")
        lines.append("Classification: ACCEPTABLE_ALTERNATIVE")
        lines.append("")
        lines.append(f"Matched in calibrated rescore: {'yes' if matched else 'no (policy present; not emitted this run)'}")
        lines.append("")
        lines.append(f"Why educationally valid: {entry.get('reason')}")
        lines.append("")
        lines.append(f"Why not REQUIRED: {entry.get('why_not_required')}")
        lines.append("")
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifact["accepted_alternative_audit"] = str(audit_path)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
