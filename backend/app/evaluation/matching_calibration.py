"""Rescore stored generations under strict / fair / curated_alias matching (no LLM)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset import load_dataset
from app.evaluation.inspect import _graph_from_row
from app.evaluation.matching_modes import (
    MATCHING_MODES,
    MATCHING_VERSIONS,
    adapt_example_for_mode,
    approved_alias_records,
    load_curated_aliases,
    resolve_matching_mode,
)
from app.evaluation.metrics import (
    aggregate_scores,
    compare_graphs,
    match_topic,
    normalize_topic,
    score_graph,
)
from app.evaluation.reporting import write_benchmark_result
from app.evaluation.topic_equivalence import classify_unmatched_topic, extract_unmatched_topics

_METRIC_KEYS = (
    "topic_precision",
    "topic_recall",
    "topic_f1",
    "dependency_precision",
    "dependency_recall",
    "dependency_f1",
    "missing_prerequisite_rate",
    "extra_dependency_rate",
    "hallucinated_topic_rate",
)


def _score_block(scores_list: list) -> dict[str, Any]:
    agg = aggregate_scores(scores_list)
    return agg


def _count_unmatched(example, graph) -> int:
    if not graph.parse_ok:
        return 0
    return len(compare_graphs(example, graph).get("extra_topics") or [])


def rescore_matching_modes(
    result_path: str | Path,
    *,
    modes: list[str] | None = None,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
    output_dir: str | Path | None = None,
    curated_path: str | Path | None = None,
) -> Path:
    """Compare matching modes on identical stored generations. No new LLM calls."""
    target = Path(result_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    examples = {e.id: e for e in load_dataset(dataset_path)}
    registry = load_curated_aliases(curated_path)
    resolved_modes = [resolve_matching_mode(m) for m in (modes or list(MATCHING_MODES))]

    # Prefer requested system; fall back to first available
    systems = payload.get("systems") or {}
    if system not in systems and systems:
        system = next(iter(systems))
    rows = (systems.get(system) or {}).get("example_results") or []

    mode_metrics: dict[str, Any] = {}
    mode_details: dict[str, Any] = {}
    fair_unmatched_by_case: dict[str, list[str]] = {}

    for mode in resolved_modes:
        scores = []
        unmatched_total = 0
        per_case = []
        for row in rows:
            eid = str(row.get("example_id") or "")
            base = examples.get(eid)
            if base is None:
                continue
            graph = _graph_from_row(row)
            ex = adapt_example_for_mode(base, mode, curated_registry=registry)
            sc = score_graph(ex, graph) if graph.parse_ok else None
            if sc is not None:
                scores.append(sc)
            n_unmatched = _count_unmatched(ex, graph)
            unmatched_total += n_unmatched
            if mode == "fair" and graph.parse_ok:
                fair_unmatched_by_case[eid] = extract_unmatched_topics(base, graph)
            per_case.append(
                {
                    "example_id": eid,
                    "unmatched_topics": n_unmatched,
                    "topic_f1": sc.topic_f1 if sc else None,
                    "dependency_f1": sc.dependency_f1 if sc else None,
                    "hallucinated_topic_rate": sc.hallucinated_topic_rate if sc else None,
                },
            )
        agg = _score_block(scores)
        mode_metrics[mode] = {
            **agg,
            "unmatched_topic_count": unmatched_total,
            "matching_version": MATCHING_VERSIONS[mode],
        }
        mode_details[mode] = per_case

    # Newly accepted due to curated aliases (fair → curated_alias)
    newly_accepted: list[dict[str, Any]] = []
    if "fair" in resolved_modes and "curated_alias" in resolved_modes:
        for row in rows:
            eid = str(row.get("example_id") or "")
            base = examples.get(eid)
            if base is None:
                continue
            graph = _graph_from_row(row)
            if not graph.parse_ok:
                continue
            fair_ex = adapt_example_for_mode(base, "fair", curated_registry=registry)
            cur_ex = adapt_example_for_mode(base, "curated_alias", curated_registry=registry)
            for title in graph.topics:
                fair_hit = match_topic(title, fair_ex)
                cur_hit = match_topic(title, cur_ex)
                if fair_hit is None and cur_hit is not None:
                    rule = None
                    for entry in approved_alias_records(registry):
                        if normalize_topic(entry.get("canonical") or "") != normalize_topic(cur_hit):
                            continue
                        for a in entry.get("aliases") or []:
                            if normalize_topic(a) == normalize_topic(title):
                                rule = entry
                                break
                        if rule:
                            break
                    newly_accepted.append(
                        {
                            "case_id": eid,
                            "generated_topic": title,
                            "canonical_gold_topic": cur_hit,
                            "alias_rule": {
                                "canonical": (rule or {}).get("canonical"),
                                "aliases": (rule or {}).get("aliases"),
                                "classification": (rule or {}).get("classification"),
                                "reason": (rule or {}).get("reason"),
                            },
                        },
                    )

    # Classification tallies on fair-unmatched topics
    class_counts: dict[str, int] = {}
    true_hallucinations = 0
    unknown_unmatched = 0
    rejected_candidates = 0
    for eid, titles in fair_unmatched_by_case.items():
        base = examples.get(eid)
        if base is None:
            continue
        fair_ex = adapt_example_for_mode(base, "fair", curated_registry=registry)
        for title in titles:
            # Skip if curated alias now accepts it
            cur_ex = adapt_example_for_mode(base, "curated_alias", curated_registry=registry)
            if match_topic(title, cur_ex) is not None:
                continue
            c = classify_unmatched_topic(title, fair_ex)
            cat = c["proposed_classification"]
            class_counts[cat] = class_counts.get(cat, 0) + 1
            if cat == "GENUINE_HALLUCINATION":
                true_hallucinations += 1
            if cat == "UNKNOWN":
                unknown_unmatched += 1
            if c.get("candidate_alias"):
                rejected_candidates += 1

    comparison_table = []
    fair_m = mode_metrics.get("fair") or {}
    for key in _METRIC_KEYS:
        row = {"metric": key}
        for mode in resolved_modes:
            row[mode] = float((mode_metrics.get(mode) or {}).get(key) or 0.0)
        if "fair" in resolved_modes and "curated_alias" in resolved_modes:
            row["fair_to_alias_delta"] = row["curated_alias"] - row["fair"]
        comparison_table.append(row)
    # unmatched counts
    um = {"metric": "unmatched_topic_count"}
    for mode in resolved_modes:
        um[mode] = int((mode_metrics.get(mode) or {}).get("unmatched_topic_count") or 0)
    if "fair" in resolved_modes and "curated_alias" in resolved_modes:
        um["fair_to_alias_delta"] = um["curated_alias"] - um["fair"]
    comparison_table.append(um)
    comparison_table.append(
        {
            "metric": "topics_newly_accepted_due_to_aliases",
            "strict": None,
            "fair": 0,
            "curated_alias": len(newly_accepted),
            "fair_to_alias_delta": len(newly_accepted),
        },
    )

    stamp = datetime.now(timezone.utc)
    artifact = {
        "timestamp": stamp.isoformat(),
        "benchmark_type": "matching_calibration",
        "rescored_from": str(target),
        "system": system,
        "dataset": Path(dataset_path).stem if dataset_path else payload.get("dataset"),
        "dataset_version": Path(dataset_path).stem if dataset_path else payload.get("dataset_version"),
        "model": payload.get("model"),
        "prompt_variant": payload.get("prompt_variant"),
        "matching_modes": resolved_modes,
        "matching_versions": {m: MATCHING_VERSIONS[m] for m in resolved_modes},
        "curated_alias_count": len(approved_alias_records(registry)),
        "metrics_by_mode": mode_metrics,
        "comparison_table": comparison_table,
        "accepted_alias_matches": newly_accepted,
        "accepted_alias_match_count": len(newly_accepted),
        "rejected_candidate_aliases": rejected_candidates,
        "unknown_unmatched_topics": unknown_unmatched,
        "true_hallucinations": true_hallucinations,
        "fair_unmatched_classification_remaining": dict(sorted(class_counts.items())),
        "notes": [
            "Identical stored generations rescored under different matching modes only.",
            "No new LLM calls. Metric gains are measurement calibration, not model improvement.",
            "Curated aliases are manually approved; decomposition/abstraction/related concepts are not aliases.",
        ],
    }

    out = Path(output_dir) if output_dir else target.parent
    out.mkdir(parents=True, exist_ok=True)
    model = str(payload.get("model") or "unknown")
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in model)
    path = out / f"{stamp.strftime('%Y-%m-%d_%H%M%S')}_matching_calibration_{safe}.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Markdown summary
    md = [
        f"# Matching calibration — {artifact['timestamp']}",
        "",
        f"- Source generations: `{target}`",
        f"- System: `{system}`",
        f"- Curated aliases approved: {artifact['curated_alias_count']}",
        f"- Newly accepted alias matches: {len(newly_accepted)}",
        "",
        "| Metric | Strict | Fair | Curated Alias | Fair → Alias Δ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_table:
        def fmt(x):
            if x is None:
                return "n/a"
            if isinstance(x, float):
                return f"{x:.3f}"
            return str(x)

        md.append(
            "| {m} | {s} | {f} | {c} | {d} |".format(
                m=row["metric"],
                s=fmt(row.get("strict")),
                f=fmt(row.get("fair")),
                c=fmt(row.get("curated_alias")),
                d=fmt(row.get("fair_to_alias_delta")),
            ),
        )
    md.extend(
        [
            "",
            "## Calibration counters",
            "",
            f"- accepted_alias_matches: {len(newly_accepted)}",
            f"- rejected_candidate_aliases (still unmatched): {rejected_candidates}",
            f"- unknown_unmatched_topics: {unknown_unmatched}",
            f"- true_hallucinations (remaining fair-unmatched): {true_hallucinations}",
            "",
            "## Interpretation",
            "",
            "Curated alias matching calibrates semantic equivalence measurement. "
            "It does not improve underlying generation quality.",
            "",
        ],
    )
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Audit every newly accepted match
    fail_dir = Path(__file__).resolve().parents[3] / "results" / "failure_analysis"
    fail_dir.mkdir(parents=True, exist_ok=True)
    audit_path = fail_dir / f"{stamp.strftime('%Y-%m-%d_%H%M%S')}_accepted_alias_matches.md"
    audit_lines = [
        f"# Accepted alias matches — {artifact['timestamp']}",
        "",
        "Every newly accepted match under curated_alias vs fair. Inspectable; nothing silent.",
        "",
    ]
    if not newly_accepted:
        audit_lines.append("_No new alias matches on this artifact._")
    else:
        by_canon: dict[str, list[dict[str, Any]]] = {}
        for item in newly_accepted:
            by_canon.setdefault(item["canonical_gold_topic"], []).append(item)
        for canon, items in sorted(by_canon.items()):
            gens = sorted({i["generated_topic"] for i in items})
            cases = sorted({i["case_id"] for i in items})
            rule = items[0].get("alias_rule") or {}
            audit_lines.append(f"## {gens[0] if len(gens)==1 else ', '.join(gens)}")
            audit_lines.append(f"→ **{canon}**")
            audit_lines.append("")
            audit_lines.append(f"- Case IDs: {', '.join(cases)}")
            audit_lines.append(f"- Alias rule aliases: {rule.get('aliases')}")
            audit_lines.append(f"- Classification: {rule.get('classification')}")
            audit_lines.append(f"- Reason: {rule.get('reason')}")
            audit_lines.append("")
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    artifact["accepted_alias_audit"] = str(audit_path)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Also write a compact quality-shaped artifact for the curated mode (optional convenience)
    return path


def rescore_benchmark_with_mode(
    result_path: str | Path,
    *,
    matching_mode: str = "fair",
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    curated_path: str | Path | None = None,
) -> Path:
    """Single-mode rescore of stored generations (backward-compatible extension of inspect.rescore)."""
    from app.evaluation.reporting import write_benchmark_result

    target = Path(result_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    examples = {e.id: e for e in load_dataset(dataset_path)}
    registry = load_curated_aliases(curated_path)
    mode = resolve_matching_mode(matching_mode)
    systems_out: dict[str, Any] = {}
    for sys_name, block in (payload.get("systems") or {}).items():
        scores = []
        new_rows = []
        for row in block.get("example_results") or []:
            base = examples.get(str(row.get("example_id")))
            if base is None:
                new_rows.append(row)
                continue
            graph = _graph_from_row(row)
            ex = adapt_example_for_mode(base, mode, curated_registry=registry)
            sc = score_graph(ex, graph) if graph.parse_ok else None
            if sc is not None:
                scores.append(sc)
                row = {
                    **row,
                    "scores": {
                        "topic_precision": sc.topic_precision,
                        "topic_recall": sc.topic_recall,
                        "topic_f1": sc.topic_f1,
                        "dependency_precision": sc.dependency_precision,
                        "dependency_recall": sc.dependency_recall,
                        "dependency_f1": sc.dependency_f1,
                        "graph_valid": sc.graph_valid,
                        "cycle_attempt": sc.cycle_attempt,
                        "missing_prerequisite_rate": sc.missing_prerequisite_rate,
                        "hallucinated_topic_rate": sc.hallucinated_topic_rate,
                        "extra_dependency_rate": sc.extra_dependency_rate,
                        "dependency_direction_error_rate": sc.dependency_direction_error_rate,
                    },
                    "failures": sc.failures,
                }
            new_rows.append(row)
        systems_out[sys_name] = {
            **block,
            "metrics": aggregate_scores(scores),
            "example_results": new_rows,
        }

    dataset_version = Path(dataset_path).stem if dataset_path else payload.get("dataset")
    rescored = {
        **payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "quality",
        "dataset_version": dataset_version,
        "matching_mode": mode,
        "matching_version": MATCHING_VERSIONS[mode],
        "rescored_from": str(target),
        "systems": systems_out,
        "metrics": {name: b.get("metrics") for name, b in systems_out.items()},
        "notes": list(payload.get("notes") or [])
        + [f"Rescored with matching_mode={mode}; no new LLM calls."],
        "model": f"rescored-{mode}-{payload.get('model') or 'unknown'}",
    }
    out = Path(output_dir) if output_dir else target.parent
    return write_benchmark_result(rescored, out)
