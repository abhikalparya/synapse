"""Deterministic A/B comparison of two quality-benchmark artifacts (same dataset/model)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset import load_dataset
from app.evaluation.inspect import _graph_from_row, classify_comparison
from app.evaluation.metrics import compare_graphs, score_graph

_METRIC_KEYS = (
    "topic_precision",
    "topic_recall",
    "topic_f1",
    "dependency_precision",
    "dependency_recall",
    "dependency_f1",
    "missing_prerequisite_rate",
    "extra_dependency_rate",
    "redundant_transitive_edge_rate",
    "dependency_direction_error_rate",
    "hallucinated_topic_rate",
)


def _system_metrics(payload: dict[str, Any], system: str = "synapse") -> dict[str, float]:
    return dict(((payload.get("systems") or {}).get(system) or {}).get("metrics") or {})


def _system_cost_latency(payload: dict[str, Any], system: str = "synapse") -> dict[str, Any]:
    block = ((payload.get("systems") or {}).get(system) or {})
    return {
        "p50_ms": (block.get("latency") or {}).get("p50_ms"),
        "avg_cost_usd": (block.get("cost") or {}).get("average_cost_usd"),
    }


def build_variant_comparison_table(
    baseline: dict[str, Any],
    concept: dict[str, Any],
    *,
    system: str = "synapse",
) -> list[dict[str, Any]]:
    b = _system_metrics(baseline, system)
    c = _system_metrics(concept, system)
    rows = []
    for key in _METRIC_KEYS:
        bv = float(b.get(key) or 0.0)
        cv = float(c.get(key) or 0.0)
        rows.append({"metric": key, "baseline": bv, "concept_direct_prerequisite": cv, "delta": cv - bv})
    b_lat = _system_cost_latency(baseline, system)
    c_lat = _system_cost_latency(concept, system)
    for key in ("p50_ms", "avg_cost_usd"):
        bv = b_lat.get(key)
        cv = c_lat.get(key)
        delta = None
        if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
            delta = float(cv) - float(bv)
        rows.append({"metric": key, "baseline": bv, "concept_direct_prerequisite": cv, "delta": delta})
    return rows


def compare_prompt_variant_runs(
    baseline_path: str | Path,
    concept_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
    max_cases: int = 15,
    output_dir: str | Path | None = None,
) -> Path:
    """Write a human-readable pairwise failure comparison (no LLM judge)."""
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    concept = json.loads(Path(concept_path).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[3]
    quality = root / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    legacy = root / "data" / "eval" / "learning_graph_eval_v1.jsonl"
    ds = Path(dataset_path) if dataset_path else (quality if quality.is_file() else legacy)
    examples = {e.id: e for e in load_dataset(ds)}

    b_rows = {r["example_id"]: r for r in ((baseline.get("systems") or {}).get(system) or {}).get("example_results") or []}
    c_rows = {r["example_id"]: r for r in ((concept.get("systems") or {}).get(system) or {}).get("example_results") or []}
    shared_ids = [i for i in b_rows if i in c_rows and i in examples]

    # Prefer cases where topic titles or edges differ, then fill to max_cases
    differing: list[str] = []
    identical: list[str] = []
    for eid in shared_ids:
        b, c = b_rows[eid], c_rows[eid]
        if b.get("generated_topics") != c.get("generated_topics") or b.get("generated_dependencies") != c.get(
            "generated_dependencies",
        ):
            differing.append(eid)
        else:
            identical.append(eid)
    selected = (differing + identical)[:max_cases]

    cases_out: list[dict[str, Any]] = []
    for eid in selected:
        example = examples[eid]
        b_row, c_row = b_rows[eid], c_rows[eid]
        b_graph = _graph_from_row(b_row)
        c_graph = _graph_from_row(c_row)
        b_score = score_graph(example, b_graph) if b_graph.parse_ok else None
        c_score = score_graph(example, c_graph) if c_graph.parse_ok else None
        b_comp = compare_graphs(example, b_graph) if b_graph.parse_ok else {}
        c_comp = compare_graphs(example, c_graph) if c_graph.parse_ok else {}
        b_fail = classify_comparison(example, b_comp) if b_comp else []
        c_fail = classify_comparison(example, c_comp) if c_comp else []

        def _score_dict(s):
            if s is None:
                return None
            return {
                "topic_f1": s.topic_f1,
                "dependency_f1": s.dependency_f1,
                "missing_prerequisite_rate": s.missing_prerequisite_rate,
                "extra_dependency_rate": s.extra_dependency_rate,
                "redundant_transitive_edge_rate": s.redundant_transitive_edge_rate,
                "dependency_direction_error_rate": s.dependency_direction_error_rate,
                "hallucinated_topic_rate": s.hallucinated_topic_rate,
            }

        bs, cs = _score_dict(b_score), _score_dict(c_score)
        deltas = {}
        if bs and cs:
            for k in bs:
                deltas[k] = cs[k] - bs[k]

        explanation_parts = []
        if set(b_row.get("generated_topics") or []) != set(c_row.get("generated_topics") or []):
            explanation_parts.append("Topic titles differ between variants.")
        if (b_row.get("generated_dependencies") or []) != (c_row.get("generated_dependencies") or []):
            explanation_parts.append("Dependency edges differ between variants.")
        if bs and cs:
            if cs["extra_dependency_rate"] < bs["extra_dependency_rate"] - 1e-9:
                explanation_parts.append("Concept prompt reduced extra-dependency rate.")
            if cs["redundant_transitive_edge_rate"] < bs["redundant_transitive_edge_rate"] - 1e-9:
                explanation_parts.append("Concept prompt reduced redundant transitive edges.")
            if cs["dependency_f1"] > bs["dependency_f1"] + 1e-9:
                explanation_parts.append("Concept prompt improved dependency F1.")
            elif cs["dependency_f1"] < bs["dependency_f1"] - 1e-9:
                explanation_parts.append("Concept prompt worsened dependency F1.")
        if not explanation_parts:
            explanation_parts.append("Graphs are identical or metric deltas are negligible.")

        cases_out.append(
            {
                "case_id": eid,
                "learning_goal": example.goal,
                "baseline_topics": b_row.get("generated_topics"),
                "concept_topics": c_row.get("generated_topics"),
                "baseline_dependencies": b_row.get("generated_dependencies"),
                "concept_dependencies": c_row.get("generated_dependencies"),
                "baseline_scores": bs,
                "concept_scores": cs,
                "metric_deltas": deltas,
                "baseline_failure_categories": sorted({f["category"] for f in b_fail}),
                "concept_failure_categories": sorted({f["category"] for f in c_fail}),
                "explanation": " ".join(explanation_parts),
            },
        )

    table = build_variant_comparison_table(baseline, concept, system=system)
    artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": system,
        "baseline_artifact": str(baseline_path),
        "concept_artifact": str(concept_path),
        "baseline_prompt": {
            "prompt_variant": baseline.get("prompt_variant"),
            "prompt_version": baseline.get("prompt_version"),
            "prompt_hash": baseline.get("prompt_hash"),
        },
        "concept_prompt": {
            "prompt_variant": concept.get("prompt_variant"),
            "prompt_version": concept.get("prompt_version"),
            "prompt_hash": concept.get("prompt_hash"),
        },
        "model": baseline.get("model") or concept.get("model"),
        "dataset": baseline.get("dataset_version") or baseline.get("dataset"),
        "comparison_table": table,
        "representative_cases": cases_out,
        "note": (
            "Deterministic pairwise comparison of stored generations. No LLM judge. "
            "n=1 repetition is not statistically significant."
        ),
    }

    out = Path(output_dir) if output_dir else Path(__file__).resolve().parents[3] / "results" / "failure_analysis"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    path = out / f"{stamp}_prompt_ab_comparison.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        f"# Prompt A/B comparison — {artifact['timestamp']}",
        "",
        f"- Model: `{artifact['model']}`",
        f"- System: `{system}`",
        f"- Baseline: `{baseline_path}`",
        f"- Concept: `{concept_path}`",
        "",
        "| Metric | Baseline | Concept + Direct Prerequisite | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in table:
        b, c, d = row["baseline"], row["concept_direct_prerequisite"], row["delta"]
        def fmt(x, metric=row["metric"]):
            if x is None:
                return "n/a"
            if isinstance(x, float):
                if "cost" in metric:
                    return f"{x:.6f}"
                if abs(x) >= 100:
                    return f"{x:.1f}"
                return f"{x:.3f}"
            return str(x)
        md.append(f"| {row['metric']} | {fmt(b)} | {fmt(c)} | {fmt(d)} |")
    md.extend(["", "## Representative cases", ""])
    for case in cases_out:
        md.append(f"### {case['case_id']}")
        md.append(f"- Goal: {case['learning_goal']}")
        md.append(f"- Explanation: {case['explanation']}")
        md.append(f"- Baseline topics: {case['baseline_topics']}")
        md.append(f"- Concept topics: {case['concept_topics']}")
        md.append("")
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return path
